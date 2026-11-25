"""Observer equivalence and CLI integrity on synthetic software fixtures.

These tests use the actual pinned KF/SOF and do not measure dataset accuracy.
"""
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import test_native_adapter as native_fixture
import test_identity_native as identity_fixture
import test_runner as runner_fixture
import cv2

from botsort_plus.config import PlusConfig
from botsort_plus.diagnostics.recorder import DiagnosticRecorder, SCHEMA, file_sha256, validate_capture_config
from botsort_plus.io import write_json
from botsort_plus.native_adapter import BoTSORTPlus
from botsort_plus.recovery import RecoveryConfig


class DiagnosticCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="diagnostic_capture_fixture_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        cv2.setNumThreads(1)

    def test_real_native_state_outputs_events_rng_and_inputs_are_exactly_equal(self):
        f = native_fixture
        config = PlusConfig(name="B_capture_test", adapter_mode="v2",
                            recovery=RecoveryConfig(enabled=True))
        sequence = [
            (f.detections(f.row()), [f.packets()]),
            (f.detections(), []),
            (f.detections(f.row(score=.3)), [{}]),
            (f.detections(f.row(score=.3)), [{}]),
            (f.detections(f.row(score=.3)), [{}]),
            (f.detections(f.row(), f.row(x=100)), [f.packets(), f.packets(f.E2)]),
            (f.detections(f.row(), f.row(x=100)), [f.packets(), f.packets(f.E2)]),
            (f.detections(f.row(score=.5), f.row(x=100, score=.1)), [{}, {}]),
            (f.detections(), []),
        ]
        input_state = identity_fixture.frozen(sequence)
        texture = np.random.default_rng(9123).integers(0, 256, (512, 640, 3), dtype=np.uint8)
        images = [texture.copy() for _ in sequence]
        # Motion in the final frames exercises non-identity SOF compensation.
        for index in range(6, len(images)):
            images[index] = cv2.warpAffine(texture, np.float32([[1, 0, index-5], [0, 1, 0]]), (640, 512))
        image_state = identity_fixture.frozen(images)
        runs, ransac_counts, rng_outputs = [], [], []
        for enabled in (False, True):
            cv2.setRNGSeed(147)
            tracker = BoTSORTPlus(config, with_reid=True, **f.PARAMS)
            if enabled:
                recorder = DiagnosticRecorder(self.root / "seq.jsonl.gz", sequence="seq",
                                              expected_frames=len(sequence))
                tracker.diagnostic_recorder = recorder
            captured = []
            with patch.object(cv2, "estimateAffinePartial2D", wraps=cv2.estimateAffinePartial2D) as fit:
                with patch.object(tracker.cmc, "apply", wraps=tracker.cmc.apply) as cmc:
                    for (dets, packets), image in zip(sequence, images):
                        output = tracker.update(dets, image, features=packets)
                        captured.append((output.copy(), identity_fixture.reference_state(tracker),
                            identity_fixture.frozen(tracker.last_events),
                            identity_fixture.frozen({"frame": tracker.cmc.prev_frame,
                                "points": tracker.cmc.prev_keypoints, "initialized": tracker.cmc.initialized})))
                    self.assertEqual(cmc.call_count, len(sequence))
                ransac_counts.append(fit.call_count)
            random = np.empty((16, 16), dtype=np.float32)
            cv2.randu(random, 0, 1)
            rng_outputs.append(random)
            runs.append(captured)
            if enabled:
                manifest = recorder.close()
        self.assertEqual(ransac_counts[0], ransac_counts[1])
        self.assertGreater(ransac_counts[0], 0)
        np.testing.assert_array_equal(*rng_outputs)
        self.assertEqual(identity_fixture.frozen(sequence), input_state)
        self.assertEqual(identity_fixture.frozen(images), image_state)
        for index, (reference, observed) in enumerate(zip(*runs), 1):
            with self.subTest(frame=index):
                np.testing.assert_array_equal(reference[0], observed[0])
                self.assertEqual(reference[1:], observed[1:])
        with gzip.open(recorder.path, "rt", encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream]
        self.assertEqual(manifest["status"], "COMPLETE")
        self.assertEqual(manifest["records"], len(sequence))
        self.assertEqual(manifest["data_sha256"], file_sha256(recorder.path))
        self.assertEqual(records[0]["schema"], SCHEMA)
        self.assertEqual(records[1]["detections"], [])
        self.assertFalse(records[2]["detections"][0]["rgb_valid"])
        self.assertIsNone(records[2]["detections"][0]["rgb_quality"])
        self.assertTrue(any(e["event"] == "recovered" for r in records for e in r["events"]))
        self.assertTrue(any(e["event"] == "recovery_check" for r in records for e in r["events"]))
        self.assertTrue(any(r["unconfirmed"] for r in records))
        for record, image in zip(records, images):
            self.assertEqual(record["image_sha256"], hashlib.sha256(image.tobytes()).hexdigest())
            self.assertFalse(record["gmc"]["native_inlier_mask_observed"])
            for track in record["after_gmc"]:
                self.assertEqual(np.shape(track["projected_covariance"]), (4, 4))
                self.assertEqual(np.shape(track["covariance"]), (8, 8))
        # Frozen snapshots are genuinely pre-update: a frame-2 observation has
        # not yet overwritten the preceding last_seen field.
        self.assertEqual(records[1]["after_gmc"][0]["last_seen"], 1)

    def test_missing_frames_cannot_produce_complete_manifest(self):
        recorder = DiagnosticRecorder(self.root / "incomplete.jsonl.gz", sequence="incomplete",
                                      expected_frames=2)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            recorder.close()
        manifest = json.loads(recorder.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "FAILED")
        self.assertFalse(recorder.path.exists())
        self.assertTrue(recorder.partial_path.exists())
        with self.assertRaises(FileExistsError):
            DiagnosticRecorder(recorder.path, sequence="incomplete", expected_frames=2)

    def test_unsupported_paths_fail_before_replay(self):
        for config in (PlusConfig(), PlusConfig(recovery=RecoveryConfig(enabled=True)),
                       PlusConfig(adapter_mode="v2", recovery=RecoveryConfig(enabled=True,
                                                                             output_prediction_frames=1))):
            with self.subTest(config=config.name):
                with self.assertRaisesRegex(ValueError, "requires v2"):
                    validate_capture_config(config, {"cmc_method": "sof"})
        config = PlusConfig(adapter_mode="v2", recovery=RecoveryConfig(enabled=True))
        with self.assertRaisesRegex(ValueError, "native SOF"):
            validate_capture_config(config, {"cmc_method": "ecc"})

    def test_cli_provenance_and_predictions_match_observer_off_run(self):
        fixture = runner_fixture.RunnerTests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        config = PlusConfig(name="B_diag_fixture", adapter_mode="v2",
                            recovery=RecoveryConfig(enabled=True))
        config_path = fixture.root / "B_diag_fixture.json"
        write_json(config_path, config.to_dict())
        fixture.configs[config.name] = config_path
        fixture.output = fixture.root / "reference"
        reference = fixture.invoke(config.name)
        fixture.assert_success(reference)
        fixture.output = fixture.root / "observed"
        observed = fixture.invoke(config.name, extra_args=("--diagnostics",))
        fixture.assert_success(observed)
        seq = runner_fixture.SEQ
        protocols = []
        for root in (fixture.root / "reference", fixture.root / "observed"):
            protocols.append(json.loads((root / "protocols" / (config.name + ".json")).read_text()))
        self.assertEqual(protocols[0]["output_prediction_hashes"], protocols[1]["output_prediction_hashes"])
        self.assertEqual(protocols[0]["decoded_image_hashes"], protocols[1]["decoded_image_hashes"])
        self.assertEqual(protocols[1]["diagnostics"]["status"], "COMPLETE")
        info = protocols[1]["diagnostics"]["traces"][seq]
        self.assertEqual(info["records"], 6)
        self.assertEqual(info["manifest_sha256"], file_sha256(fixture.output / info["manifest_path"]))
        self.assertEqual(info["data_sha256"], file_sha256(fixture.output / info["path"]))
        self.assertIn("botsort_plus/diagnostics/recorder.py", protocols[1]["implementation_sha256"])
        with gzip.open(fixture.output / info["path"], "rt", encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream]
        self.assertEqual([r["frame"] for r in records], list(range(1, 7)))
        events = [json.loads(line) for line in (fixture.output / "events" / config.name /
                                               (seq + ".jsonl")).read_text().splitlines()]
        for record in records:
            for event in record["events"]:
                self.assertIn({"sequence": seq, "frame": record["frame"], **event}, events)


if __name__ == "__main__":
    unittest.main()
