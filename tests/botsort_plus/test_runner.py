"""Actual CLI replay smoke/regression tests using synthetic software fixtures.

These JPEGs and cached unit vectors are deliberately generated test inputs. They
do not represent extracted model features or any VT-Tiny-MOT accuracy result.
Subprocess isolation keeps prediction's no-GT audit hook local to that process.
"""
import csv
import gzip
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

for _dependency in ("boxmot", "torch", "cv2"):
    if importlib.util.find_spec(_dependency) is None:
        raise unittest.SkipTest(f"End-to-end replay tests require {_dependency}")

import cv2
from botsort_plus.candidates import BirthConfig
from botsort_plus.config import PlusConfig
from botsort_plus.io import detection_fingerprint, sha256, write_json
from botsort_plus.recovery import RecoveryConfig


REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "tools/run_botsort_plus.py"
SEQ = "SYNTHETIC_SOFTWARE_TEST_ONLY"
DETECTOR_SHA, RGB_SHA = "a" * 64, "b" * 64


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="botsort_plus_synthetic_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.baseline = self.root / "baseline"
        self.images = self.root / "images"
        self.features = self.root / "features"
        self.output = self.root / "output"
        for directory in (self.baseline / "detections", self.images, self.features):
            directory.mkdir(parents=True)
        texture = np.random.default_rng(8831).integers(0, 256, (512, 640, 3), dtype=np.uint8)
        records = []
        for frame in range(1, 7):
            image = f"frame_{frame:06d}.jpg"
            self.assertTrue(cv2.imwrite(str(self.images / image), texture))
            if frame == 1:
                dets = [[50, 50, 54, 54, .95, 0]]
            elif frame == 2:
                dets = []
            else:
                dets = [[50, 50, 54, 54, .3, 0], [150, 50, 154, 54, .3, 0]]
            records.append({"frame": frame, "image": image, "detections": dets})
        self.cache = {"model_sha256": DETECTOR_SHA, "inference_wall_seconds": 0.0, "frames": records,
                      "test_fixture": "Synthetic software test; not a dataset or model result"}
        self.cache_path = self.baseline / "detections" / (SEQ + ".json.gz")
        with gzip.open(self.cache_path, "wt", encoding="utf-8") as stream:
            json.dump(self.cache, stream)
        native = dict(det_thresh=.5, track_high_thresh=.5, track_low_thresh=.1, new_track_thresh=.6,
                      match_thresh=.8, track_buffer=60, frame_rate=15, with_reid=True,
                      cmc_method="sof", proximity_thresh=.5, appearance_thresh=.4,
                      fuse_first_associate=False)
        write_json(self.baseline / "protocol.json", {
            "weight_sha256": DETECTOR_SHA, "reid_sha256": RGB_SHA,
            "sequences": {SEQ: [{"frame": r["frame"], "image": r["image"]} for r in records]},
            "trackers": {"botsort_reid": native}, "fixture_only": True,
        })
        offsets = np.r_[0, np.cumsum([len(r["detections"]) for r in records])]
        n = int(offsets[-1])
        rgb = np.zeros((n, 2), dtype=np.float32)
        valid = np.zeros(n, dtype=bool)
        quality = np.zeros(n, dtype=np.float32)
        rgb[0], valid[0], quality[0] = [1., 0.], True, 1.
        self.feature_path = self.features / (SEQ + ".npz")
        np.savez_compressed(self.feature_path, rgb=rgb, rgb_valid=valid, rgb_quality=quality,
                            frame_ids=np.arange(1, 7, dtype=np.int64), offsets=offsets.astype(np.int64))
        write_json(self.feature_path.with_suffix(".json"), {
            "schema": "botsort_plus_features_v1", "features_sha256": sha256(self.feature_path),
            "detection_fingerprint": detection_fingerprint(self.cache),
            "encoders": {"rgb": {"sha256": RGB_SHA, "name": "synthetic_unit_vector_fixture_only"}},
            "extraction_seconds": 0.0, "synthetic_software_test_only": True,
        })
        self.configs = {}
        for config in (PlusConfig(name="baseline"), PlusConfig(name="ab", birth=BirthConfig(enabled=True),
                                                              recovery=RecoveryConfig(enabled=True))):
            path = self.root / (config.name + ".json")
            write_json(path, config.to_dict())
            self.configs[config.name] = path

    def invoke(self, name="ab", preflight=False, extra_args=()):
        command = [sys.executable, str(RUNNER), "--baseline-root", str(self.baseline),
                   "--features-dir", str(self.features), "--images-root", str(self.images),
                   "--config", str(self.configs[name]), "--output-dir", str(self.output), "--trace"]
        if preflight:
            command.append("--preflight")
        command.extend(extra_args)
        env = os.environ.copy()
        env.pop("TRACKING_RUNTIME_ROOT", None)
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["PYTHONPATH"] = str(REPO)
        return subprocess.run(command, text=True, capture_output=True, cwd=REPO, env=env, timeout=45)

    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, msg=result.stdout + "\n" + result.stderr)

    def test_same_weights_different_detections_are_not_frozen_baseline(self):
        alternate=self.root/"alternate_detections"
        alternate.mkdir()
        changed=json.loads(json.dumps(self.cache))
        changed["frames"][0]["detections"][0][0]+=1
        with gzip.open(alternate/(SEQ+".json.gz"),"wt") as stream:
            json.dump(changed,stream)
        result=self.invoke(preflight=True,extra_args=("--detections-dir",str(alternate)))
        self.assertNotEqual(result.returncode,0)
        self.assertIn("differs from frozen baseline",result.stderr)

    def test_lower_high_gate_checks_real_rgb_for_new_high_rows(self):
        data=json.loads(self.configs["ab"].read_text())
        data["native_overrides"]={"track_high_thresh":.1}
        write_json(self.configs["ab"],data)
        result=self.invoke(preflight=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn("missing high-score",result.stderr)

    def test_mixed_thermal_encoders_across_sequences_are_rejected(self):
        with np.load(self.feature_path) as archive:
            arrays={key:archive[key].copy() for key in archive.files}
        n=len(arrays["rgb"])
        arrays.update(thermal=np.tile(np.array([[1.,0.]],np.float32),(n,1)),
                      thermal_valid=np.ones(n,dtype=bool),thermal_quality=np.ones(n,dtype=np.float32))
        original_meta=json.loads(self.feature_path.with_suffix(".json").read_text())
        other=SEQ+"_2"
        with gzip.open(self.baseline/"detections"/(other+".json.gz"),"wt") as stream:
            json.dump(self.cache,stream)
        protocol=json.loads((self.baseline/"protocol.json").read_text())
        protocol["sequences"][other]=protocol["sequences"][SEQ]
        write_json(self.baseline/"protocol.json",protocol)
        for name,digest in ((SEQ,"c"*64),(other,"d"*64)):
            feature_path=self.features/(name+".npz")
            np.savez_compressed(feature_path,**arrays)
            meta=json.loads(json.dumps(original_meta))
            meta["features_sha256"]=sha256(feature_path)
            meta["encoders"]["thermal"]={"sha256":digest,"output_dim":2}
            meta["spatial_mapping"]={"mode":"aligned","declared_by_user":True}
            write_json(feature_path.with_suffix(".json"),meta)
        result=self.invoke(preflight=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn("Mixed thermal encoder",result.stderr)

    def test_complete_baseline_and_ab_with_causal_mot_and_trace(self):
        self.assert_success(self.invoke("baseline"))
        self.assert_success(self.invoke("ab"))
        records = {}
        for name in ("baseline", "ab"):
            protocol = json.loads((self.output / "protocols" / (name + ".json")).read_text())
            self.assertEqual(protocol["status"], "COMPLETE")
            self.assertEqual(protocol["sequences"], {SEQ: 6})
            self.assertEqual(protocol["backfilled_frames"], 0)
            self.assertTrue(protocol["causal"])
            dest = self.output / "trackers" / name / "data" / (SEQ + ".txt")
            self.assertEqual(protocol["output_prediction_hashes"][SEQ], sha256(dest))
            with dest.open() as stream:
                records[name] = [list(map(float, r)) for r in csv.reader(stream)]
            timing = json.loads((self.output / "timing" / name / (SEQ + ".json")).read_text())
            self.assertEqual(timing["frames"], 6)
            self.assertEqual(timing["output_rows"], len(records[name]))
        self.assertEqual([(int(r[0]), int(r[1])) for r in records["baseline"]], [(1, 1)])
        self.assertEqual(sorted((int(r[0]), int(r[1])) for r in records["ab"]),
                         [(1, 1), (5, 1), (5, 2), (6, 1), (6, 2)])
        # Make the causal claim explicit, independent of native output order.
        self.assertEqual({int(r[0]) for r in records["ab"]}, {1, 5, 6})
        self.assertEqual(len(records["ab"]), len({(r[0], r[1]) for r in records["ab"]}))
        events = [json.loads(line) for line in
                  (self.output / "events" / "ab" / (SEQ + ".jsonl")).read_text().splitlines()]
        recovered = [e for e in events if e["event"] == "recovered"]
        births = [e for e in events if e["event"] == "weak_birth"]
        self.assertEqual([(e["frame"], e["track_id"]) for e in recovered], [(5, 1)])
        self.assertEqual([(e["frame"], e["track_id"]) for e in births], [(5, 2)])

    def test_missing_image_preflight_exits_two(self):
        (self.images / "frame_000003.jpg").unlink()
        result = self.invoke(preflight=True)
        self.assertEqual(result.returncode, 2, result.stderr)
        protocol = json.loads(result.stdout)
        self.assertEqual(protocol["status"], "MISSING_INPUTS")
        self.assertGreater(protocol["missing_count"], 0)
        self.assertFalse((self.output / "protocols" / "ab.json").exists())

    def test_missing_feature_preflight_exits_two(self):
        self.feature_path.unlink()
        result = self.invoke(preflight=True)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "MISSING_INPUTS")
        self.assertFalse((self.output / "protocols" / "ab.json").exists())

    def test_stale_checksum_fails_before_complete_manifest(self):
        with self.feature_path.open("ab") as stream:
            stream.write(b"synthetic-cache-corruption")
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("checksum mismatch", result.stderr)
        self.assertFalse((self.output / "protocols" / "ab.json").exists())

    def test_existing_run_is_not_overwritten(self):
        self.assert_success(self.invoke("baseline"))
        prediction = self.output / "trackers" / "baseline" / "data" / (SEQ + ".txt")
        protocol = self.output / "protocols" / "baseline.json"
        prior = (sha256(prediction), sha256(protocol))
        result = self.invoke("baseline")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("already exists", result.stderr)
        self.assertEqual((sha256(prediction), sha256(protocol)), prior)

    def test_partially_readable_input_records_failed_never_complete(self):
        # Preflight sees an existing path, but original-image dimensions fail
        # after frame 1 was processed. A partial .tmp output is not a result.
        cv2.imwrite(str(self.images / "frame_000003.jpg"), np.zeros((128, 128, 3), np.uint8))
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        protocol = json.loads((self.output / "protocols" / "ab.json").read_text())
        self.assertEqual(protocol["status"], "FAILED")
        self.assertNotEqual(protocol["status"], "COMPLETE")
        self.assertIn("512x640", protocol["error"])
        self.assertFalse((self.output / "trackers" / "ab" / "data" / (SEQ + ".txt")).exists())


if __name__ == "__main__":
    unittest.main()
