"""C2 real CLI tests with explicit SOFTWARE-only images and cached embeddings.

No model is trained/run and no benchmark accuracy is measured by these tests.
The fixture contains no GT; prediction keeps its normal no-GT audit hook.
"""
import csv
import gzip
import json
import shutil
import subprocess
import sys
import unittest

import numpy as np

import test_runner as fixture_module
from botsort_plus.config import PlusConfig
from botsort_plus.io import detection_fingerprint, sha256, write_json


class IdentityRunnerTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture_module.RunnerTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        # Birth, stable history, native unconfirmed, low, missing frame, B
        # recovery, native update after recovery, and another unconfirmed ID.
        a = [50., 50., 70., 70., .95, 0.]
        low = [50., 50., 70., 70., .3, 0.]
        b = [150., 50., 170., 70., .95, 0.]
        c = [250., 50., 270., 70., .95, 0.]
        frames = [[a], [a], [a, b], [a, b], [low, b], [],
                  [low], [low], [low], [a, c], [a, c], [low, c]]
        records, embedding_rows = [], []
        near = [float(np.cos(.1)), float(np.sin(.1))]
        for frame, dets in enumerate(frames, 1):
            image = f"frame_{frame:06d}.jpg"
            if not (f.images / image).exists():
                shutil.copyfile(f.images / "frame_000006.jpg", f.images / image)
            records.append({"frame": frame, "image": image, "detections": dets})
            for detection in dets:
                embedding_rows.append(([0., 1.] if detection[0] >= 100 or frame == 5
                                       else near if frame >= 10 else [1., 0.]))
        f.cache["frames"] = records
        with gzip.open(f.cache_path, "wt", encoding="utf-8") as stream:
            json.dump(f.cache, stream)
        protocol = json.loads((f.baseline / "protocol.json").read_text())
        protocol["sequences"][fixture_module.SEQ] = [
            {"frame": r["frame"], "image": r["image"]} for r in records]
        write_json(f.baseline / "protocol.json", protocol)
        offsets = np.r_[0, np.cumsum([len(r["detections"]) for r in records])]
        rgb = np.asarray(embedding_rows, np.float32)
        np.savez_compressed(f.feature_path, rgb=rgb, rgb_valid=np.ones(len(rgb), bool),
                            rgb_quality=np.ones(len(rgb), np.float32),
                            frame_ids=np.arange(1, 13, dtype=np.int64), offsets=offsets.astype(np.int64))
        metadata = json.loads(f.feature_path.with_suffix(".json").read_text())
        metadata.update(features_sha256=sha256(f.feature_path),
                        detection_fingerprint=detection_fingerprint(f.cache))
        metadata["encoders"]["rgb"]["output_dim"] = 2
        write_json(f.feature_path.with_suffix(".json"), metadata)

    def test_all_seven_configs_preflight_replay_protocols_and_shadow_audit(self):
        f, repo, seq = self.fixture, fixture_module.REPO, fixture_module.SEQ
        files = ["12_baseline_ref.json", "13_B_ref.json", "27_B_C2_shadow.json",
                 "28_B_C2_memory.json", "29_B_C2_association.json", "30_B_C2.json", "31_C2.json"]
        outputs, all_events = {}, {}
        for filename in files:
            path = repo / "configs/botsort_plus" / filename
            cfg = PlusConfig.load(path)
            f.configs[cfg.name] = path
        preflight = f.invoke("B_C2", preflight=True)
        self.assertEqual(preflight.returncode, 0, preflight.stdout + preflight.stderr)
        prepared = json.loads(preflight.stdout)
        self.assertEqual(prepared["status"], "READY")
        self.assertEqual(prepared["missing_count"], 0)
        self.assertEqual(set(prepared["encoder_contracts"]), {"rgb"})
        self.assertFalse((f.output / "protocols/B_C2.json").exists())
        self.assertFalse(list(f.root.rglob("gt.txt")))
        for filename in files:
            cfg = PlusConfig.load(repo / "configs/botsort_plus" / filename)
            result = f.invoke(cfg.name)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            protocol = json.loads((f.output / "protocols" / (cfg.name + ".json")).read_text())
            self.assertEqual(protocol["status"], "COMPLETE")
            self.assertEqual(protocol["implementation_version"], "0.3.0")
            self.assertEqual(protocol["config"], cfg.to_dict())
            self.assertEqual(protocol["sequences"], {seq: 12})
            self.assertTrue(protocol["causal"])
            self.assertEqual(protocol["backfilled_frames"], 0)
            self.assertFalse(protocol["interpolation"])
            for source in ("identity_reliability.py", "identity_adapter.py", "adapter_v2.py"):
                relative = "botsort_plus/" + source
                self.assertEqual(protocol["implementation_sha256"][relative], sha256(repo / relative))
            prediction = f.output / "trackers" / cfg.name / "data" / (seq + ".txt")
            self.assertEqual(protocol["output_prediction_hashes"][seq], sha256(prediction))
            self.assertEqual(protocol["feature_hashes"][seq]["features"], sha256(f.feature_path))
            outputs[cfg.name] = prediction.read_bytes()
            with prediction.open() as stream:
                rows = [list(map(float, row)) for row in csv.reader(stream)]
            self.assertEqual(len(rows), len({(row[0], row[1]) for row in rows}))
            self.assertEqual([row[0] for row in rows], sorted(row[0] for row in rows))
            events = [json.loads(line) for line in
                      (f.output / "events" / cfg.name / (seq + ".jsonl")).read_text().splitlines()]
            all_events[cfg.name] = events
            if cfg.identity.enabled:
                decisions = [e for e in events if e["event"] == "identity_decision"]
                self.assertTrue(decisions)
                self.assertTrue(any(e["stage"] == "low" for e in decisions))
                self.assertTrue(any(e["stage"] == "unconfirmed" for e in decisions))
                self.assertTrue(any(e["reason"] == "history_conflict" for e in decisions))
                self.assertTrue(all(e["observe_only"] == cfg.identity.observe_only for e in decisions))
                self.assertTrue(all(e["memory_gate_applied"] ==
                                    (cfg.identity.memory_enabled and not cfg.identity.observe_only)
                                    for e in decisions))
                if cfg.recovery.enabled:
                    self.assertTrue(any(e["reason"] == "recovered_quarantine" for e in decisions))
        self.assertEqual(outputs["B_ref"], outputs["B_C2_shadow"])
        # Both shared algorithms actually exercised B recovery in this fixture.
        for name in ("B_ref", "B_C2_shadow", "B_C2"):
            recovered = [e for e in all_events[name] if e["event"] == "recovered"]
            self.assertEqual([(e["frame"], e["track_id"]) for e in recovered], [(9, 1)])
        audit_path = f.root / "identity_shadow_audit.json"
        audited = subprocess.run([
            sys.executable, str(repo / "tools/audit_botsort_plus_a2.py"),
            "--run-dir", str(f.output), "--trackers", "B_ref", "B_C2_shadow",
            "--compare-pair", "B_ref", "B_C2_shadow", "--output", str(audit_path)],
            text=True, capture_output=True, timeout=30)
        self.assertEqual(audited.returncode, 0, audited.stdout + audited.stderr)
        report = json.loads(audit_path.read_text())
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["comparisons"][0]["identical_sequences"], [seq])
        identity_audit_path = f.root / "identity_mechanism_audit.json"
        checked = subprocess.run([
            sys.executable, str(repo / "tools/audit_botsort_plus_identity.py"),
            "--run-dir", str(f.output), "--trackers", *outputs,
            "--compare-pair", "B_ref", "B_C2_shadow", "--output", str(identity_audit_path)],
            text=True, capture_output=True, timeout=30)
        self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
        identity_report = json.loads(identity_audit_path.read_text())
        self.assertEqual(identity_report["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
