"""Full CLI + cached-feature + SOF replay of labelled SOFTWARE fixtures only."""
import csv
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

import numpy as np

import test_runner as fixture_module
from botsort_plus.config import PlusConfig
from botsort_plus import __version__
from botsort_plus.io import detection_fingerprint, sha256, write_json


class A2RunnerTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture_module.RunnerTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        for frame in (7, 8):
            image = f"frame_{frame:06d}.jpg"
            shutil.copyfile(f.images / "frame_000006.jpg", f.images / image)
            f.cache["frames"].append({"frame": frame, "image": image,
                                      "detections": [[50, 50, 54, 54, .3, 0],
                                                     [150, 50, 154, 54, .3, 0]]})
        with gzip.open(f.cache_path, "wt", encoding="utf-8") as stream:
            json.dump(f.cache, stream)
        protocol = json.loads((f.baseline / "protocol.json").read_text())
        protocol["sequences"][fixture_module.SEQ] = [
            {"frame": r["frame"], "image": r["image"]} for r in f.cache["frames"]]
        write_json(f.baseline / "protocol.json", protocol)
        offsets = np.r_[0, np.cumsum([len(r["detections"]) for r in f.cache["frames"]])]
        n = int(offsets[-1])
        rgb, valid, quality = np.zeros((n, 2), np.float32), np.zeros(n, bool), np.zeros(n, np.float32)
        rgb[0], valid[0], quality[0] = [1., 0.], True, 1.
        np.savez_compressed(f.feature_path, rgb=rgb, rgb_valid=valid, rgb_quality=quality,
                            frame_ids=np.arange(1, 9, dtype=np.int64), offsets=offsets.astype(np.int64))
        metadata = json.loads(f.feature_path.with_suffix(".json").read_text())
        metadata.update(features_sha256=sha256(f.feature_path),
                        detection_fingerprint=detection_fingerprint(f.cache))
        write_json(f.feature_path.with_suffix(".json"), metadata)

    def test_real_cli_two_stage_recovery_shadow_and_audit(self):
        f, repo, seq = self.fixture, fixture_module.REPO, fixture_module.SEQ
        configs = ["12_baseline_ref.json", "13_B_ref.json", "16_A2.json", "19_A2B.json",
                   "25_A2_shadow.json", "26_A2B_shadow.json"]
        results = {}
        for filename in configs:
            path = repo / "configs/botsort_plus" / filename
            cfg = PlusConfig.load(path)
            f.configs[cfg.name] = path
            result = f.invoke(cfg.name)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            prediction = f.output / "trackers" / cfg.name / "data" / (seq + ".txt")
            with prediction.open() as stream:
                results[cfg.name] = [(int(float(r[0])), int(float(r[1]))) for r in csv.reader(stream)]
            protocol = json.loads((f.output / "protocols" / (cfg.name + ".json")).read_text())
            self.assertEqual(protocol["implementation_version"], __version__)
            self.assertEqual(protocol["implementation_sha256"]["botsort_plus/birth_policy.py"],
                             sha256(repo / "botsort_plus/birth_policy.py"))
            self.assertEqual(protocol["status"], "COMPLETE")
            self.assertEqual(protocol["backfilled_frames"], 0)
        self.assertEqual(results["baseline_ref"], [(1, 1)])
        self.assertEqual(results["A2_shadow"], results["baseline_ref"])
        self.assertEqual(results["B_ref"], [(1, 1), (5, 1), (6, 1), (7, 1), (8, 1)])
        self.assertEqual(results["A2B_shadow"], results["B_ref"])
        self.assertEqual(results["A2"], [(1, 1), (7, 2), (8, 2)])
        self.assertEqual(sorted(results["A2B"]),
                         [(1, 1), (5, 1), (6, 1), (7, 1), (7, 2), (8, 1), (8, 2)])
        events = [json.loads(line) for line in (f.output / "events/A2B" / (seq + ".jsonl")).read_text().splitlines()]
        committed = [e for e in events if e["event"] == "birth_committed"]
        self.assertEqual([(e["frame"], e["track_id"]) for e in committed], [(7, 2)])
        self.assertEqual(committed[0]["support"]["armed_frames"], [3, 4, 5])
        self.assertEqual(committed[0]["support"]["verify_frames"], [6, 7])
        lifetime = {e["track_id"]: e for e in events if e["event"] == "track_lifetime"}
        self.assertEqual(lifetime[1]["recovery_count"], 1)
        self.assertEqual(lifetime[2]["output_frames"], 2)
        self.assertEqual(lifetime[2]["first_emitted"], 7)
        output = f.root / "a2_audit.json"
        command = [sys.executable, str(repo / "tools/audit_botsort_plus_a2.py"),
                   "--run-dir", str(f.output), "--trackers", *results,
                   "--compare-pair", "baseline_ref", "A2_shadow",
                   "--compare-pair", "B_ref", "A2B_shadow", "--output", str(output)]
        audited = subprocess.run(command, text=True, capture_output=True, timeout=30)
        self.assertEqual(audited.returncode, 0, audited.stdout + audited.stderr)
        self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
