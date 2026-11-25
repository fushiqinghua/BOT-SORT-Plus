"""Exercise the delivered four-diagnostic CLI with actual pinned BoxMOT.

The tiny JPEGs, labels and features are synthetic SOFTWARE fixtures. None of
these assertions is a measurement of VT-Tiny-MOT tracking accuracy.
"""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
import zipfile

import test_runner as fixtures
from botsort_plus.io import write_json
from tools.run_botsort_diagnostics import frozen_config


REPO = Path(__file__).resolve().parents[2]
DRIVER = REPO / "tools/run_botsort_diagnostics.py"
TRACKER = "B_early_moderate"


class DiagnosticSuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = fixtures.RunnerTests()
        cls.fixture.setUp()
        cls.addClassCleanup(cls.fixture.doCleanups)
        f = cls.fixture
        # Explicitly declare this as a subset: the evaluator correctly refuses
        # to call a one-sequence software fixture a complete full-30 experiment.
        protocol = json.loads((f.baseline / "protocol.json").read_text())
        protocol["sequences"][fixtures.SEQ + "_NOT_SELECTED"] = protocol["sequences"][fixtures.SEQ]
        write_json(f.baseline / "protocol.json", protocol)
        cls.gt_root = f.root / "truth"
        gt_dir = cls.gt_root / fixtures.SEQ / "gt"
        gt_dir.mkdir(parents=True)
        truth = []
        for frame in range(1, 7):
            truth.append(f"{frame},1,50,50,4,4,1,1,1\n")
            if frame >= 3:
                truth.append(f"{frame},2,150,50,4,4,1,1,1\n")
        (gt_dir / "gt.txt").write_text("".join(truth), encoding="utf-8")
        cls.output = f.root / "diagnostic_suite"
        cls.settings = f.root / "settings.json"
        write_json(cls.settings, {
            "schema": "botsort_diagnostics_settings_v1",
            "baseline_root": str(f.baseline), "features_dir": str(f.features),
            "images_root": str(f.images), "gt_root": str(cls.gt_root),
            "output_root": str(cls.output), "runtime_root": None,
            "sequences": [fixtures.SEQ], "max_review": 12,
        })
        cls.run_result = cls.invoke("run")
        if cls.run_result.returncode:
            raise AssertionError(cls.run_result.stdout + "\n" + cls.run_result.stderr)

    @classmethod
    def invoke(cls, action, *extra):
        env = os.environ.copy()
        env.pop("TRACKING_RUNTIME_ROOT", None)
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["PYTHONPATH"] = str(REPO)
        return subprocess.run(
            [sys.executable, "-X", "utf8", str(DRIVER), action,
             "--settings", str(cls.settings), *map(str, extra)],
            cwd=REPO, env=env, text=True, encoding="utf-8", capture_output=True,
            timeout=120,
        )

    def test_complete_pipeline_and_censored_future(self):
        experiment = json.loads((self.output / "experiment.json").read_text())
        self.assertEqual(experiment["status"], "COMPLETE")
        self.assertFalse(experiment["prediction_reads_gt"])
        self.assertEqual(experiment["stages"]["parity"], "PASS")
        self.assertEqual(experiment["stages"]["D4_natural_blur_labels"], "PENDING_MANUAL_REVIEW")
        verification = json.loads((self.output / "verification.json").read_text())
        self.assertTrue(verification["sequences"][fixtures.SEQ]["byte_identical"])
        for name in ("detection_audit.csv", "track_error.csv", "frame_error.csv",
                     "recovery_events.csv", "recovery_windows.csv", "recovery_checks.csv",
                     "summary.json", "REPORT.md"):
            self.assertTrue((self.output / "analysis" / name).is_file(), name)
        for name in ("observation_quality.csv", "quality_strata.csv", "manual_review.csv",
                     "manual_label_summary.json", "summary.json"):
            self.assertTrue((self.output / "blur" / name).is_file(), name)
        with (self.output / "analysis/recovery_windows.csv").open(encoding="utf-8-sig") as stream:
            windows = list(csv.DictReader(stream))
        self.assertTrue(windows, "Fixture must actually exercise B recovery")
        self.assertTrue(all(r["right_censored"].lower() == "true" for r in windows))
        self.assertTrue(all(not r["stable_all_frames"] for r in windows))
        with (self.output / "blur/manual_review.csv").open(encoding="utf-8-sig") as stream:
            review = list(csv.DictReader(stream))
        self.assertTrue(review)
        self.assertTrue(all(r["label"] == "unreviewed" for r in review))
        self.assertTrue(list((self.output / "blur/contact_sheets").glob("*.png")))

    def test_input_identity_mismatch_rejects_and_original_outputs_survive(self):
        path = self.output / "observed/protocols" / f"{TRACKER}.json"
        original = path.read_bytes()
        try:
            protocol = json.loads(original)
            protocol["decoded_image_hashes"][fixtures.SEQ] = "0" * 64
            write_json(path, protocol)
            result = self.invoke("verify")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("decoded_image_hashes", result.stderr)
        finally:
            path.write_bytes(original)
        self.assertEqual(self.invoke("verify").returncode, 0)

    def test_missing_trace_is_rejected_by_verify(self):
        trace = self.output / "observed/diagnostics" / TRACKER / f"{fixtures.SEQ}.jsonl.gz"
        temporary = trace.with_name(trace.name + ".saved")
        trace.rename(temporary)
        try:
            result = self.invoke("verify")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("jsonl.gz", result.stderr)
        finally:
            temporary.rename(trace)

    def test_nondefault_recovery_threshold_cannot_change_frozen_reference(self):
        changed = self.fixture.root / "changed_config.json"
        config = json.loads((REPO / "configs/botsort_plus/B_early_moderate.json").read_text())
        config["recovery"]["max_cost"] = .123
        write_json(changed, config)
        with self.assertRaisesRegex(ValueError, "complete B_early_moderate configuration"):
            frozen_config(changed)

    def test_bad_gt_is_rejected_during_offline_preflight(self):
        path = self.gt_root / fixtures.SEQ / "gt/gt.txt"
        original = path.read_bytes()
        path.write_text("99,1,50,50,4,4,1,1,1\n", encoding="utf-8")
        try:
            result = self.invoke("preflight")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("GT frame outside video", result.stderr)
        finally:
            path.write_bytes(original)

    def test_overwrite_protection_and_evidence_scope(self):
        result = self.invoke("run")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("new output_root", result.stderr)
        archive = self.output.parent / f"{self.output.name}_evidence.zip"
        with zipfile.ZipFile(archive) as packed:
            names = packed.namelist()
            self.assertIn("EVIDENCE_MANIFEST.json", names)
            self.assertIn("verification.json", names)
            self.assertFalse(any(n.endswith(".jsonl.gz") for n in names))
            self.assertFalse(any(n.endswith((".npz", "gt.txt", ".pt")) for n in names))
        full = self.output.parent / "with_traces.zip"
        result = self.invoke("package", "--include-traces", "--output", full)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        with zipfile.ZipFile(full) as packed:
            self.assertTrue(any(n.endswith(".jsonl.gz") for n in packed.namelist()))


if __name__ == "__main__":
    unittest.main()
