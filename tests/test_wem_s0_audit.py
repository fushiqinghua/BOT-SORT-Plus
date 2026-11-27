"""Mechanism tests for S0 inventory; these are not real-data W validation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from tools.research.wem_v01 import cache_audit as audit


class S0AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.inputs = self.root / "detections"
        self.inputs.mkdir()
        self.output = self.root / "audit"

    def cache(self, name="sample", frames=None):
        path = self.inputs / f"{name}.json"
        path.write_text(json.dumps({"frames": frames if frames is not None else []}), encoding="utf-8")
        return path

    @staticmethod
    def row(score):
        return [10, 20, 18, 26, score, 3]

    def run_audit(self, **kwargs):
        return audit.audit_directory(
            self.inputs, self.output, low_thresh=.1, high_thresh=.5, **kwargs,
        )

    def test_float32_boundary_and_high_hole_are_distinct(self):
        self.cache(frames=[{"frame": 1, "detections": [
            self.row(0), self.row(.05), self.row(.09999999),
            self.row(.1), self.row(.2), self.row(.5), self.row(.50000006),
        ]}])
        result = self.run_audit()
        counts = result["totals"]
        self.assertEqual((counts["s0"], counts["native_low"], counts["native_high"]), (3, 2, 1))
        self.assertEqual(counts["high_threshold_gap"], 1)
        # float32(.1) is > Python float(.1) after native hstack promotion,
        # yet equals the exporter's float32 threshold and lacks policy coverage.
        self.assertEqual(counts["native_eligible_reid_policy_uncovered"], 1)
        self.assertEqual(result["status"], "INVENTORY_ONLY_NOT_W_OPPORTUNITY")

    def test_empty_frames_and_gaps_do_not_inflate_frame_count(self):
        self.cache(frames=[
            {"frame": 4, "detections": []},
            {"frame": 7, "detections": [self.row(.05)]},
        ])
        result = self.run_audit()
        self.assertEqual(result["totals"]["frames"], 2)
        self.assertEqual(result["totals"]["frames_with_detections"], 1)
        self.assertEqual(result["frame_gaps"][0]["ranges"], [[5, 6]])
        self.assertEqual(result["frame_gaps"][0]["missing_frame_ids_inside_span"], 2)

    def test_empty_sequence_is_explicit_zero(self):
        self.cache()
        result = self.run_audit()
        self.assertEqual(result["totals"]["frames"], 0)
        self.assertEqual(result["totals"]["detections"], 0)
        self.assertEqual((self.output / "s0_candidates.jsonl").read_text(), "")

    def test_original_row_identity_and_actual_features_unknown(self):
        source = self.cache(name="船只_01", frames=[{
            "frame": 2, "image": "rgb/000002.jpg",
            "detections": [self.row(.8), self.row(.07), self.row(.06)],
        }])
        before = source.read_bytes()
        self.run_audit()
        rows = [json.loads(line) for line in (self.output / "s0_candidates.jsonl").read_text().splitlines()]
        self.assertEqual([row["original_row"] for row in rows], [1, 2])
        self.assertEqual(rows[0]["observation_key"], {
            "sequence": "船只_01", "frame_id": 2, "source": "native_cache", "original_row": 1,
        })
        self.assertIsNone(rows[0]["actual_rgb_feature_valid"])
        self.assertEqual(rows[0]["owner_or_reservation"], "NOT_AUDITED")
        self.assertEqual(before, source.read_bytes())
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(manifest["inputs"][0]["sha256"], hashlib.sha256(before).hexdigest())
        self.assertFalse(manifest["tracker_executed"])

    def test_bad_later_sequence_leaves_no_partial_output(self):
        self.cache("a", [{"frame": 1, "detections": [self.row(.07)]}])
        self.cache("z", [{"frame": 1, "detections": [[0, 0, 1, 1, .05]]}])
        with self.assertRaises(audit.AuditError):
            self.run_audit()
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".audit.staging-*")), [])

    def test_duplicate_or_unsorted_frame_ids_rejected(self):
        for ids in ([1, 1], [2, 1]):
            with self.subTest(ids=ids):
                self.cache(frames=[{"frame": i, "detections": []} for i in ids])
                with self.assertRaises(audit.AuditError):
                    self.run_audit()
                self.assertFalse(self.output.exists())

    def test_invalid_numbers_boxes_and_rows_rejected(self):
        invalid_rows = [
            [0, 0, 0, 1, .05, 0],
            [0, 0, 1, 1, float("nan"), 0],
            [0, 0, 1e100, 1, .05, 0],
            [0, 0, 1, 1, True, 0],
            [0, 0, 1, 1, -0.1, 0],
            [0, 0, 1, 1, .05, 1.5],
            [0, 0, 1, 1, .05],
        ]
        for row in invalid_rows:
            with self.subTest(row=row):
                self.cache(frames=[{"frame": 1, "detections": [row]}])
                with self.assertRaises(audit.AuditError):
                    self.run_audit()
                self.assertFalse(self.output.exists())

    def test_nonempty_output_is_never_overwritten(self):
        self.cache()
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("existing experiment", encoding="utf-8")
        with self.assertRaises(audit.AuditError):
            self.run_audit()
        self.assertEqual(marker.read_text(), "existing experiment")

    def test_input_output_overlap_rejected(self):
        self.cache()
        for output in (self.inputs, self.inputs / "audit", self.root):
            with self.subTest(output=output):
                with self.assertRaises(audit.AuditError):
                    audit.audit_directory(self.inputs, output, low_thresh=.1, high_thresh=.5)

    def test_subset_and_missing_or_duplicate_selections(self):
        self.cache("a", [{"frame": 1, "detections": [self.row(.07)]}])
        self.cache("b", [{"frame": 1, "detections": [self.row(.9)]}])
        for selected in (["missing"], ["a", "a"]):
            with self.subTest(selected=selected):
                with self.assertRaises(audit.AuditError):
                    self.run_audit(sequences=selected)
        result = self.run_audit(sequences=["a"])
        self.assertEqual(result["sequences"], 1)
        self.assertEqual(result["totals"]["detections"], 1)

    def test_input_change_between_passes_is_rejected(self):
        path = self.cache(frames=[{"frame": 1, "detections": [self.row(.07)]}])
        original = audit._inventory
        calls = 0

        def mutate_before_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                path.write_text('{"frames":[]}', encoding="utf-8")
            return original(*args, **kwargs)

        with patch.object(audit, "_inventory", side_effect=mutate_before_second):
            with self.assertRaises(audit.AuditError):
                self.run_audit()
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".audit.staging-*")), [])

    def test_cli_from_external_cwd_with_relative_asset_paths(self):
        self.cache(frames=[{"frame": 1, "detections": [self.row(.07)]}])
        result = subprocess.run([
            sys.executable, str(REPO_ROOT / "tools/audit_wem_s0.py"),
            "--detections-dir", "detections", "--output-dir", "audit",
            "--low-thresh", ".1", "--high-thresh", ".5", "--sequence", "sample",
        ], cwd=self.root, text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["totals"]["s0"], 1)
        self.assertTrue((self.output / "manifest.json").is_file())

    def test_duplicate_json_keys_and_invalid_thresholds_rejected(self):
        path = self.inputs / "sample.json"
        path.write_text('{"frames": [], "frames": []}', encoding="utf-8")
        with self.assertRaises(audit.AuditError):
            self.run_audit()
        self.cache()
        for low, high in ((.5, .1), (.1, .1), (-.1, .5), (.1, float("nan"))):
            with self.subTest(low=low, high=high):
                with self.assertRaises(audit.AuditError):
                    audit.audit_directory(self.inputs, self.output, low_thresh=low, high_thresh=high)


if __name__ == "__main__":
    unittest.main()
