"""Coverage counts must respect one-to-one assignment and frozen score casts."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from botsort_research.detection_coverage import candidate_pools, diagnose_detection_coverage
from botsort_research.diagnostics import match_boxes


class CandidatePoolTests(unittest.TestCase):
    def test_boundary_casts_match_frozen_hstack_sequence(self):
        below_low = float(np.nextafter(np.float32(.1), np.float32(0)))
        above_low = float(np.nextafter(np.float32(.1), np.float32(1)))
        below_high = float(np.nextafter(np.float32(.5), np.float32(0)))
        above_high = float(np.nextafter(np.float32(.5), np.float32(1)))
        below_birth = float(np.nextafter(np.float32(.6), np.float32(0)))
        scores = [below_low, .1, above_low, below_high, .5, above_high, below_birth, .6,
                  .599999999999, .500000000001]
        detections = np.array([[0, 0, 4, 4, score, 0] for score in scores])
        actual = candidate_pools(detections)
        # Exact reviewed runtime operations, not an approximation of boundaries.
        runtime_rows = np.hstack([detections.astype(np.float32),
                                  np.arange(len(detections)).reshape(-1, 1)])
        confidence = runtime_rows[:, 4]
        expected = {
            "all_cached": np.arange(len(scores)),
            "high": np.flatnonzero(confidence > .5),
            "association_eligible": np.flatnonzero((confidence > .5) | ((confidence > .1) & (confidence < .5))),
            "birth_eligible": np.flatnonzero(confidence >= .6),
        }
        for pool in expected:
            np.testing.assert_array_equal(actual[pool], expected[pool])
        self.assertNotIn(0, actual["association_eligible"])
        self.assertIn(1, actual["association_eligible"], "literal .1 rounds above .1 before float64 comparison")
        self.assertNotIn(4, actual["association_eligible"], "exact .5 remains excluded")
        self.assertNotIn(6, actual["birth_eligible"])
        self.assertIn(7, actual["birth_eligible"])
        self.assertIn(8, actual["birth_eligible"], "rounding before comparison matters near .6")
        self.assertNotIn(9, actual["high"], "rounding before comparison matters near .5")

    def test_input_float32_scores_use_same_boundaries_and_empty_is_valid(self):
        detections = np.array([[0, 0, 4, 4, .1, 0], [0, 0, 4, 4, .5, 0],
                               [0, 0, 4, 4, .6, 0]], dtype=np.float32)
        pools = candidate_pools(detections)
        np.testing.assert_array_equal(pools["association_eligible"], [0, 2])
        np.testing.assert_array_equal(pools["birth_eligible"], [2])
        self.assertTrue(all(len(value) == 0 for value in candidate_pools([]).values()))

    def test_cardinality_dominates_total_iou_and_one_detection_cannot_cover_two_gt(self):
        truth = np.array([[0, 0, 10, 10], [5, 0, 15, 10]], dtype=float)
        candidates = np.array([[0, 0, 10, 10], [-3, 0, 7, 10]], dtype=float)
        matches, overlap = match_boxes(truth, candidates, .3)
        self.assertEqual(matches, {0: 1, 1: 0})
        self.assertLess(overlap[0, 1] + overlap[1, 0], overlap[0, 0],
                        "The two valid matches deliberately have lower summed IoU than one perfect match")
        matches, _ = match_boxes(truth, candidates[:1], .3)
        self.assertEqual(len(matches), 1)


class CoverageReportTests(unittest.TestCase):
    def fixture(self, root, gt_text="1,1,0,0,4,4,1,1,1\n1,2,0,0,12,12,1,1,1\n"):
        cache = root / "cache"
        cache.mkdir()
        gt = root / "gt" / "s" / "gt"
        gt.mkdir(parents=True)
        path = cache / "s.json"
        path.write_text(json.dumps({"model_sha256": "f" * 64,
            "frames": [{"frame": 1, "detections": [[0, 0, 8, 8, .9, 0]]}]}))
        gt_path = gt / "gt.txt"
        gt_path.write_text(gt_text)
        return cache, root / "gt", path, gt_path

    def test_size_counts_partition_global_matching_and_input_hashes_are_saved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache, gt, source, gt_path = self.fixture(root)
            result = diagnose_detection_coverage(cache, gt, root / "out", sequences=["s"], thresholds=[.2])
            rows = [row for row in result["results"] if row["sequence"] == "s" and row["pool"] == "all_cached"]
            by_size = {row["size"]: row for row in rows}
            self.assertEqual(by_size["all"]["matched"], 1)
            self.assertEqual(by_size["<8"]["matched"], 0)
            self.assertEqual(by_size["8-16"]["matched"], 1)
            self.assertEqual(sum(row["matched"] for size, row in by_size.items() if size != "all"), 1)
            self.assertEqual(by_size["all"]["coverage_percent"], 50.)
            self.assertEqual(result["inputs"]["s"]["detections_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertEqual(result["inputs"]["s"]["gt_sha256"], hashlib.sha256(gt_path.read_bytes()).hexdigest())
            with (root / "out/frames.csv").open(encoding="utf-8-sig") as stream:
                frames = list(csv.DictReader(stream))
            self.assertEqual(len(frames), 4)
            self.assertTrue(all(row["matched"] == "1" for row in frames))

    def test_empty_gt_sequence_is_retained_without_nan_or_invented_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache, gt, _, _ = self.fixture(root, gt_text="")
            result = diagnose_detection_coverage(cache, gt, root / "out", sequences=["s"], thresholds=[.5])
            self.assertEqual(result["status"], "COMPLETE")
            rows = result["results"]
            self.assertEqual(len(rows), 8)  # sequence + combined, four pools
            self.assertTrue(all(row["gt"] == row["matched"] == 0 for row in rows))
            self.assertTrue(all(row["coverage_percent"] is None for row in rows))
            json.dumps(result, allow_nan=False)

    def test_out_of_range_gt_fails_and_marks_manifest_failed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache, gt, _, _ = self.fixture(root, gt_text="2,1,0,0,4,4,1,1,1\n")
            with self.assertRaisesRegex(ValueError, "GT frame outside"):
                diagnose_detection_coverage(cache, gt, root / "out", sequences=["s"])
            manifest = json.loads((root / "out/manifest.json").read_text())
            self.assertEqual(manifest["status"], "FAILED")
            self.assertFalse((root / "out/summary.json").exists())

    def test_duplicate_gt_identity_fails_and_preserves_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache, gt, _, _ = self.fixture(root, gt_text="1,1,0,0,4,4,1,1,1\n1,1,10,10,4,4,1,1,1\n")
            with self.assertRaisesRegex(ValueError, "Duplicate GT identity"):
                diagnose_detection_coverage(cache, gt, root / "out", sequences=["s"])
            self.assertEqual(json.loads((root / "out/manifest.json").read_text())["status"], "FAILED")

    def test_existing_output_cannot_be_overwritten_after_gt_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache, gt, _, gt_path = self.fixture(root)
            diagnose_detection_coverage(cache, gt, root / "out", sequences=["s"])
            original = (root / "out/summary.json").read_bytes()
            gt_path.write_text("1,1,100,100,4,4,1,1,1\n")
            with self.assertRaises(FileExistsError):
                diagnose_detection_coverage(cache, gt, root / "out", sequences=["s"])
            self.assertEqual((root / "out/summary.json").read_bytes(), original)
            self.assertNotEqual(json.loads(original)["inputs"]["s"]["gt_sha256"],
                                hashlib.sha256(gt_path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
