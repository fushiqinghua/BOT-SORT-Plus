"""Small deterministic cohorts verify diagnostic meaning, not dataset results."""
import gzip
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from botsort_research.cmc import SOFProbe, affine_residuals, spatial_coverage
from botsort_research.diagnostics import (box_iou, candidate_label, candidate_from_scores,
    diagnose_appearance, load_feature_cache, match_boxes)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class DiagnosticGeometryTests(unittest.TestCase):
    def test_iou_and_assignment_handle_empty_and_disjoint(self):
        boxes = np.array([[0., 0., 4., 4.]])
        self.assertEqual(box_iou([], boxes).shape, (0, 1))
        self.assertEqual(match_boxes(boxes, np.array([[9., 0., 13., 4.]]))[0], {})
        self.assertEqual(match_boxes(boxes, boxes)[0], {0: 0})

    def test_confidence_boundary_preserves_native_gate(self):
        self.assertEqual(candidate_label(.5), "outside_association_gate")
        self.assertEqual(candidate_label(.1), "outside_association_gate")
        self.assertEqual(candidate_label(.49), "low_association")
        self.assertEqual(candidate_label(.6), "birth_eligible")
        self.assertEqual(candidate_from_scores([.5, .4]), ("low_association", .4))

    def test_motion_geometry_is_native_pixel_interpretable(self):
        points = np.array([[5., 5.], [95., 5.], [5., 95.], [95., 95.]])
        self.assertEqual(spatial_coverage(points, (100, 100), grid=2), 1.)
        warp = np.array([[1., 0., 3.], [0., 1., -2.]])
        np.testing.assert_allclose(affine_residuals(points, points + [3., -2.], warp), 0.)


class FeatureCacheTests(unittest.TestCase):
    def test_gzip_hash_and_zero_feature_exclusion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frames = [{"frame": 1, "detections": [[0, 0, 4, 4, .9, 0], [5, 5, 9, 9, .8, 0]]}]
            raw = json.dumps({"frames": frames}).encode()
            cache = root / "s.json.gz"
            cache.write_bytes(gzip.compress(raw))
            feature = root / "s.npz"
            np.savez(feature, features=np.array([[1., 0.], [0., 0.]]),
                     indices=np.array([0, 1]), offsets=np.array([0, 2]), frame_ids=np.array([1]))
            metadata = {"detection_sha256": hashlib.sha256(raw).hexdigest(),
                        "features_sha256": digest(feature)}
            feature.with_suffix(".json").write_text(json.dumps(metadata))
            vectors, report = load_feature_cache(cache, frames, root, "s")
            self.assertEqual(report["status"], "available")
            self.assertEqual(report["zero_vectors_excluded"], 1)
            self.assertEqual(set(vectors[1]), {0})
            metadata["detection_sha256"] = "different-source"
            feature.with_suffix(".json").write_text(json.dumps(metadata))
            vectors, report = load_feature_cache(cache, frames, root, "s")
            self.assertEqual(vectors, {})
            self.assertEqual(report["reason"], "detection_sha256_mismatch")

    def test_duplicate_indices_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frames = [{"frame": 1, "detections": [[0, 0, 4, 4, .9, 0]]}]
            cache = root / "detections.json"
            cache.write_text(json.dumps({"frames": frames}))
            feature = root / "s.npz"
            np.savez(feature, features=np.array([[1., 0.], [1., 0.]]),
                     indices=np.array([0, 0]), offsets=np.array([0, 2]), frame_ids=np.array([1]))
            feature.with_suffix(".json").write_text(json.dumps({
                "detection_sha256": digest(cache), "features_sha256": digest(feature)}))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_feature_cache(cache, frames, root, "s")


class AppearanceCohortTests(unittest.TestCase):
    def test_gt_outside_cache_frame_range_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, gt = root / "cache", root / "gt" / "s" / "gt"
            cache.mkdir()
            gt.mkdir(parents=True)
            (cache / "s.json").write_text(json.dumps({"frames": [{"frame": 1, "detections": []}]}))
            (gt / "gt.txt").write_text("2,7,0,0,4,4,1,1,1\n")
            with self.assertRaisesRegex(ValueError, "GT contains frames outside"):
                diagnose_appearance(cache, root / "gt", root / "report")

    def test_no_embeddings_still_reports_low_score_runs_and_gaps(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            gt = root / "gt" / "s" / "gt"
            pred = root / "pred"
            for path in (cache, gt, pred):
                path.mkdir(parents=True)
            frames = [{"frame": f, "detections": [[0, 0, 4, 4, score, 0]]}
                      for f, score in enumerate([.9, .4, .3, .9], 1)]
            (cache / "s.json").write_text(json.dumps({"frames": frames}))
            (gt / "gt.txt").write_text("".join(f"{f},7,0,0,4,4,1,1,1\n" for f in range(1, 5)))
            (pred / "s.txt").write_text("1,10,0,0,4,4,.9,-1,-1,-1\n4,10,0,0,4,4,.9,-1,-1,-1\n")
            result = diagnose_appearance(cache, root / "gt", root / "report", predictions_dir=pred)
            self.assertEqual(result["counts"]["low_score_runs"], {"1-3": 1})
            self.assertEqual(result["counts"]["low_score_run_frames"], {"1-3": 2})
            self.assertEqual(result["sequences"][0]["features"]["reason"], "embeddings_not_supplied")
            self.assertEqual(result["counts"]["unmatched_output_by_candidate"], {"low_association": 2})
            rows = (root / "report" / "prediction_gaps.csv").read_text().splitlines()
            self.assertEqual(len(rows), 2)
            self.assertIn("s,7,1,4,2,0,1,<8", rows[1])

    def test_geometry_gated_appearance_pairs_separate_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, gt, embeddings = root / "cache", root / "gt" / "s" / "gt", root / "embeddings"
            for path in (cache, gt, embeddings):
                path.mkdir(parents=True)
            detections = [[0, 0, 4, 4, .9, 0], [10, 0, 14, 4, .9, 0]]
            frames = [{"frame": f, "detections": detections} for f in (1, 2)]
            path = cache / "s.json"
            path.write_text(json.dumps({"frames": frames}))
            (gt / "gt.txt").write_text("".join(f"{f},{i},{x},0,4,4,1,1,1\n" for f in (1, 2) for i, x in ((1, 0), (2, 10))))
            feature = embeddings / "s.npz"
            np.savez(feature, features=np.array([[1., 0.], [0., 1.], [1., 0.], [0., 1.]]),
                     indices=np.array([0, 1, 0, 1]), offsets=np.array([0, 2, 4]), frame_ids=np.array([1, 2]))
            feature.with_suffix(".json").write_text(json.dumps({
                "detection_sha256": digest(path), "features_sha256": digest(feature)}))
            result = diagnose_appearance(cache, root / "gt", root / "report", embeddings_dir=embeddings)
            distributions = result["appearance_distributions"]
            self.assertEqual(distributions["same_id"]["samples"], 2)
            self.assertEqual(distributions["same_id"]["mean"], 0.)
            self.assertEqual(distributions["different_id"]["samples"], 2)
            self.assertEqual(distributions["different_id"]["mean"], 1.)


@unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV required for motion-image tests")
class MotionImageTests(unittest.TestCase):
    def test_textureless_failure_is_not_a_zero_motion_estimate(self):
        probe = SOFProbe()
        for _ in range(2):
            report = probe.step(np.zeros((512, 640, 3), dtype=np.uint8))
            self.assertFalse(report["estimated"])
            self.assertEqual(report["status"], "insufficient_initial_features")
            self.assertIsNone(report["residual_median_native_px"])

    def test_known_translation_is_measured_in_native_pixels(self):
        import cv2
        rng = np.random.default_rng(17)
        image = rng.integers(0, 256, (512, 640, 3), dtype=np.uint8)
        warp = np.array([[1., 0., 4.], [0., 1., -3.]], dtype=np.float32)
        following = cv2.warpAffine(image, warp, (640, 512), borderMode=cv2.BORDER_REFLECT)
        probe = SOFProbe(scale=.5)
        self.assertEqual(probe.step(image)["status"], "initialized_no_pair_estimate")
        result = probe.step(following)
        self.assertTrue(result["estimated"])
        np.testing.assert_allclose(np.asarray(result["warp"])[:, 2], [4., -3.], atol=.7)
        self.assertGreater(result["inlier_fraction"], .5)


if __name__ == "__main__":
    unittest.main()
