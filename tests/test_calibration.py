import json
import tempfile
from pathlib import Path
import unittest

import numpy as np

from botsort_research.calibration import (
    build_split, calibrate, fit_noise_model, flight_group, select_pairs, validate_split,
)


class SplitTests(unittest.TestCase):
    def test_recordings_never_cross_partitions(self):
        names = ["DJI_0028_1", "DJI_0028_2", "DJI_0030_1", "DJI_0031_2", "DJI_0031_3"]
        split = build_split(names, seed=7)
        self.assertEqual(split, build_split(reversed(names), seed=7))
        self.assertEqual(split["interpretation"], "retrospective_development_only")
        left = {flight_group(s) for s in split["calibration"]}
        right = {flight_group(s) for s in split["evaluation"]}
        self.assertFalse(left & right)
        self.assertEqual(set(names), set(split["calibration"] + split["evaluation"]))

    def test_rejects_clip_level_leakage(self):
        split = {"schema_version": 1, "interpretation": "development", "calibration": ["DJI_0028_1"], "evaluation": ["DJI_0028_2"]}
        with self.assertRaisesRegex(ValueError, "Recording-group leakage"):
            validate_split(split)

    def test_rejects_sequence_overlap_missing_and_small_split(self):
        split = build_split(["a", "b", "c"])
        with self.assertRaisesRegex(ValueError, "unavailable"):
            validate_split(split, ["a", "b"])
        split["evaluation"] = split["calibration"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_split(split)
        with self.assertRaises(ValueError):
            build_split(["DJI_0028_1", "DJI_0028_2"])


class PairSelectionTests(unittest.TestCase):
    def test_nonoverlapping_tiny_boxes_can_calibrate(self):
        det = np.array([[10, 10, 12, 12, 0.8, 0]], dtype=float)
        gt = np.array([[13, 10, 15, 12, 7]], dtype=float)
        pairs, stats = select_pairs(det, gt)
        self.assertEqual(pairs.tolist(), [[0, 0]])
        self.assertEqual(stats["accepted_pairs"], 1)

    def test_duplicate_detections_are_ambiguous(self):
        det = np.array([[10, 10, 20, 20, 0.8, 0], [10.1, 10, 20.1, 20, 0.9, 0]])
        gt = np.array([[10, 10, 20, 20, 7]])
        pairs, stats = select_pairs(det, gt)
        self.assertEqual(len(pairs), 0)
        self.assertEqual(stats["ambiguous_pairs"], 1)

    def test_score_gate_and_empty_frames(self):
        det = np.array([[10, 10, 20, 20, 0.1, 0], [100, 100, 110, 110, 0.2, 0]])
        gt = np.array([[10, 10, 20, 20, 7], [100, 100, 110, 110, 8]])
        pairs, stats = select_pairs(det, gt)
        self.assertEqual(pairs.tolist(), [[1, 1]])
        self.assertEqual(stats["below_tracking_score"], 1)
        self.assertEqual(select_pairs(np.empty((0, 6)), gt)[0].shape, (0, 2))
        self.assertEqual(select_pairs(det, np.empty((0, 5)))[0].shape, (0, 2))

    def test_rejects_bad_geometry_and_scores(self):
        gt = np.array([[10, 10, 20, 20, 7]])
        for det in [np.array([[10, 10, 9, 20, .8, 0]]), np.array([[10, 10, 20, 20, 1.1, 0]]), np.array([[10, np.nan, 20, 20, .8, 0]])]:
            with self.assertRaises(ValueError):
                select_pairs(det, gt)


class NoiseFitTests(unittest.TestCase):
    def test_recovers_known_variance_components(self):
        rng = np.random.default_rng(8)
        n = 300
        measurements = np.column_stack((rng.uniform(0, 512, (n, 2)), rng.uniform(2, 40, (n, 2))))
        scores = rng.uniform(.11, .99, n)
        floor = np.array([1.0, 2.0, 1.5, 2.5])
        size = np.array([.05, .07, .10, .15])
        confidence = np.array([2.0, 2.5, 3.0, 3.5])
        variance = floor ** 2 + (measurements[:, [2, 3, 2, 3]] * size) ** 2 + (1 - scores[:, None]) ** 2 * confidence ** 2
        residuals = np.sqrt(variance) * rng.choice([-1, 1], size=(n, 4))
        model, diagnostics = fit_noise_model(measurements, residuals, scores)
        np.testing.assert_allclose(model["floor_std"], floor, rtol=1e-8)
        np.testing.assert_allclose(model["size_scale"], size, rtol=1e-8)
        np.testing.assert_allclose(model["score_scale"], confidence, rtol=1e-8)
        self.assertEqual(diagnostics["design_rank"], [3] * 4)
        self.assertEqual(model["mode"], "calibrated")

    def test_refuses_sparse_data_instead_of_using_evaluation(self):
        z = np.tile([10, 10, 4, 4], (8, 1))
        with self.assertRaisesRegex(ValueError, "Do not substitute evaluation"):
            fit_noise_model(z, np.ones((8, 4)), np.full(8, .8))

    def test_degenerate_design_is_explicit(self):
        z = np.tile([10, 10, 4, 4], (100, 1))
        model, diagnostics = fit_noise_model(z, np.ones((100, 4)), np.full(100, .8))
        self.assertEqual(diagnostics["design_rank"], [1] * 4)
        self.assertTrue(any("rank deficient" in x for x in diagnostics["warnings"]))
        self.assertTrue(all(x >= 0 for x in model["floor_std"]))


class CalibrationIntegrationTests(unittest.TestCase):
    def test_cache_gt_fit_and_noise_model_contract(self):
        from botsort_research.noise import NoiseModel
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache = root / "cache"
            cache.mkdir()
            gt_root = root / "gt"
            frames, rows = [], []
            for frame in range(1, 121):
                width = 8 + frame % 13
                height = 7 + frame % 9
                x, y = 40 + frame * .1, 30 + frame * .2
                dx = ((frame % 5) - 2) * .2
                frames.append({"frame": frame, "image": "unused.jpg", "detections": [[x + dx, y, x + width + dx, y + height, .2 + .6 * ((frame % 11) / 10), 0]]})
                rows.append(f"{frame},1,{x},{y},{width},{height},1,1,1\n")
            (cache / "DJI_0001_1.json").write_text(json.dumps({"model_sha256": "frozen-detector", "frames": frames}))
            (cache / "DJI_0002_1.json").write_text(json.dumps({"model_sha256": "frozen-detector", "frames": [{"frame": 1, "detections": []}]}))
            target = gt_root / "DJI_0001_1" / "gt"
            target.mkdir(parents=True)
            (target / "gt.txt").write_text("".join(rows))
            split = {"schema_version": 1, "interpretation": "retrospective_development_only", "calibration": ["DJI_0001_1"], "evaluation": ["DJI_0002_1"]}
            output = root / "fitted.json"
            fitted = calibrate(cache, gt_root, split["calibration"], output, split=split)
            self.assertEqual(fitted["diagnostics"]["samples"], 120)
            self.assertEqual(json.loads(output.read_text()), fitted)
            self.assertEqual(fitted["provenance"]["calibration_sequences"], ["DJI_0001_1"])
            self.assertEqual(len(fitted["provenance"]["inputs"]["DJI_0001_1"]["gt_sha256"]), 64)
            covariance = NoiseModel.from_dict(fitted["noise"]).covariance([30, 20, 10, 9], .8)
            self.assertTrue(np.all(np.linalg.eigvalsh(covariance) > 0))
            with self.assertRaisesRegex(ValueError, "exactly equal"):
                calibrate(cache, gt_root, split["evaluation"], split=split)
            with self.assertRaises(FileExistsError):
                calibrate(cache, gt_root, split["calibration"], output, split=split)


if __name__ == "__main__":
    unittest.main()
