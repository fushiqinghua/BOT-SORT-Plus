"""Numerical tests plus opt-in tests against the real, frozen BoxMOT runtime."""
import itertools
import os
import unittest

import numpy as np

from botsort_research.association import assign_with_dummies, gaussian_cost, stage_config
from botsort_research.noise import NoiseModel, correct_with_covariance


class NoiseTests(unittest.TestCase):
    def test_noise_roundtrip_and_formula(self):
        model = NoiseModel.from_dict({"floor_std": [1, 2, 3, 4], "size_scale": [0.1] * 4,
                                      "score_scale": [2] * 4, "mode": "calibrated"})
        z = np.array([12, 20, 4, 8])
        expected = np.square([1, 2, 3, 4]) + np.square([0.4, 0.8, 0.4, 0.8]) + 1
        np.testing.assert_allclose(np.diag(model.covariance(z, 0.5)), expected)
        self.assertEqual(model.to_dict(), NoiseModel.from_dict(model.to_dict()).to_dict())

    def test_baseline_uses_predicted_size_and_no_confidence_scaling(self):
        model = NoiseModel(mode="baseline")
        predicted = [20, 20, 8, 10]
        a = model.covariance([20, 20, 4, 5], 0.2, predicted)
        b = model.covariance([20, 20, 4, 5], 0.9, predicted)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(np.diag(a), [0.16, 0.25, 0.16, 0.25])
        with self.assertRaises(ValueError):
            model.covariance([20, 20, 4, 5], 0.9)

    def test_invalid_noise_parameters(self):
        for config in ({"floor_std": [1, -1, 1, 1]}, {"mode": "typo"},
                       {"min_variance": 0}, {"size_scale": [1, 2]}, {"unknown": 1}):
            with self.assertRaises(ValueError):
                NoiseModel.from_dict(config)
        with self.assertRaises(ValueError):
            NoiseModel().covariance([0, 0, 0, 3], 0.5)
        with self.assertRaises(ValueError):
            NoiseModel().covariance([0, 0, 3, 3], 1.1)

    def test_joseph_update_positive_definite_and_same_R(self):
        rng = np.random.default_rng(42)
        a = rng.normal(size=(8, 8))
        p = a @ a.T + np.eye(8) * 1e-4
        mean = np.array([10, 12, 4, 5, 0, 0, 0, 0.0])
        z = np.array([11, 11, 5, 4.0])
        r = NoiseModel().covariance(z, 0.3)
        new_mean, new_p = correct_with_covariance(mean, p, z, r)
        s = p[:4, :4] + r
        gain = np.linalg.solve(s, p[:, :4].T).T
        np.testing.assert_allclose(new_mean, mean + gain @ (z - mean[:4]), atol=1e-12)
        np.testing.assert_allclose(new_p, p - gain @ s @ gain.T, atol=1e-11)
        np.testing.assert_allclose(new_p, new_p.T, atol=1e-12)
        self.assertGreater(np.linalg.eigvalsh(new_p).min(), 0)

    def test_larger_R_reduces_measurement_pull(self):
        mean = np.array([10, 10, 4, 4, 0, 0, 0, 0.0])
        p = np.eye(8)
        z = np.array([12, 12, 4, 4.0])
        high, _ = correct_with_covariance(mean, p, z, NoiseModel().covariance(z, 0.95))
        low, _ = correct_with_covariance(mean, p, z, NoiseModel().covariance(z, 0.2))
        self.assertLess(np.linalg.norm(low[:2] - mean[:2]), np.linalg.norm(high[:2] - mean[:2]))


class AssignmentTests(unittest.TestCase):
    def test_empty_all_forbidden_and_negative_costs(self):
        for shape in ((0, 0), (2, 0), (0, 3), (2, 3)):
            matches, ut, ud = assign_with_dummies(np.full(shape, np.inf), 3, 4)
            self.assertEqual(matches.shape, (0, 2))
            np.testing.assert_array_equal(ut, np.arange(shape[0]))
            np.testing.assert_array_equal(ud, np.arange(shape[1]))
        matches, _, _ = assign_with_dummies(np.array([[-4.0]]), 0, 0)
        np.testing.assert_array_equal(matches, [[0, 0]])

    def test_dummy_rejection_and_tie(self):
        for cost in (6.0, 7.0):
            matches, _, _ = assign_with_dummies(np.array([[cost]]), 3, 3)
            self.assertEqual(len(matches), 0)
        matches, _, _ = assign_with_dummies(np.array([[5.9]]), 3, 3)
        self.assertEqual(len(matches), 1)

    def test_global_solution_equals_brute_force(self):
        rng = np.random.default_rng(10)
        for _ in range(20):
            n, m = 3, 2
            cost = rng.uniform(-2, 9, (n, m))
            cost[rng.random((n, m)) < 0.2] = np.inf
            miss = np.array([1.2, 2.3, 3.4])
            unassigned = np.array([2.1, 1.7])
            matches, ut, ud = assign_with_dummies(cost, miss, unassigned)
            observed = sum(cost[i, j] for i, j in matches) + miss[ut].sum() + unassigned[ud].sum()
            optimum = np.inf
            for choice in itertools.product(range(-1, m), repeat=n):
                selected = [j for j in choice if j >= 0]
                if len(set(selected)) != len(selected):
                    continue
                value = sum(miss[i] if j == -1 else cost[i, j] for i, j in enumerate(choice))
                value += sum(unassigned[j] for j in range(m) if j not in selected)
                optimum = min(optimum, value)
            self.assertAlmostEqual(observed, optimum, places=10)

    def test_invalid_costs_and_configuration(self):
        with self.assertRaises(ValueError):
            assign_with_dummies([[np.nan]], 1, 1)
        with self.assertRaises(ValueError):
            assign_with_dummies([[0]], -1, 1)
        with self.assertRaises(ValueError):
            stage_config({"gate_chi2": 0})
        with self.assertRaises(ValueError):
            stage_config({"max_size_ratio": 0.9})


class LikelihoodTests(unittest.TestCase):
    def test_nll_retains_gaussian_normalization_constant(self):
        # P=0 and R=I gives standard 4-D normal at one standard deviation.
        model = NoiseModel(floor_std=(1, 1, 1, 1), size_scale=(0, 0, 0, 0),
                           score_scale=(0, 0, 0, 0))
        costs, _ = gaussian_cost([[10, 10, 4, 4, 0, 0, 0, 0]], [np.zeros((8, 8))],
                                 [[11, 10, 4, 4]], [0.9], model)
        self.assertAlmostEqual(costs[0, 0], 0.5 + 2 * np.log(2 * np.pi), places=12)

    def test_logdet_penalty_prevents_diffuse_track_stealing(self):
        means = np.array([[10, 10, 4, 4, 0, 0, 0, 0], [10, 10, 4, 4, 0, 0, 0, 0.0]])
        covs = np.array([np.eye(8), np.eye(8) * 10000])
        costs, diag = gaussian_cost(means, covs, [[10, 10, 4, 4]], [0.9], NoiseModel())
        self.assertEqual(diag["mahalanobis2"][0, 0], 0)
        self.assertEqual(diag["mahalanobis2"][1, 0], 0)
        self.assertLess(costs[0, 0], costs[1, 0])
        matches, _, _ = assign_with_dummies(costs, 8, 8)
        np.testing.assert_array_equal(matches, [[0, 0]])

    def test_nonoverlapping_tiny_boxes_can_match(self):
        means = [[10, 10, 2, 2, 0, 0, 0, 0]]
        costs, diag = gaussian_cost(means, [np.eye(8)], [[12.5, 10, 2, 2]], [0.9], NoiseModel())
        self.assertTrue(np.isfinite(costs[0, 0]))
        matches, _, _ = assign_with_dummies(costs, 8, 8)
        self.assertEqual(len(matches), 1)
        self.assertEqual(diag["gate_reason"][0, 0], 0)

    def test_physical_gates_reject_large_uncertainty_pairs(self):
        mean = [[10, 10, 4, 4, 0, 0, 0, 0]]
        costs, diag = gaussian_cost(mean, [np.eye(8) * 1e6],
                                    [[100, 10, 4, 4], [10, 10, 40, 4]], [0.9, 0.9], NoiseModel())
        self.assertTrue(np.isinf(costs).all())
        self.assertTrue(diag["gate_reason"][0, 0] & 2)
        self.assertTrue(diag["gate_reason"][0, 1] & 4)

    def test_mahalanobis_and_invalid_state_rejection(self):
        mean = [[10, 10, 4, 4, 0, 0, 0, 0], [10, 10, -4, 4, 0, 0, 0, 0]]
        costs, diag = gaussian_cost(mean, [np.eye(8) * 0.01] * 2,
                                    [[30, 10, 4, 4]], [0.9], NoiseModel())
        self.assertTrue(np.isinf(costs).all())
        self.assertTrue(diag["gate_reason"][0, 0] & 1)
        self.assertTrue(diag["gate_reason"][1, 0] & 8)

    def test_empty_inputs_and_selected_R(self):
        costs, _ = gaussian_cost([], [], [[10, 10, 4, 4]], [0.9], NoiseModel())
        self.assertEqual(costs.shape, (0, 1))
        model = NoiseModel()
        means = [[10, 10, 4, 4, 0, 0, 0, 0]]
        z = [11, 10, 5, 4]
        _, details = gaussian_cost(means, [np.eye(8)], [z], [0.2], model)
        np.testing.assert_array_equal(details["measurement_covariance"][0, 0], model.covariance(z, 0.2))


@unittest.skipUnless(os.environ.get("M2_TEST_BOXMOT") == "1", "Set M2_TEST_BOXMOT=1 in frozen BoxMOT runtime")
class BoxMOTIntegrationTests(unittest.TestCase):
    CONFIG = dict(det_thresh=0.5, track_high_thresh=0.5, track_low_thresh=0.1,
                  new_track_thresh=0.6, match_thresh=0.8, track_buffer=60,
                  frame_rate=15, with_reid=False, cmc_method="sof")

    @classmethod
    def setUpClass(cls):
        from botsort_research.runner import configure_runtime
        configure_runtime()

    @staticmethod
    def identity_cmc(tracker):
        class Identity:
            def apply(self, image, dets):
                return np.eye(2, 3)
        tracker.cmc = Identity()
        return tracker

    def test_native_factory_is_exactly_original(self):
        from pathlib import Path
        import torch
        from boxmot.trackers.botsort.botsort import BotSort
        from botsort_research.adapter import build_tracker
        image = np.zeros((100, 100, 3), np.uint8)
        detections = [[], [[10, 10, 16, 16, 0.9, 0]], [[10.2, 10, 16.2, 16, 0.9, 0]],
                      [[10.4, 10, 16.4, 16, 0.2, 0]], [], [[10.8, 10, 16.8, 16, 0.9, 0]]]
        native = self.identity_cmc(BotSort(reid_weights=Path("unused.pt"), device=torch.device("cpu"),
                                          half=False, **self.CONFIG))
        expected = [native.update(np.array(d, np.float32).reshape(-1, 6), image).copy() for d in detections]
        wrapped = self.identity_cmc(build_tracker("native_iou", self.CONFIG))
        actual = [wrapped.update(np.array(d, np.float32).reshape(-1, 6), image).copy() for d in detections]
        for a, b in zip(expected, actual):
            np.testing.assert_array_equal(a, b)

    def test_m2_low_score_update_gap_recovery_and_threshold_edges(self):
        from botsort_research.adapter import build_tracker
        image = np.zeros((100, 100, 3), np.uint8)
        for variant in ("m2_nll_baseline_r", "m2_nll_floor"):
            events = []
            tracker = self.identity_cmc(build_tracker(variant, self.CONFIG, trace_callback=events.append))
            def step(score=None):
                detections = [] if score is None else [[10, 10, 14, 14, score, 0]]
                return tracker.update(np.array(detections, np.float32).reshape(-1, 6), image)
            first = step(0.9)
            self.assertEqual(len(first), 1)
            identity = first[0, 4]
            self.assertEqual(step(0.2)[0, 4], identity)  # second stage can update active track
            self.assertEqual(len(step()), 0)
            self.assertEqual(len(step(0.2)), 0)  # low score cannot recover a lost track
            self.assertEqual(step(0.9)[0, 4], identity)
            self.assertEqual(len(step(0.5)), 0)  # preserve native strict threshold boundary
            self.assertEqual(len(step(0.1)), 0)
            self.assertEqual(tracker.max_time_lost, 30)
            self.assertTrue(any(e["stage"] == "second" and e["matches"] for e in events))
            self.assertTrue(any(any(m["was_lost"] for m in e["matches"]) for e in events))

    def test_unconfirmed_track_uses_adaptive_R(self):
        from botsort_research.adapter import build_tracker
        tracker = self.identity_cmc(build_tracker("m2_nll_floor", self.CONFIG))
        image = np.zeros((100, 100, 3), np.uint8)
        empty = np.empty((0, 6), np.float32)
        det = np.array([[10, 10, 16, 16, 0.9, 0]], np.float32)
        tracker.update(empty, image)
        self.assertEqual(len(tracker.update(det, image)), 0)
        self.assertEqual(len(tracker.update(det, image)), 1)
        self.assertIsNone(tracker.active_tracks[0].kalman_filter.pending_R)

    def test_second_stage_updates_with_the_likelihood_R(self):
        from botsort_research.adapter import build_tracker
        from boxmot.trackers.botsort.botsort_track import STrack
        noise_config = {"mode": "floor", "floor_std": [1.5] * 4,
                        "size_scale": [0.05] * 4, "score_scale": [2] * 4}
        noise = NoiseModel.from_dict(noise_config)
        tracker = self.identity_cmc(build_tracker("m2_nll_floor", self.CONFIG,
                                                   {"noise": noise_config}))
        image = np.zeros((100, 100, 3), np.uint8)
        tracker.update(np.array([[10, 10, 14, 14, 0.9, 0]], np.float32), image)
        track = tracker.active_tracks[0]
        pm, pc = STrack.shared_kalman.multi_predict(track.mean[None].copy(), track.covariance[None].copy())
        det = np.array([[11, 10, 15, 14, 0.2, 0]], np.float32)
        z = np.array([13, 12, 4, 4.0])
        costs, details = gaussian_cost(pm, pc, [z], [det[0, 4]], noise)
        self.assertTrue(np.isfinite(costs[0, 0]))
        expected_mean, expected_cov = correct_with_covariance(pm[0], pc[0], z,
                                                               details["measurement_covariance"][0, 0])
        result = tracker.update(det, image)
        self.assertEqual(len(result), 1)
        np.testing.assert_allclose(tracker.active_tracks[0].mean, expected_mean, rtol=0, atol=1e-12)
        np.testing.assert_allclose(tracker.active_tracks[0].covariance, expected_cov, rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
