"""Round2 causal controls: separate assignment, correction and output boxes."""
import os
import unittest

import numpy as np

from botsort_research.association import nwd_config, nwd_cost
from botsort_research.output_policy import apply_output_policy, output_policy_for_variant


class NWDTests(unittest.TestCase):
    def test_closed_form_native_pixel_distance(self):
        cost, reasons = nwd_cost([[10, 20, 4, 6]], [[13, 24, 4, 6], [10, 20, 8, 6]], normalizer_px=10)
        np.testing.assert_allclose(cost, [[1 - np.exp(-0.5), 1 - np.exp(-0.2)]])
        self.assertFalse(reasons.any())

    def test_no_overlap_required_and_physical_gates_remain(self):
        cost, reasons = nwd_cost([[10, 10, 2, 2]], [[13, 10, 2, 2], [100, 10, 2, 2], [10, 10, 20, 2]])
        self.assertTrue(np.isfinite(cost[0, 0]))
        self.assertTrue(np.isinf(cost[0, 1:]).all())
        self.assertEqual(reasons[0, 1], 2)
        self.assertEqual(reasons[0, 2], 4)
        self.assertEqual(nwd_cost([], [])[0].shape, (0, 0))
        self.assertEqual(nwd_cost([[10, 10, 2, 2]], [])[0].shape, (1, 0))

    def test_reject_invalid_normalization(self):
        for params in ({"normalizer_px": 0}, {"second_threshold": 1}, {"first_threshold": float("nan")}, {"unknown": 3}):
            with self.assertRaises(ValueError):
                nwd_config(params)


class OutputPolicyTests(unittest.TestCase):
    def setUp(self):
        self.dets = np.array([[11, 10, 15, 14, .9, 0], [40, 40, 48, 48, .8, 0]])
        self.rows = np.array([[10.2, 10, 14.2, 14, 1, .9, 0, 0], [40.2, 40, 48.2, 48, 2, .8, 0, 1]])

    def test_changes_only_coordinates_without_mutation(self):
        original_rows, original_dets = self.rows.copy(), self.dets.copy()
        converted = apply_output_policy(self.rows, self.dets, "detection")
        np.testing.assert_array_equal(converted[:, :4], self.dets[:, :4])
        np.testing.assert_array_equal(converted[:, 4:], self.rows[:, 4:])
        np.testing.assert_array_equal(self.rows, original_rows)
        np.testing.assert_array_equal(self.dets, original_dets)
        np.testing.assert_array_equal(apply_output_policy(self.rows, self.dets), self.rows)
        self.assertFalse(np.shares_memory(converted, self.rows))

    def test_rejects_stale_noninteger_and_duplicate_mapping(self):
        for values in ([0, 2], [0, .5], [0, 0]):
            changed = self.rows.copy()
            changed[:, 7] = values
            with self.assertRaises(ValueError):
                apply_output_policy(changed, self.dets, "detection")
        changed = self.rows.copy()
        changed[0, 5] = .2
        with self.assertRaises(ValueError):
            apply_output_policy(changed, self.dets, "detection")

    def test_empty_frame_has_no_output_and_policies_are_explicit(self):
        self.assertEqual(apply_output_policy(np.array([]), np.empty((0, 6)), "detection").shape, (0, 8))
        self.assertEqual(output_policy_for_variant("m2_nll_floor_native_update_det"), "detection")
        self.assertEqual(output_policy_for_variant("m2_nll_floor_native_update"), "kalman")
        with self.assertRaises(ValueError):
            apply_output_policy([], [], "extrapolate")


@unittest.skipUnless(os.environ.get("M2_TEST_BOXMOT") == "1", "Set M2_TEST_BOXMOT=1 in frozen BoxMOT runtime")
class Round2RuntimeTests(unittest.TestCase):
    CONFIG = dict(det_thresh=.5, track_high_thresh=.5, track_low_thresh=.1,
                  new_track_thresh=.6, match_thresh=.8, track_buffer=60,
                  frame_rate=15, with_reid=False, cmc_method="sof")
    NOISE = {"noise": {"mode": "floor", "floor_std": [1.5] * 4,
                       "size_scale": [.05] * 4, "score_scale": [2.] * 4}}

    @classmethod
    def setUpClass(cls):
        from botsort_research.runner import configure_runtime
        configure_runtime()

    def tracker(self, variant, **extra):
        from botsort_research.adapter import build_tracker
        from botsort_research.runner import IdentityCMC
        tracker = build_tracker(variant, self.CONFIG, self.NOISE, **extra)
        tracker.cmc = IdentityCMC()
        return tracker

    @staticmethod
    def det(x=10, score=.9):
        return np.array([[x, 10, x + 4, 14, score, 0]], np.float32)

    @staticmethod
    def step(tracker, det=None):
        return tracker.update(np.empty((0, 6), np.float32) if det is None else det,
                              np.zeros((64, 64, 3), np.uint8))

    def test_native_update_matches_exact_upstream_correction(self):
        from boxmot.trackers.botsort.botsort_track import STrack
        from boxmot.motion.kalman_filters.aabb.xywh_kf import KalmanFilterXYWH
        for variant in ("m2_nll_floor_native_update", "m2_nll_first_native_update", "m2_nll_second_native_update", "nwd_native_update"):
            events = []
            tracker = self.tracker(variant, trace_callback=events.append)
            self.step(tracker, self.det())
            track = tracker.active_tracks[0]
            mean, covariance = STrack.shared_kalman.multi_predict(track.mean[None].copy(), track.covariance[None].copy())
            expected_mean, expected_cov = KalmanFilterXYWH().update(mean[0], covariance[0], np.array([13, 12, 4, 4.]))
            self.assertEqual(len(self.step(tracker, self.det(11))), 1)
            np.testing.assert_allclose(track.mean, expected_mean, rtol=0, atol=1e-12)
            np.testing.assert_allclose(track.covariance, expected_cov, rtol=0, atol=1e-12)
            event = next(e for e in events if e["matches"])
            self.assertEqual(event["matches"][0]["update_mode"], "baseline")
            if event["association_method"] == "nll":
                self.assertFalse(np.allclose(event["matches"][0]["association_R_diag"], event["matches"][0]["update_R_diag"]))

    def test_first_second_stage_ablations_do_not_leak(self):
        tracker = self.tracker("m2_nll_first_native_update")
        first = self.step(tracker, self.det())
        moved = self.step(tracker, self.det(12.5))
        self.assertEqual(first[0, 4], moved[0, 4])
        tracker = self.tracker("m2_nll_first_native_update")
        self.step(tracker, self.det())
        self.assertEqual(len(self.step(tracker, self.det(12.5, .2))), 0)
        tracker = self.tracker("m2_nll_second_native_update")
        first = self.step(tracker, self.det())
        moved = self.step(tracker, self.det(12.5, .2))
        self.assertEqual(first[0, 4], moved[0, 4])

    def test_output_choice_does_not_change_future_states_or_ids(self):
        variants = (("native_iou", "native_iou_det"), ("m2_nll_floor", "m2_nll_floor_det"),
                    ("m2_nll_floor_native_update", "m2_nll_floor_native_update_det"))
        sequence = [self.det(), self.det(11), self.det(11.5, .2), None, self.det(12)]
        for standard, raw in variants:
            results = []
            for variant in (standard, raw):
                tracker = self.tracker(variant)
                snapshots = []
                for det in sequence:
                    out = self.step(tracker, det)
                    internal_before = [(t.id, t.mean.copy(), t.covariance.copy()) for t in tracker.active_tracks + tracker.lost_stracks]
                    adjusted = apply_output_policy(out, np.empty((0, 6)) if det is None else det, output_policy_for_variant(variant))
                    internal_after = [(t.id, t.mean.copy(), t.covariance.copy()) for t in tracker.active_tracks + tracker.lost_stracks]
                    for before, after in zip(internal_before, internal_after):
                        self.assertEqual(before[0], after[0])
                        np.testing.assert_array_equal(before[1], after[1])
                        np.testing.assert_array_equal(before[2], after[2])
                    snapshots.append((adjusted, internal_after))
                results.append(snapshots)
            for (a, sa), (b, sb) in zip(*results):
                np.testing.assert_array_equal(a[:, 4:], b[:, 4:])
                self.assertEqual(len(sa), len(sb))
                for ta, tb in zip(sa, sb):
                    self.assertEqual(ta[0], tb[0])
                    np.testing.assert_array_equal(ta[1], tb[1])
                    np.testing.assert_array_equal(ta[2], tb[2])

    def test_all_variants_keep_low_birth_and_lost_recovery_rules(self):
        from botsort_research.adapter import SUPPORTED_VARIANTS
        for variant in SUPPORTED_VARIANTS:
            if variant == "m2_nll_calibrated":
                continue
            tracker = self.tracker(variant)
            first = self.step(tracker, self.det())
            self.assertEqual(self.step(tracker, self.det(score=.2))[0, 4], first[0, 4])
            self.assertEqual(len(self.step(tracker)), 0)
            self.assertEqual(len(self.step(tracker, self.det(score=.2))), 0)
            self.assertEqual(self.step(tracker, self.det())[0, 4], first[0, 4])
            self.assertEqual(tracker.max_time_lost, 30)
            new = self.tracker(variant)
            self.assertEqual(len(self.step(new, self.det(score=.2))), 0)
            self.assertEqual(len(new.active_tracks), 0)

    def test_iou_floor_update_changes_only_correction_and_handles_unconfirmed(self):
        from botsort_research.noise import NoiseModel, correct_with_covariance
        tracker = self.tracker("iou_floor_update")
        self.step(tracker)
        self.assertEqual(len(self.step(tracker, self.det())), 0)
        track = tracker.active_tracks[0]
        old_mean, old_cov = track.mean.copy(), track.covariance.copy()
        z = np.array([12.5, 12, 4, 4.])
        noise = NoiseModel.from_dict(self.NOISE["noise"])
        expected, _ = correct_with_covariance(old_mean, old_cov, z, noise.covariance(z, float(self.det(10.5)[0, 4])))
        self.assertEqual(len(self.step(tracker, self.det(10.5))), 1)
        np.testing.assert_allclose(track.mean, expected, rtol=0, atol=1e-12)
        self.assertIsNone(track.kalman_filter.pending_R)

    def test_iou_relaxed_uses_native_factory_and_explicit_threshold(self):
        from boxmot.trackers.botsort.botsort import BotSort
        from botsort_research.adapter import build_tracker
        cfg = {**self.CONFIG, "match_thresh": .9}
        tracker = build_tracker("iou_relaxed", cfg)
        self.assertIs(type(tracker), BotSort)
        self.assertEqual(tracker.match_thresh, .9)

    def test_float32_point_one_matches_native_promoted_score_boundary(self):
        # Native hstack adds an int64 detection index, then compares the
        # promoted float64 value of float32(.1), which is slightly above .1.
        for variant in ("native_iou", "m2_nll_floor_native_update"):
            tracker = self.tracker(variant)
            identity = self.step(tracker, self.det())[0, 4]
            output = self.step(tracker, self.det(score=.1))
            self.assertEqual(output[0, 4], identity)
            self.assertGreater(output[0, 5], .1)


if __name__ == "__main__":
    unittest.main()
