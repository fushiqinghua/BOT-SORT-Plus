"""Integration tests against the actual pinned BoxMOT implementation.

Synthetic textured images exercise the real sparse optical-flow GMC and Kalman
filter. Synthetic unit embeddings are SOFTWARE fixtures, not network outputs or
VT-Tiny-MOT benchmark evidence. Run in the optional native-runtime environment.
"""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

for _dependency in ("boxmot", "torch", "cv2"):
    if importlib.util.find_spec(_dependency) is None:
        raise unittest.SkipTest(f"Native integration tests require {_dependency}")

import torch
from boxmot.trackers.botsort.botsort import BotSort
from boxmot.trackers.botsort.basetrack import BaseTrack, TrackState

from botsort_plus.appearance import AppearanceConfig
from botsort_plus.candidates import BirthConfig
from botsort_plus.config import PlusConfig
from botsort_plus.native_adapter import BoTSORTPlus, CacheOnlyEncoder, verify_upstream
from botsort_plus.recovery import RecoveryConfig
from botsort_plus.types import Feature


PARAMS = dict(reid_weights=Path("unused-cache-only-weights.pt"), device=torch.device("cpu"),
              half=False, det_thresh=.5, track_high_thresh=.5, track_low_thresh=.1,
              new_track_thresh=.6, match_thresh=.8, track_buffer=60, frame_rate=15,
              cmc_method="sof", proximity_thresh=.5, appearance_thresh=.4,
              fuse_first_associate=False)
E1, E2 = np.array([1., 0.], np.float32), np.array([0., 1.], np.float32)
IMAGE = np.random.default_rng(9281).integers(0, 256, (128, 160, 3), dtype=np.uint8)


def detections(*rows):
    return np.asarray(rows, np.float32).reshape(-1, 6)


def row(x=50., y=50., w=4., h=4., score=.95):
    return [x, y, x+w, y+h, score, 0]


def packets(rgb=E1, rgb_quality=1., thermal=None, thermal_quality=1.):
    result = {}
    if rgb is not None:
        result["rgb"] = Feature(np.asarray(rgb, np.float32).copy(), rgb_quality)
    if thermal is not None:
        result["thermal"] = Feature(np.asarray(thermal, np.float32).copy(), thermal_quality)
    return result


def dense_features(dets):
    output = np.zeros((len(dets), 2), np.float32)
    for i, det in enumerate(dets):
        if det[4] > .5:
            output[i] = E1 if det[0] < 70 else E2
    return output


def native():
    # This is precisely the real native algorithm, with only model construction
    # disabled because update() always receives explicit cached features.
    tracker = BotSort(with_reid=False, **PARAMS)
    tracker.with_reid = True
    tracker.model = CacheOnlyEncoder()
    return tracker


def enhanced(**sections):
    return BoTSORTPlus(PlusConfig(**sections), with_reid=True, **PARAMS)


class NativeAdapterTests(unittest.TestCase):
    def assert_consistent_lists(self, tracker):
        active = [t.id for t in tracker.active_tracks]
        lost = [t.id for t in tracker.lost_stracks if t.state == TrackState.Lost]
        self.assertEqual(len(active), len(set(active)))
        self.assertEqual(len(lost), len(set(lost)))
        self.assertFalse(set(active) & set(lost))

    def test_upstream_source_is_pinned(self):
        self.assertEqual(len(verify_upstream()), 5)

    def test_all_disabled_matches_native_exactly_every_frame(self):
        sequence = [
            detections(),
            detections(row(x=20), row(x=90), row(x=120, score=.3), row(x=130, score=.5)),
            detections(row(x=20.5), row(x=90.5), row(x=121, score=.3)),
            detections(row(x=21, score=.3), row(x=91)),
            detections(),
            detections(row(x=22, score=.3), row(x=92, score=.5)),
            detections(row(x=22.5), row(x=92.5)),
            detections(row(x=23), row(x=93, score=.1)),
            detections(row(x=23.5, score=.6), row(x=94)),
            detections(), detections(),
            detections(row(x=24), row(x=94.5)),
        ]
        original = native()
        expected = [original.update(d.copy(), IMAGE.copy(), embs=dense_features(d)).copy()
                    for d in sequence]
        # Native uses a class-wide ID counter. Complete the first run before
        # constructing the second, which legitimately resets that counter.
        plus = enhanced()
        actual = [plus.update(d.copy(), IMAGE.copy(), embs=dense_features(d)).copy()
                  for d in sequence]
        self.assertEqual(len(expected), len(actual))
        self.assertTrue(any(len(result) > 0 for result in actual))
        for frame, (a, b) in enumerate(zip(expected, actual), 1):
            with self.subTest(frame=frame):
                np.testing.assert_array_equal(a, b)

    def test_a_can_birth_without_feature_then_accept_real_rgb(self):
        tracker = enhanced(birth=BirthConfig(enabled=True))
        weak = detections(row(score=.3))
        observed = [tracker.update(weak, IMAGE.copy(), features=[{}]).copy() for _ in range(3)]
        self.assertEqual([len(x) for x in observed], [0, 0, 1])
        tid = int(observed[-1][0, 4])
        self.assertIsNone(tracker.active_tracks[0].smooth_feat)
        result = tracker.update(detections(row(score=.9)), IMAGE.copy(), features=[packets()])
        self.assertEqual(result[:, 4].astype(int).tolist(), [tid])
        np.testing.assert_allclose(tracker.active_tracks[0].smooth_feat, E1)
        self.assert_consistent_lists(tracker)

    def test_a_enabled_preserves_native_matching_when_no_weak_candidates_exist(self):
        sequence = [detections(row(x=20)), detections(row(x=20.5)), detections(),
                    detections(row(x=21)), detections(row(x=21.5)), detections()]
        original = native()
        expected = [original.update(d, IMAGE.copy(), embs=dense_features(d)).copy() for d in sequence]
        tracker = enhanced(birth=BirthConfig(enabled=True))
        actual = [tracker.update(d, IMAGE.copy(), embs=dense_features(d)).copy() for d in sequence]
        for frame, (a, b) in enumerate(zip(expected, actual), 1):
            with self.subTest(frame=frame):
                np.testing.assert_array_equal(a.reshape(-1,8), b.reshape(-1,8))

    def test_a_preserves_strict_native_score_boundary(self):
        tracker = enhanced(birth=BirthConfig(enabled=True))
        # .5 is exactly representable. Use an actually subthreshold value for
        # .1: native hstack promotes float32(.1) to 0.10000000149 in float64.
        lower = float(np.nextafter(np.float32(.1), np.float32(0.)))
        for _ in range(7):
            result = tracker.update(detections(row(score=.5), row(x=80, score=lower)),
                                    IMAGE.copy(), features=[{}, {}])
            self.assertEqual(len(result), 0)
        self.assertFalse(tracker.candidate_pool.candidates)

    def test_b_low_short_track_recovers_same_id(self):
        tracker = enhanced(recovery=RecoveryConfig(enabled=True))
        first = tracker.update(detections(row()), IMAGE.copy(), features=[packets()])
        tid = int(first[0, 4])
        tracker.update(detections(), IMAGE.copy(), features=[])
        outputs = []
        for _ in range(3):
            outputs.append(tracker.update(detections(row(score=.3)), IMAGE.copy(), features=[{}]).copy())
            self.assert_consistent_lists(tracker)
        self.assertEqual([len(x) for x in outputs], [0, 0, 1])
        self.assertEqual(int(outputs[-1][0, 4]), tid)
        self.assertEqual(BaseTrack._count, 1)
        self.assertTrue(any(e["event"] == "recovered" for e in tracker.last_events))

    def test_b_reservation_defers_native_high_birth_until_verified(self):
        tracker = enhanced(recovery=RecoveryConfig(enabled=True))
        first = tracker.update(detections(row()), IMAGE.copy(), features=[packets()])
        tid = int(first[0, 4])
        tracker.update(detections(), IMAGE.copy(), features=[])
        # 3px shift of a 4px box: native IoU cost > .8, while the bounded
        # tiny-object geometry allows a plausible multiframe recovery.
        for frame_index in range(3):
            output = tracker.update(detections(row(x=53)), IMAGE.copy(), features=[packets()])
            if frame_index < 2:
                self.assertEqual(len(output), 0)
                self.assertEqual(BaseTrack._count, 1)
                self.assertTrue(any(e["event"] == "deferred_birth" for e in tracker.last_events))
            else:
                self.assertEqual(output[:, 4].astype(int).tolist(), [tid])
                self.assertTrue(any(e["event"] == "recovered" for e in tracker.last_events))
            self.assert_consistent_lists(tracker)

    def test_ab_prioritizes_recovery_and_consumes_detection_once(self):
        tracker = enhanced(birth=BirthConfig(enabled=True), recovery=RecoveryConfig(enabled=True))
        first = tracker.update(detections(row()), IMAGE.copy(), features=[packets()])
        tid = int(first[0, 4])
        tracker.update(detections(), IMAGE.copy(), features=[])
        for _ in range(3):
            output = tracker.update(detections(row(score=.3)), IMAGE.copy(), features=[{}])
        self.assertEqual(output[:, 4].astype(int).tolist(), [tid])
        self.assertEqual(len({int(x) for x in output[:, 7]}), len(output))
        self.assertEqual(BaseTrack._count, 1)
        self.assertFalse(tracker.candidate_pool.candidates)
        self.assertEqual(sum(e["event"] == "recovered" for e in tracker.last_events), 1)
        self.assertFalse(any(e["event"] == "weak_birth" for e in tracker.last_events))
        self.assert_consistent_lists(tracker)

    def test_b_ambiguous_pool_link_cannot_bypass_into_native_high_birth(self):
        tracker = enhanced(recovery=RecoveryConfig(enabled=True))
        tracker.update(detections(row(x=40, score=.3), row(x=44, score=.3)),
                       IMAGE.copy(), features=[{}, {}])
        output = tracker.update(detections(row(x=42)), IMAGE.copy(), features=[packets()])
        self.assertEqual(len(output), 0)
        self.assertEqual(BaseTrack._count, 0)
        self.assertFalse(tracker.active_tracks)
        self.assertTrue(any(e["event"] == "candidate_seed_deferred" for e in tracker.last_events))

    def test_duplicate_pool_suppression_cannot_bypass_into_native_high_birth(self):
        tracker = enhanced(birth=BirthConfig(enabled=True))
        output = tracker.update(detections(row(w=20, h=20), row(w=20, h=20, score=.9)),
                                IMAGE.copy(), features=[packets(), packets()])
        self.assertEqual(len(output), 1)
        self.assertEqual(BaseTrack._count, 1)
        output = tracker.update(detections(row(w=20, h=20), row(w=20, h=20, score=.9)),
                                IMAGE.copy(), features=[packets(), packets()])
        self.assertEqual(len(output), 1)
        self.assertEqual(BaseTrack._count, 1)
        self.assertTrue(any(e["event"] == "candidate_observation_suppressed"
                            for e in tracker.last_events))

    def test_c_changes_real_native_high_association(self):
        start = detections(row(x=10, y=10, w=20, h=20), row(x=18, y=10, w=20, h=20))
        overlap = detections(row(x=14, y=10, w=20, h=20), row(x=14, y=10, w=20, h=20))
        first_features = [packets(E1, thermal=E1), packets(E2, thermal=E2)]
        current_features = [packets(E2, .2, E1), packets(E1, .2, E2)]
        off = enhanced()
        off.update(start, IMAGE.copy(), features=first_features)
        out_off = off.update(overlap, IMAGE.copy(), features=current_features)
        on = enhanced(appearance=AppearanceConfig(enabled=True))
        on.update(start, IMAGE.copy(), features=first_features)
        out_on = on.update(overlap, IMAGE.copy(), features=current_features)
        mapping_off = {int(r[4]): int(r[7]) for r in out_off}
        mapping_on = {int(r[4]): int(r[7]) for r in out_on}
        self.assertEqual(mapping_off, {1: 1, 2: 0})
        self.assertEqual(mapping_on, {1: 0, 2: 1})
        # Low-quality RGB did not overwrite the prior identity prototypes.
        np.testing.assert_allclose(on.memories[1].prototypes["rgb"], E1)
        np.testing.assert_allclose(on.memories[2].prototypes["rgb"], E2)

    def test_c_low_quality_updates_geometry_but_freezes_both_memories(self):
        tracker = enhanced(appearance=AppearanceConfig(enabled=True))
        tracker.update(detections(row(w=20, h=20)), IMAGE.copy(),
                       features=[packets(E1, thermal=E1)])
        memory = tracker.memories[1]
        previous = {k: v.copy() for k, v in memory.prototypes.items()}
        counts = dict(memory.update_counts)
        original_mean = tracker.active_tracks[0].mean.copy()
        output = tracker.update(detections(row(x=51, w=20, h=20)), IMAGE.copy(),
                                features=[packets(E2, 0., E2, 0.)])
        self.assertEqual(output[:, 4].astype(int).tolist(), [1])
        self.assertGreater(tracker.active_tracks[0].mean[0], original_mean[0])
        self.assertEqual(memory.update_counts, counts)
        for key in previous:
            np.testing.assert_array_equal(memory.prototypes[key], previous[key])
        np.testing.assert_array_equal(tracker.active_tracks[0].smooth_feat, E1)

    def test_c_unconfirmed_gate_uses_raw_fused_distance_before_cost_scaling(self):
        tracker = enhanced(appearance=AppearanceConfig(enabled=True, max_distance=.6))
        tracker.update(detections(), IMAGE.copy(), features=[])
        tracker.update(detections(row(x=20, y=20, w=20, h=20)), IMAGE.copy(),
                       features=[packets()])
        self.assertFalse(tracker.active_tracks[0].is_activated)
        # Cosine distance .8 is outside C's raw .6 gate. Native unconfirmed
        # association scales costs by 1/2, but .4 must NOT evade that gate.
        incompatible = np.array([.2, np.sqrt(.96)], np.float32)
        tracker.update(detections(row(x=24, y=20, w=20, h=20, score=.7)), IMAGE.copy(),
                       features=[packets(incompatible)])
        matches = [e for e in tracker.last_events if e["event"] == "match"
                   and e["stage"] == "unconfirmed"]
        self.assertEqual(len(matches), 1)
        # Fallback geometry: 1 - IoU(4px shift of 20px box) * .7 = .53333.
        self.assertAlmostEqual(matches[0]["cost"], 1 - (16 / 24) * .7, places=5)

    def test_missing_high_rgb_feature_fails_instead_of_using_zero(self):
        tracker = enhanced(birth=BirthConfig(enabled=True))
        with self.assertRaisesRegex(ValueError, "no real RGB"):
            tracker.update(detections(row()), IMAGE.copy(), features=[{}])
        with self.assertRaisesRegex(ValueError, "no real RGB"):
            tracker.update(detections(row()), IMAGE.copy(), embs=np.zeros((1, 2), np.float32))

    def test_recovery_buffer_must_match_fps_scaled_native_retention(self):
        with self.assertRaisesRegex(ValueError, "FPS-scaled"):
            enhanced(recovery=RecoveryConfig(enabled=True, max_age=31))


if __name__ == "__main__":
    unittest.main()
