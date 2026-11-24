"""Behavior tests for B; no detector, GT, weights, or native runtime required."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

from botsort_plus.geometry import GeometryConfig
from botsort_plus.recovery import RecoveryConfig, RecoveryMatcher
from botsort_plus.types import Feature, LostTrackView, Observation, Prediction


BOX = np.array([10., 10., 14., 14.])


def candidate(cid=1, frames=(3, 4, 5), box=BOX, features=None):
    observations = [Observation(f, cid - 1, np.array(box).copy(), 0.3,
                                features or {}) for f in frames]
    return SimpleNamespace(candidate_id=cid, observations=observations,
                           last_frame=frames[-1], latest=observations[-1], evidence=2.)


def view(tid=7, last_seen=1, frames=range(2, 45), box=BOX, memory=None, covariance=1.):
    return LostTrackView(tid, last_seen,
                         {f: Prediction(f, np.array(box).copy(), np.eye(2) * covariance)
                          for f in frames}, memory)


def cosine(memory, features):
    if memory is None or "rgb" not in features:
        return None
    feature = features["rgb"]
    if feature.quality <= 0 or np.linalg.norm(feature.vector) == 0:
        return None
    a = np.asarray(memory, dtype=float)
    b = np.asarray(feature.vector, dtype=float)
    return float(np.clip(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), 0, 2))


class RecoveryTests(unittest.TestCase):
    def matcher(self, **kwargs):
        return RecoveryMatcher(RecoveryConfig(enabled=True, **kwargs), GeometryConfig())

    def test_disabled_has_no_side_effect_or_reservation(self):
        m = RecoveryMatcher(RecoveryConfig(), GeometryConfig())
        result = m.match(5, [view()], [candidate()])
        self.assertFalse(result.matches)
        self.assertFalse(result.reserved_candidate_ids)

    def test_three_observations_recover_existing_id_current_frame_only(self):
        result = self.matcher().match(5, [view()], [candidate()])
        self.assertEqual(len(result.matches), 1)
        match = result.matches[0]
        self.assertEqual((match.track_id, match.candidate_id, match.det_index), (7, 1, 0))
        self.assertEqual((match.age, match.hits), (4, 3))
        self.assertIsNone(match.appearance_cost)
        self.assertFalse(result.reserved_candidate_ids)
        # This module returns a current detection index, not fabricated rows for
        # the missing frame or the earlier two confirmation observations.
        self.assertFalse(hasattr(match, "backfilled_rows"))

    def test_one_hit_is_reserved_not_promoted(self):
        result = self.matcher().match(3, [view()], [candidate(frames=(3,))])
        self.assertFalse(result.matches)
        self.assertEqual(result.reserved_candidate_ids, {1})

    def test_competing_old_identities_reject_and_reserve(self):
        result = self.matcher().match(5, [view(7), view(8)], [candidate()])
        self.assertFalse(result.matches)
        self.assertEqual(result.reserved_candidate_ids, {1})
        self.assertTrue(all(x["reason"] == "ambiguous" for x in result.diagnostics))

    def test_competing_unready_short_track_still_counts_as_ambiguity(self):
        result = self.matcher().match(5, [view()], [candidate(), candidate(2, frames=(5,))])
        self.assertFalse(result.matches)
        self.assertEqual(result.reserved_candidate_ids, {1, 2})

    def test_distinct_tracks_are_assigned_one_to_one(self):
        far = BOX + np.array([90., 0., 90., 0.])
        result = self.matcher().match(5, [view(7), view(8, box=far)],
                                     [candidate(1), candidate(2, box=far)])
        self.assertEqual({(m.track_id, m.candidate_id) for m in result.matches}, {(7, 1), (8, 2)})

    def test_actual_appearance_resolves_geometric_tie(self):
        c = candidate(features={"rgb": Feature(np.array([1., 0.]), 1.)})
        result = self.matcher().match(5, [view(7, memory=[1., 0.]), view(8, memory=[0., 1.])],
                                     [c], appearance_cost=cosine)
        self.assertEqual([m.track_id for m in result.matches], [7])
        self.assertAlmostEqual(result.matches[0].appearance_cost, 0.)

    def test_current_contradiction_cannot_be_averaged_away(self):
        c = candidate(features={"rgb": Feature(np.array([1., 0.]), 1.)})
        c.observations[-1].features = {"rgb": Feature(np.array([0., 1.]), 1.)}
        result = self.matcher().match(5, [view(memory=[1., 0.])], [c], appearance_cost=cosine)
        self.assertFalse(result.matches)
        self.assertFalse(result.reserved_candidate_ids)
        self.assertEqual(result.diagnostics[0]["reason"], "appearance_contradiction")

    def test_missing_appearance_is_not_perfect_similarity(self):
        m = self.matcher(require_appearance=True)
        c = candidate(features={"rgb": Feature(np.zeros(2), 1.)})
        result = m.match(5, [view(memory=[1., 0.])], [c], appearance_cost=cosine)
        self.assertFalse(result.matches)
        self.assertEqual(result.diagnostics[0]["reason"], "appearance_required")

    def test_future_observation_is_rejected_before_callback(self):
        calls = []
        with self.assertRaisesRegex(ValueError, "Future"):
            self.matcher().match(5, [view()], [candidate(frames=(3, 4, 6))],
                                 appearance_cost=lambda *args: calls.append(args))
        self.assertEqual(calls, [])

    def test_missing_archived_prediction_cannot_be_synthesized(self):
        result = self.matcher().match(5, [view(frames=(3, 5))], [candidate()])
        self.assertFalse(result.matches)
        self.assertEqual(result.diagnostics[0]["reason"], "missing_same_frame_prediction")

    def test_expired_identity_is_never_resurrected(self):
        result = self.matcher().match(35, [view()], [candidate(frames=(33, 34, 35))])
        self.assertFalse(result.matches)
        self.assertFalse(result.reserved_candidate_ids)
        self.assertEqual(result.diagnostics[0]["reason"], "outside_retention")

    def test_high_covariance_cannot_authorize_unbounded_jump(self):
        far = BOX + 10000.
        result = self.matcher().match(5, [view(covariance=1e12)], [candidate(box=far)])
        self.assertFalse(result.matches)
        self.assertEqual(result.diagnostics[0]["reason"], "geometry_gate")

    def test_no_current_hit_no_recovery_or_reservation(self):
        result = self.matcher().match(6, [view()], [candidate()])
        self.assertFalse(result.matches)
        self.assertFalse(result.reserved_candidate_ids)

    def test_candidate_cannot_recover_identity_it_previously_coexisted_with(self):
        result = self.matcher().match(5, [view(last_seen=3)], [candidate()])
        self.assertFalse(result.matches)
        self.assertEqual(result.diagnostics[0]["reason"], "overlaps_last_observation")

    def test_reservation_expires_and_missing_frame_does_not_renew_it(self):
        matcher = self.matcher()
        old = [view(7), view(8)]
        first = matcher.match(3, old, [candidate(frames=(3,))])
        self.assertEqual(first.reserved_candidate_ids, {1})
        matcher.match(4, old, [])  # No current observation, same candidate may survive.
        later = matcher.match(8, old, [candidate(frames=(6, 7, 8))])
        self.assertFalse(later.matches)
        self.assertFalse(later.reserved_candidate_ids)

    def test_input_ids_and_time_must_be_consistent(self):
        m = self.matcher()
        with self.assertRaisesRegex(ValueError, "unique"):
            m.match(5, [view(7), view(7)], [candidate()])
        with self.assertRaisesRegex(ValueError, "nondecreasing"):
            m.match(4, [view(7)], [candidate(frames=(2, 3, 4))])

    def test_invalid_config_fails_early(self):
        for kwargs in ({"min_hits": 6}, {"max_age": 1}, {"max_cost": -1},
                       {"reservation_frames": 1}, {"geometry_weight": 0},
                       {"enabled": "false"}, {"enabled": 1}, {"require_appearance": "false"},
                       {"max_cost": True}, {"max_cost": "0.65"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RecoveryConfig(**kwargs)

    def test_fractional_or_boolean_frame_is_rejected(self):
        for frame in (True, 2.5, "5"):
            with self.subTest(frame=frame), self.assertRaises(ValueError):
                self.matcher().match(frame, [], [])


if __name__ == "__main__":
    unittest.main()
