"""Behavioral contracts of A v0.2's new-ID guard; no public IDs or recovery."""
from dataclasses import asdict
from types import SimpleNamespace
import copy
import json
import unittest

import numpy as np

from botsort_plus.birth_identity import BirthIdentityConfig, BirthIdentityGuard
from botsort_plus.geometry import GeometryConfig, box_iou
from botsort_plus.types import Observation, Prediction, LostTrackView


def box(x=50.):
    return np.array([x, 50., x+4., 54.])


def candidate(frames, x=50., cid=1, features=None):
    observations = [Observation(f, 0, box(x), .2, {} if features is None else features[f])
                    for f in frames]
    return SimpleNamespace(candidate_id=cid, observations=observations,
                           last_frame=observations[-1].frame)


def view(frames, x=50., tid=1, last_seen=1, cov=None, memory=None):
    covariance = np.zeros((2, 2)) if cov is None else cov
    return LostTrackView(tid, last_seen,
                         {f: Prediction(f, box(x), covariance.copy()) for f in frames}, memory)


def guard(**kwargs):
    return BirthIdentityGuard(BirthIdentityConfig(**kwargs), GeometryConfig())


class BirthIdentityTests(unittest.TestCase):
    def test_no_lost_identity_is_clear_but_not_a_recovery(self):
        result = guard().inspect(3, candidate([2, 3]), [])
        self.assertEqual(result.status, "CLEAR_NEW")
        self.assertIsNone(result.first_conflict_frame)
        self.assertEqual(result.conflicting_track_ids, [])
        self.assertNotIn("RECOVERED", asdict(result).values())
        json.dumps(asdict(result), allow_nan=False)

    def test_current_far_geometry_excludes_old_identity(self):
        result = guard().inspect(3, candidate([2, 3], x=100), [view([2, 3])])
        self.assertEqual(result.status, "CLEAR_NEW")
        self.assertEqual(result.diagnostics[0]["reason"], "current_geometry_excludes")

    def test_near_current_without_appearance_defers(self):
        result = guard().inspect(3, candidate([2, 3]), [view([2, 3])])
        self.assertEqual(result.status, "DEFER")
        self.assertEqual(result.conflicting_track_ids, [1])
        self.assertEqual(result.diagnostics[0]["reason"], "compatible_geometry_appearance_unknown")

    def test_tiny_boxes_with_low_iou_still_have_identity_risk(self):
        self.assertLess(box_iou(box(), box(53)), .5)
        result = guard().inspect(3, candidate([2, 3], x=53), [view([2, 3])])
        self.assertEqual(result.status, "DEFER")
        self.assertGreater(result.diagnostics[0]["current_geometry"], 0)

    def test_uncertainty_never_removes_the_absolute_search_cap(self):
        result = guard().inspect(3, candidate([2, 3], x=100),
                                 [view([2, 3], cov=np.eye(2)*1e9)])
        self.assertEqual(result.status, "CLEAR_NEW")

    def test_missing_current_prediction_is_unknown_not_far(self):
        result = guard().inspect(3, candidate([2, 3]), [view([2], x=100)])
        self.assertEqual(result.status, "DEFER")
        self.assertEqual(result.diagnostics[0]["reason"], "unknown_current_prediction")

    def test_missing_historical_prediction_does_not_clear_current_risk(self):
        result = guard().inspect(4, candidate([2, 3, 4]), [view([4])])
        self.assertEqual(result.status, "DEFER")
        self.assertEqual(result.diagnostics[0]["reason"], "unknown_history_prediction")

    def test_bad_past_geometry_does_not_clear_a_near_current_pair(self):
        old = view([2, 3])
        old.predictions[2].box = box(200)
        result = guard().inspect(3, candidate([2, 3]), [old])
        self.assertEqual(result.status, "DEFER")
        self.assertEqual(result.diagnostics[0]["reason"], "current_compatible_history_uncertain")
        self.assertIsNone(result.diagnostics[0]["history_geometry"][0]["cost"])

    def test_history_overlapping_last_seen_does_not_certify_new_identity(self):
        result = guard().inspect(4, candidate([2, 3, 4]), [view([2, 3, 4], last_seen=3)])
        self.assertEqual(result.status, "DEFER")
        self.assertEqual(result.diagnostics[0]["history_before_last_seen"], [2, 3])

    def test_trusted_current_appearance_contradiction_can_exclude_old_id(self):
        result = guard().inspect(3, candidate([2, 3]), [view([2, 3])],
                                 appearance_cost=lambda *_: .9)
        self.assertEqual(result.status, "CLEAR_NEW")
        self.assertEqual(result.diagnostics[0]["reason"], "current_appearance_excludes")

    def test_old_appearance_contradiction_with_missing_current_stays_unknown(self):
        c = candidate([2, 3], features={2: {"cost": .9}, 3: {}})
        result = guard().inspect(3, c, [view([2, 3])],
                                 appearance_cost=lambda _m, f: f.get("cost"))
        self.assertEqual(result.status, "DEFER")
        self.assertIsNone(result.diagnostics[0]["current_appearance"])
        self.assertEqual(result.diagnostics[0]["history_appearance_mean"], .9)

    def test_one_excluded_old_id_does_not_hide_another_competitor(self):
        result = guard().inspect(3, candidate([2, 3]),
                                 [view([2, 3], tid=2, memory=.2),
                                  view([2, 3], tid=1, memory=.9)],
                                 appearance_cost=lambda m, _f: m)
        self.assertEqual(result.conflicting_track_ids, [2])
        self.assertEqual(result.status, "DEFER")

    def test_expired_and_currently_observed_views_are_not_lost_candidates(self):
        result = guard(max_age=2).inspect(4, candidate([3, 4]),
                                          [view([3, 4], tid=1, last_seen=1),
                                           view([3, 4], tid=2, last_seen=4)])
        self.assertEqual(result.status, "CLEAR_NEW")
        self.assertTrue(all(d["reason"] == "outside_retention" for d in result.diagnostics))

    def test_nonrenewable_budget_rejects_conflict_at_expiry(self):
        g = guard()
        for f in range(2, 7):
            result = g.inspect(f, candidate(range(2, f+1)), [view(range(2, f+1))])
            self.assertEqual(result.status, "DEFER")
            self.assertEqual(result.first_conflict_frame, 2)
        result = g.inspect(7, candidate(range(2, 8)), [view(range(2, 8))])
        self.assertEqual(result.status, "REJECT")
        self.assertEqual(result.reason, "identity_wait_exhausted")
        self.assertEqual(result.elapsed, 5)

    def test_misses_and_changed_competitor_do_not_reset_clock(self):
        g = guard()
        g.inspect(2, candidate([2]), [view([2])])
        missed = g.inspect(4, candidate([2]), [view([2, 3, 4])])
        self.assertEqual(missed.status, "DEFER")
        self.assertEqual(missed.reason, "no_current_observation")
        self.assertEqual(missed.elapsed, 2)
        final = g.inspect(7, candidate([2, 5, 7]), [view([2, 5, 7], tid=99)])
        self.assertEqual(final.status, "REJECT")
        self.assertEqual(final.first_conflict_frame, 2)

    def test_inherits_earlier_B_start_and_does_not_wait_twice(self):
        g = guard()
        result = g.inspect(6, candidate([4, 5, 6]), [view([4, 5, 6])], reservation_start=2)
        self.assertEqual(result.status, "DEFER")
        self.assertEqual(result.first_conflict_frame, 2)
        self.assertEqual(result.elapsed, 4)
        result = g.inspect(7, candidate([4, 5, 6, 7]), [view([4, 5, 6, 7])], reservation_start=5)
        self.assertEqual(result.status, "REJECT")
        self.assertEqual(result.first_conflict_frame, 2)

    def test_later_clear_observation_does_not_reset_an_expired_clock(self):
        g = guard(wait_frames=2)
        g.inspect(2, candidate([2]), [view([2])])
        blocked = g.inspect(4, candidate([2, 3, 4]), [view([2, 3, 4])])
        self.assertEqual(blocked.status, "REJECT")
        cleared = g.inspect(5, candidate([2, 3, 4, 5]), [])
        self.assertEqual(cleared.status, "CLEAR_NEW")
        self.assertEqual(cleared.first_conflict_frame, 2)
        again = g.inspect(6, candidate([2, 3, 4, 5, 6]), [view([2, 3, 4, 5, 6])])
        self.assertEqual(again.status, "REJECT")
        self.assertEqual(again.first_conflict_frame, 2)

    def test_forget_releases_private_bookkeeping_only(self):
        g = guard()
        c = candidate([2])
        g.inspect(2, c, [view([2])])
        g.forget([1])
        result = g.inspect(3, candidate([2, 3]), [])
        self.assertIsNone(result.first_conflict_frame)
        self.assertEqual(len(c.observations), 1)

    def test_guard_reads_but_does_not_modify_candidate_prediction_or_memory(self):
        c = candidate([2, 3])
        old = view([2, 3], memory={"trusted": np.array([1., 0.])})
        before_c, before_old = copy.deepcopy(c), copy.deepcopy(old)
        result = guard().inspect(3, c, [old], appearance_cost=lambda *_: .1)
        self.assertEqual(result.status, "DEFER")
        for a, b in zip(c.observations, before_c.observations):
            np.testing.assert_array_equal(a.box, b.box)
            self.assertEqual(a.frame, b.frame)
        for f, prediction in old.predictions.items():
            np.testing.assert_array_equal(prediction.box, before_old.predictions[f].box)
            np.testing.assert_array_equal(prediction.center_cov, before_old.predictions[f].center_cov)
        np.testing.assert_array_equal(old.memory["trusted"], before_old.memory["trusted"])
        json.dumps(asdict(result), allow_nan=False)

    def test_disabled_guard_does_not_inspect_or_reserve(self):
        g = guard(enabled=False)
        def must_not_call(*_):
            raise AssertionError("disabled guard must not compute appearance")
        result = g.inspect(3, candidate([2, 3]), [view([2, 3])],
                           appearance_cost=must_not_call, reservation_start=2)
        self.assertEqual(result.status, "CLEAR_NEW")
        self.assertEqual(result.reason, "identity_guard_disabled")
        self.assertIsNone(result.first_conflict_frame)
        self.assertEqual(result.diagnostics, [])

    def test_current_observation_is_required_even_without_lost_tracks(self):
        result = guard().inspect(4, candidate([2, 3]), [])
        self.assertEqual(result.status, "DEFER")
        self.assertEqual(result.reason, "no_current_observation")

    def test_invalid_histories_and_futures_fail_without_mutating_clock(self):
        g = guard()
        for c in (candidate([2, 2]), candidate([3, 2]), candidate([2, 4])):
            with self.subTest(frames=[o.frame for o in c.observations]):
                with self.assertRaises(ValueError):
                    g.inspect(3, c, [view([2, 3])])
        c = candidate([2, 3])
        c.last_frame = 2
        with self.assertRaises(ValueError):
            g.inspect(3, c, [])
        self.assertFalse(g._first_conflict)

    def test_invalid_views_prediction_or_reservation_fail(self):
        invalid = [
            [view([2, 3]), view([2, 3])],
            [view([2, 3], last_seen=4)],
            [view([2, 3, 4])],
        ]
        malformed = view([2, 3])
        malformed.predictions[3].frame = 2
        invalid.append([malformed])
        malformed = view([2, 3])
        malformed.predictions[3].box = np.array([1., 1., 0., 0.])
        invalid.append([malformed])
        malformed = view([2, 3], cov=-np.eye(2))
        invalid.append([malformed])
        for views in invalid:
            with self.subTest(views=views):
                with self.assertRaises(ValueError):
                    guard().inspect(3, candidate([2, 3]), views)
        with self.assertRaises(ValueError):
            guard().inspect(3, candidate([2, 3]), [], reservation_start=4)

    def test_invalid_appearance_results_fail_closed(self):
        for value in (float("nan"), float("inf"), -.1, True, "0.1"):
            with self.subTest(value=value):
                g = guard()
                with self.assertRaises(ValueError):
                    g.inspect(3, candidate([2, 3]), [view([2, 3])],
                              appearance_cost=lambda _m, _f: value)
                self.assertFalse(g._first_conflict)

    def test_config_and_nondecreasing_frame_validation(self):
        for key in ("window", "max_age", "wait_frames"):
            for value in (0, -1, 1.2, True):
                with self.subTest(key=key, value=value):
                    with self.assertRaises(ValueError):
                        BirthIdentityConfig(**{key: value})
        for value in (0, -.1, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                BirthIdentityConfig(appearance_max_cost=value)
        with self.assertRaises(ValueError):
            BirthIdentityConfig(enabled=1)
        g = guard()
        g.inspect(4, candidate([2, 3, 4]), [])
        with self.assertRaises(ValueError):
            g.inspect(3, candidate([2, 3]), [])


if __name__ == "__main__":
    unittest.main()
