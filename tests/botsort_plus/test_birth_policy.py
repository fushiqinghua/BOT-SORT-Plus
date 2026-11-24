"""Causal confirmation checks; synthetic examples are not benchmark evidence."""
from collections import deque
from dataclasses import asdict
import json
from types import SimpleNamespace
import unittest

import numpy as np

from botsort_plus.birth_policy import BirthPolicy, BirthPolicyConfig
from botsort_plus.candidates import BirthConfig, Candidate, CandidatePool
from botsort_plus.geometry import GeometryConfig
from botsort_plus.types import Observation


def observation(frame, score=0.2):
    return Observation(frame, 0, np.array([10., 10., 14., 14.]), score)


def candidate(cid=1):
    return Candidate(cid, deque(), BirthConfig(), 1000., np.array([10., 10., 14., 14.]))


class BirthPolicyTests(unittest.TestCase):
    def step(self, policy, tracklet, frame, score=0.2):
        if score is not None:
            tracklet.observations.append(observation(frame, score))
        return policy.evaluate(frame, tracklet)

    def test_near_floor_stability_does_not_become_existence_support(self):
        policy, tracklet = BirthPolicy(BirthPolicyConfig()), candidate()
        decisions = [self.step(policy, tracklet, f, .10001) for f in range(1, 21)]
        self.assertTrue(all(not d.ready for d in decisions))
        self.assertTrue(all(d.stats["armed_frame"] is None for d in decisions))
        self.assertEqual(decisions[-1].reason, "insufficient_score_support")
        self.assertEqual(tracklet.evidence, 1000.)

    def test_five_supported_weak_frames_confirm_on_current_fifth_frame(self):
        policy, tracklet = BirthPolicy(BirthPolicyConfig()), candidate()
        decisions = [self.step(policy, tracklet, f) for f in range(1, 6)]
        self.assertEqual([d.ready for d in decisions], [False, False, False, False, True])
        self.assertEqual(decisions[2].phase, "VERIFYING")
        self.assertEqual(decisions[-1].stats["armed_frames"], [1, 2, 3])
        self.assertEqual(decisions[-1].stats["verify_frames"], [4, 5])
        self.assertEqual(decisions[-1].stats["verified_at"], 5)
        self.assertTrue(all(o.score < .5 for o in tracklet.observations))
        self.assertEqual(len(tracklet.observations), 5)

    def test_first_window_spike_cannot_be_reused_as_new_score_support(self):
        policy, tracklet = BirthPolicy(BirthPolicyConfig()), candidate()
        scores = [.5, .1, .1, .1, .1, .1]
        decisions = [self.step(policy, tracklet, f, score)
                     for f, score in enumerate(scores, 1)]
        self.assertEqual(decisions[2].phase, "VERIFYING")
        self.assertTrue(all(not d.ready for d in decisions))
        self.assertEqual(decisions[4].reason, "insufficient_new_score_support")
        self.assertEqual(decisions[-1].reason, "verification_timeout")
        self.assertEqual(decisions[-1].stats["verify_frames"], [4, 5, 6])

    def test_verify_deadline_counts_missing_video_frames(self):
        policy, tracklet = BirthPolicy(BirthPolicyConfig()), candidate()
        for frame in range(1, 4):
            self.step(policy, tracklet, frame)
        self.step(policy, tracklet, 4, None)
        self.step(policy, tracklet, 5)
        failed = self.step(policy, tracklet, 6, None)
        self.assertEqual(failed.reason, "verification_timeout")
        self.assertEqual(failed.stats["verification_deadline"], 6)
        self.assertEqual(failed.stats["verify_frames"], [5])
        self.assertEqual(failed.stats["next_epoch_start"], 6)

    def test_success_on_last_allowed_frame_is_accepted(self):
        policy, tracklet = BirthPolicy(BirthPolicyConfig()), candidate()
        for frame in range(1, 4):
            self.step(policy, tracklet, frame)
        self.step(policy, tracklet, 4)
        self.step(policy, tracklet, 5, None)
        result = self.step(policy, tracklet, 6)
        self.assertTrue(result.ready)
        self.assertEqual(result.stats["verify_frames"], [4, 6])

    def test_failed_epoch_cannot_reuse_past_observations(self):
        policy, tracklet = BirthPolicy(BirthPolicyConfig()), candidate()
        for frame in range(1, 4):
            self.step(policy, tracklet, frame)
        for frame in range(4, 7):
            self.step(policy, tracklet, frame, .1)
        after = [self.step(policy, tracklet, frame) for frame in range(7, 12)]
        self.assertEqual(after[0].stats["collect_frames"], [7])
        self.assertEqual(after[2].stats["armed_frames"], [7, 8, 9])
        self.assertEqual([d.ready for d in after], [False, False, False, False, True])
        self.assertEqual(after[-1].stats["verify_frames"], [10, 11])
        self.assertEqual(len(tracklet.observations), 11)  # B history was not erased.

    def test_old_ready_requires_fresh_support_and_current_detection(self):
        policy, tracklet = BirthPolicy(BirthPolicyConfig()), candidate()
        for frame in range(1, 6):
            last = self.step(policy, tracklet, frame)
        self.assertTrue(last.ready)
        missed = self.step(policy, tracklet, 6, None)
        self.assertFalse(missed.ready)
        self.assertEqual(missed.phase, "READY_STALE")
        self.step(policy, tracklet, 7, None)
        one = self.step(policy, tracklet, 8)
        self.assertFalse(one.ready)
        self.assertEqual(one.reason, "insufficient_recent_hits")
        two = self.step(policy, tracklet, 9)
        self.assertTrue(two.ready)
        self.assertEqual(two.stats["verify_frames"], [8, 9])
        self.assertEqual(two.stats["verified_at"], 5)

    def test_old_ready_is_invalidated_by_new_low_scores(self):
        policy, tracklet = BirthPolicy(BirthPolicyConfig()), candidate()
        for frame in range(1, 6):
            self.step(policy, tracklet, frame, .2)
        stale = self.step(policy, tracklet, 6, .1)
        self.assertFalse(stale.ready)
        self.assertEqual(stale.reason, "insufficient_recent_score_support")

    def test_collection_window_counts_real_gaps(self):
        policy, tracklet = BirthPolicy(BirthPolicyConfig()), candidate()
        for frame, score in enumerate([.2, None, .2, None, .2], 1):
            armed = self.step(policy, tracklet, frame, score)
        self.assertEqual(armed.stats["armed_frames"], [1, 3, 5])
        self.assertEqual(armed.phase, "VERIFYING")
        self.step(policy, tracklet, 6, None)
        self.step(policy, tracklet, 7)
        result = self.step(policy, tracklet, 8)
        self.assertTrue(result.ready)
        self.assertEqual(result.stats["verify_frames"], [7, 8])

    def test_collection_window_drops_old_hits(self):
        policy, tracklet = BirthPolicy(BirthPolicyConfig()), candidate()
        for frame, score in enumerate([.2, None, None, None, .2, .2], 1):
            result = self.step(policy, tracklet, frame, score)
        self.assertFalse(result.ready)
        self.assertEqual(result.stats["collect_frames"], [5, 6])
        self.assertEqual(result.phase, "COLLECTING")

    def test_explicit_modes_have_different_confirmation_semantics(self):
        first_ready = {}
        for mode in ("legacy", "score_window", "fixed_window", "two_stage"):
            policy = BirthPolicy(BirthPolicyConfig(mode=mode))
            tracklet = candidate()
            for frame in range(1, 7):
                result = self.step(policy, tracklet, frame)
                if result.ready:
                    first_ready.setdefault(mode, frame)
        self.assertEqual(first_ready,
                         {"legacy": 3, "score_window": 3, "fixed_window": 5, "two_stage": 5})

    def test_legacy_obeys_original_evidence_threshold(self):
        policy, tracklet = BirthPolicy(BirthPolicyConfig(mode="legacy")), candidate()
        tracklet.evidence = 0.
        for frame in range(1, 6):
            result = self.step(policy, tracklet, frame, .9)
        self.assertFalse(result.ready)
        self.assertEqual(result.reason, "legacy_not_ready")
        self.assertEqual(result.stats["legacy_evidence"], 0.)

    def test_new_modes_do_not_access_geometric_evidence(self):
        class EvidenceForbidden:
            candidate_id = 1
            observations = [observation(1), observation(2), observation(3)]
            last_frame = 3

            @property
            def evidence(self):
                raise AssertionError("New policy must not read geometric evidence")

            def ready(self, frame):
                raise AssertionError("New policy must not delegate to legacy confirmation")

        for mode in ("score_window", "fixed_window", "two_stage"):
            result = BirthPolicy(BirthPolicyConfig(mode=mode)).evaluate(3, EvidenceForbidden())
            self.assertNotIn("legacy_evidence", result.stats)

    def test_same_frame_is_idempotent_and_decision_stats_are_not_shared(self):
        policy, tracklet = BirthPolicy(BirthPolicyConfig()), candidate()
        for frame in range(1, 4):
            armed = self.step(policy, tracklet, frame)
        repeat = policy.evaluate(3, tracklet)
        self.assertEqual(armed, repeat)
        armed.stats["collect_frames"].append(100)
        self.assertEqual(policy.evaluate(3, tracklet).stats["collect_frames"], [1, 2, 3])
        tracklet.observations[-1].score = .3
        with self.assertRaisesRegex(ValueError, "same frame"):
            policy.evaluate(3, tracklet)

    def test_observe_only_preserves_decisions_and_history(self):
        policy = BirthPolicy(BirthPolicyConfig(observe_only=True))
        tracklet = candidate()
        for frame in range(1, 6):
            result = self.step(policy, tracklet, frame)
        self.assertTrue(result.ready)
        self.assertEqual(len(tracklet.observations), 5)
        json.dumps(asdict(result), allow_nan=False)

    def test_forget_clears_only_private_state(self):
        policy, a, b = BirthPolicy(BirthPolicyConfig()), candidate(1), candidate(2)
        for frame in range(1, 4):
            self.step(policy, a, frame)
            self.step(policy, b, frame)
        policy.forget([1, 99])
        self.assertEqual(set(policy.states), {2})
        self.assertEqual(len(a.observations), 3)
        self.assertEqual(policy.states[2].armed_frame, 3)

    def test_shared_pool_and_candidate_are_not_mutated(self):
        pool = CandidatePool(BirthConfig(max_candidates=4), GeometryConfig())
        policy = BirthPolicy(BirthPolicyConfig())
        for frame in range(1, 6):
            tracklet = pool.update(frame, [observation(frame)])[0]
            before = (tracklet.evidence, tuple(tracklet.observations),
                      tracklet.velocity.copy(), tracklet.last_box.copy(),
                      tuple(pool.events), pool._next_id)
            policy.evaluate(frame, tracklet)
            self.assertEqual(tracklet.evidence, before[0])
            self.assertEqual(tuple(tracklet.observations), before[1])
            np.testing.assert_array_equal(tracklet.velocity, before[2])
            np.testing.assert_array_equal(tracklet.last_box, before[3])
            self.assertEqual(tuple(pool.events), before[4])
            self.assertEqual(pool._next_id, before[5])
        self.assertEqual(set(pool.candidates), {1})

    def test_future_duplicate_or_unsorted_observations_fail(self):
        examples = ([observation(1), observation(3)],
                    [observation(1), observation(1)],
                    [observation(2), observation(1)])
        for history in examples:
            with self.subTest(frames=[o.frame for o in history]):
                tracklet = SimpleNamespace(candidate_id=1, observations=history,
                                           last_frame=history[-1].frame)
                with self.assertRaises(ValueError):
                    BirthPolicy(BirthPolicyConfig()).evaluate(2, tracklet)

    def test_inconsistent_or_noninteger_observation_metadata_fail(self):
        for frame, score in ((True, .2), (1.5, .2), (1, float("nan")),
                             (1, float("inf")), (1, True), (1, -.1)):
            tracklet = SimpleNamespace(candidate_id=1, last_frame=frame,
                                       observations=[SimpleNamespace(frame=frame, score=score)])
            with self.subTest(frame=frame, score=score), self.assertRaises(ValueError):
                BirthPolicy(BirthPolicyConfig()).evaluate(2, tracklet)
        inconsistent = SimpleNamespace(candidate_id=1, last_frame=2,
                                       observations=[observation(1)])
        with self.assertRaisesRegex(ValueError, "last_frame"):
            BirthPolicy(BirthPolicyConfig()).evaluate(2, inconsistent)

    def test_skipped_frame_and_backward_clock_fail(self):
        policy, tracklet = BirthPolicy(BirthPolicyConfig()), candidate()
        self.step(policy, tracklet, 1)
        with self.assertRaisesRegex(ValueError, "consecutive"):
            self.step(policy, tracklet, 3)
        other = candidate(2)
        self.step(policy, other, 3)
        with self.assertRaisesRegex(ValueError, "nondecreasing"):
            policy.evaluate(2, tracklet)

    def test_invalid_frame_candidate_and_configuration_fail(self):
        for value in (False, 0, -1, 1.5, float("nan")):
            tracklet = candidate()
            tracklet.observations.append(observation(1))
            with self.subTest(frame=value), self.assertRaises(ValueError):
                BirthPolicy(BirthPolicyConfig()).evaluate(value, tracklet)
        for value in (False, 0, -1, 1.5):
            tracklet = candidate(value)
            tracklet.observations.append(observation(1))
            with self.subTest(candidate=value), self.assertRaises(ValueError):
                BirthPolicy(BirthPolicyConfig()).evaluate(1, tracklet)
        invalid = ({"mode": "probability"}, {"collect_hits": 6}, {"collect_window": 0},
                   {"verify_hits": 4}, {"verify_window": 1.5}, {"fixed_hits": 9},
                   {"fixed_window": True}, {"mean_score": float("nan")},
                   {"mean_score": float("inf")}, {"mean_score": True},
                   {"mean_score": -.1}, {"mean_score": 1.1}, {"observe_only": 1})
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(ValueError):
                BirthPolicyConfig(**options)


if __name__ == "__main__":
    unittest.main()
