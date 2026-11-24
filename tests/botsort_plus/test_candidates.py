"""Synthetic mechanism checks, not claims about dataset accuracy."""
import unittest

import numpy as np

from botsort_plus.candidates import BirthConfig, CandidatePool
from botsort_plus.geometry import GeometryConfig
from botsort_plus.types import Observation


def obs(frame, index=0, x=10.0, y=10.0, score=0.3, size=4.0):
    return Observation(frame, index, np.array([x, y, x + size, y + size]), score)


class CandidateMechanismTests(unittest.TestCase):
    def pool(self, **kwargs):
        return CandidatePool(BirthConfig(**kwargs), GeometryConfig())

    def test_three_tiny_jitter_hits_confirm_only_now(self):
        pool = self.pool()
        prior = []
        for frame, x in enumerate([10.0, 10.2, 10.4], 1):
            current = pool.update(frame, [obs(frame, x=x)])
            self.assertEqual(len(current), 1)
            prior.append([(c.candidate_id, c.ready(frame)) for c in current])
        self.assertFalse(prior[0][0][1])
        self.assertFalse(prior[1][0][1])
        self.assertTrue(prior[2][0][1])
        candidate = current[0]
        self.assertEqual(candidate.hit_count(3, 5), 3)
        self.assertFalse(candidate.ready(2))
        self.assertFalse(candidate.ready(4))

    def test_isolated_false_positive_expires(self):
        pool = self.pool()
        candidate = pool.update(1, [obs(1)])[0]
        for frame in (2, 3, 4):
            self.assertEqual(pool.update(frame, []), [])
        self.assertIsNone(pool.get(candidate.candidate_id))

    def test_empty_frames_advance_clock_and_no_backfill(self):
        pool = self.pool()
        self.assertEqual(pool.update(1, []), [])
        self.assertEqual(pool.update(2, []), [])
        for frame in (3, 4, 5):
            current = pool.update(frame, [obs(frame)])
        self.assertTrue(current[0].ready(5))
        self.assertEqual([o.frame for o in current[0].observations], [3, 4, 5])
        # CandidatePool never emits a MOT row/public ID. Consumption prevents a
        # second birth or recovery from the same accumulated observations.
        candidate_id = current[0].candidate_id
        self.assertIs(pool.remove(candidate_id), current[0])
        self.assertIsNone(pool.discard(candidate_id))

    def test_missing_hit_reduces_evidence_and_requires_current_hit(self):
        pool = self.pool()
        for frame in (1, 2, 3):
            candidate = pool.update(frame, [obs(frame)])[0]
        evidence = candidate.evidence
        pool.update(4, [])
        self.assertLess(candidate.evidence, evidence)
        self.assertFalse(candidate.ready(4))
        self.assertAlmostEqual(candidate.evidence, max(0.0, 0.8 * evidence - 0.4))

    def test_fixed_hits_is_real_ablation(self):
        pool = self.pool(mode="fixed_hits", evidence_threshold=1000.0)
        for frame in (1, 2, 3):
            candidate = pool.update(frame, [obs(frame)])[0]
        self.assertTrue(candidate.ready(3))
        adaptive = self.pool(evidence_threshold=1000.0)
        for frame in (1, 2, 3):
            candidate = adaptive.update(frame, [obs(frame)])[0]
        self.assertFalse(candidate.ready(3))

    def test_duplicate_detection_cannot_double_evidence(self):
        pool = self.pool()
        current = pool.update(1, [obs(1, 0, score=0.3), obs(1, 1, score=0.4)])
        self.assertEqual(len(current), 1)
        self.assertEqual(current[0].latest.det_index, 1)
        self.assertEqual(current[0].hit_count(1, 5), 1)
        with self.assertRaises(ValueError):
            pool.update(2, [obs(2, 0), obs(2, 0)])

    def test_occupied_track_invalidates_candidate_before_evidence(self):
        pool = self.pool()
        candidate = pool.update(1, [obs(1)])[0]
        current = pool.update(2, [obs(2)], occupied_boxes=[obs(2).box])
        self.assertEqual(current, [])
        self.assertIsNone(pool.get(candidate.candidate_id))

    def test_ambiguous_crossing_rejects_hit_and_new_seed(self):
        pool = self.pool(ambiguity_margin=0.2)
        first = pool.update(1, [obs(1, 0, x=10.0), obs(1, 1, x=13.0)])
        self.assertEqual(len(first), 2)
        current = pool.update(2, [obs(2, 0, x=11.5)])
        self.assertEqual(current, [])
        self.assertEqual(len(pool.candidates), 2)
        self.assertTrue(any(e["event"] == "candidate_seed_deferred" for e in pool.events))
        self.assertEqual([c.hit_count(2, 5) for c in pool.candidates.values()], [1, 1])

    def test_gmc_transports_prediction_not_historical_observations(self):
        pool = self.pool()
        first = obs(1)
        candidate = pool.update(1, [first])[0]
        warp = np.array([[1.0, 0.0, 20.0], [0.0, 1.0, -3.0]])
        current = pool.update(2, [obs(2, x=30.0, y=7.0)], warp=warp)
        self.assertEqual(len(current), 1)
        self.assertEqual(current[0].candidate_id, candidate.candidate_id)
        np.testing.assert_array_equal(candidate.observations[0].box, [10.0, 10.0, 14.0, 14.0])
        np.testing.assert_allclose(candidate.velocity, [0.0, 0.0], atol=1e-12)

    def test_window_counts_do_not_keep_old_hits_forever(self):
        pool = self.pool(min_hits=3, window=3, max_misses=2, mode="fixed_hits")
        pool.update(1, [obs(1)])
        pool.update(2, [obs(2)])
        pool.update(3, [])
        current = pool.update(4, [obs(4)])
        self.assertEqual(len(current), 1)
        self.assertEqual(current[0].hit_count(4, 3), 2)
        self.assertFalse(current[0].ready(4))

    def test_expired_candidate_gets_fresh_identity_and_capacity_is_bounded(self):
        pool = self.pool(max_candidates=2, max_age=5)
        pool.update(1, [obs(1, 0, x=0), obs(1, 1, x=20), obs(1, 2, x=40)])
        self.assertEqual(len(pool.candidates), 2)
        self.assertTrue(any(e.get("reason") == "capacity" for e in pool.events))

    def test_nonconsecutive_and_invalid_configuration_fail_closed(self):
        pool = self.pool()
        pool.update(1, [])
        with self.assertRaises(ValueError):
            pool.update(3, [])
        with self.assertRaises(ValueError):
            pool.update(2, [obs(3)])
        with self.assertRaises(ValueError):
            BirthConfig(window=2, min_hits=3)
        with self.assertRaises(ValueError):
            BirthConfig(decay=float("nan"))


if __name__ == "__main__":
    unittest.main()
