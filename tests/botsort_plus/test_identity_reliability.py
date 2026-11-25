"""C2 core behavioral contracts; fixtures are not benchmark ReID evidence."""
from dataclasses import replace
import json
import unittest

import numpy as np

from botsort_plus.identity_reliability import IdentityConfig, IdentityReliability
from botsort_plus.types import Feature


E1 = np.array([1., 0.])
E2 = np.array([0., 1.])


def packet(vector=E1, quality=1.):
    return {} if vector is None else {"rgb": Feature(np.asarray(vector).copy(), quality)}


def direction(angle):
    return np.array([np.cos(angle), np.sin(angle)])


def mature(vectors=(E1,), **options):
    engine = IdentityReliability(IdentityConfig(enabled=True, **options))
    engine.begin_frame(1)
    for tid, vector in enumerate(vectors, 1):
        engine.seed(tid, packet(vector), 1)
    for frame in range(2, engine.config.min_history + 1):
        engine.begin_frame(frame)
        for index, vector in enumerate(vectors):
            # Distinct global detections, singleton spatial feasible sets.
            packets = [packet(v) for v in vectors]
            decision = engine.evaluate([index+1], [index], [[.05]], packets, "high", .5)[0][0]
            assert decision.allow_write
            engine.commit(index+1, packets[index], decision, frame)
    engine.begin_frame(engine.config.min_history + 1)
    return engine


def brute_force_pair(engine, ids, indices, geometry, packets, i, j, stage, threshold):
    """Independent slow oracle for the optimized row/column aggregation.

    Enumerate every feasible competitor explicitly; do not call core helpers or
    reuse its distance matrix. This intentionally retains the expensive
    pre-optimization formulation for small randomized parity checks only.
    """
    cfg = engine.config
    def observed(column):
        feature = packets[indices[column]].get("rgb")
        if feature is None:
            return None, 0.
        vector = np.asarray(feature.vector, float)
        norm = np.linalg.norm(vector)
        return (vector / norm if norm > 1e-12 else None), float(feature.quality)
    def pair(row, column):
        state = engine._snapshot.get(ids[row])
        vector, quality = observed(column)
        distance = None if state is None or vector is None else float(np.clip(1-np.dot(state.vector, vector), 0, 2))
        known = (state is not None and vector is not None and state.updates >= cfg.min_history
                 and min(state.quality, quality) >= cfg.min_quality and min(state.quality, quality) > 0)
        return distance, known
    rivals = [(i, column) for column in range(len(indices))
              if column != j and geometry[i, column] <= threshold]
    rivals += [(row, j) for row in range(len(ids))
               if row != i and geometry[row, j] <= threshold]
    own_distance, _ = pair(i, j)
    known = bool(rivals) and all(pair(row, col)[1] for row, col in rivals)
    appearance = (min(pair(row, col)[0] - own_distance for row, col in rivals)
                  if own_distance is not None and known else None)
    geometric = min((geometry[row, col] - geometry[i, j] for row, col in rivals), default=None)
    state = engine._snapshot.get(ids[i])
    vector, quality = observed(j)
    history_updates = state.updates if state is not None else 0
    strong = geometry[i, j] <= cfg.bootstrap_max_iou_cost and (geometric is None or geometric >= cfg.geometry_margin)
    history_agrees = own_distance is not None and own_distance <= cfg.history_max_distance
    reliability, allow, reason = 0., False, "outside_geometry"
    if geometry[i, j] <= threshold:
        if vector is None or quality < cfg.min_quality or quality <= 0:
            reason = "missing_feature" if vector is None else "low_quality"
        elif history_updates < cfg.min_history:
            reliability = None
            if stage not in ("high", "unconfirmed"):
                reason = "warmup_requires_high"
            elif state is not None and not history_agrees:
                reason = "history_conflict"
            elif not strong:
                reason = "warmup_geometry_ambiguous"
            else:
                allow, reason = True, "warmup_geometry_verified"
        else:
            support = max(0., min(1., 1-own_distance/cfg.history_max_distance))
            separation = (max(0., min(1., appearance/cfg.appearance_margin))
                          if appearance is not None else 0.) if rivals else support
            reliability = min(state.quality, quality) * support * separation
            if not history_agrees:
                reason = "history_conflict"
            elif geometry[i, j] > cfg.write_max_iou_cost:
                reason = "write_geometry_weak"
            elif rivals and not known:
                reason = "competition_unknown"
            elif rivals and appearance < cfg.appearance_margin:
                reason = "appearance_ambiguous"
            elif not rivals and not strong:
                reason = "singleton_geometry_weak"
            else:
                allow, reason = True, "identity_verified"
    if stage == "recovery":
        allow, reason = False, "recovered_quarantine"
    return {"competitor_count": len(rivals), "competition_known": known,
            "distance": own_distance, "appearance_margin": appearance, "geometry_margin": geometric,
            "reliability": reliability, "allow_write": allow, "reason": reason,
            "history_updates": history_updates, "quality": quality}


class IdentityConfigTests(unittest.TestCase):
    def test_defaults_and_disabled_ablation(self):
        self.assertFalse(IdentityConfig().enabled)
        self.assertEqual(IdentityConfig().min_history, 3)
        IdentityConfig(association_enabled=False, memory_enabled=False)
        with self.assertRaises(ValueError):
            IdentityConfig(enabled=True, association_enabled=False, memory_enabled=False)

    def test_strict_flags_numbers_and_ranges(self):
        bad = ({"enabled": 1}, {"observe_only": "false"}, {"memory_enabled": 0},
               {"association_enabled": None}, {"min_quality": True}, {"min_quality": -1},
               {"min_quality": float("nan")}, {"min_quality": "0.4"},
               {"min_history": True}, {"min_history": 0}, {"min_history": 2.5},
               {"history_max_distance": 0}, {"history_max_distance": 3},
               {"appearance_margin": 0}, {"geometry_margin": -1},
               {"bootstrap_max_iou_cost": .6}, {"write_max_iou_cost": 1.1},
               {"ema": 1}, {"ema": float("inf")})
        for values in bad:
            with self.subTest(values=values), self.assertRaises(ValueError):
                IdentityConfig(**values)


class IdentityReliabilityTests(unittest.TestCase):
    def test_birth_is_provisional_and_warmup_is_reference_fallback(self):
        engine = IdentityReliability(IdentityConfig(enabled=True))
        engine.begin_frame(1)
        self.assertTrue(engine.seed(1, packet(), 1))
        self.assertEqual(engine.states[1].updates, 1)
        engine.begin_frame(2)
        decision = engine.evaluate([1], [0], [[.1]], [packet()], "high", .5)[0][0]
        self.assertIsNone(decision.reliability)
        self.assertTrue(decision.allow_write)
        self.assertEqual(decision.reason, "warmup_geometry_verified")
        np.testing.assert_array_equal(engine.calibrate([[.1]], [[.02]], [[decision]]), [[.02]])

    def test_missing_low_quality_and_zero_cannot_provide_bonus(self):
        for observation in ({}, packet([0., 0.]), packet(E1, .1), packet(E1, 0.)):
            engine = mature()
            d = engine.evaluate([1], [0], [[.2]], [observation], "high", .5)[0][0]
            self.assertEqual(d.reliability, 0.)
            self.assertFalse(d.allow_write)
            np.testing.assert_array_equal(engine.calibrate([[.2]], [[.01]], [[d]]), [[.2]])
        # Even cold fallback must not bless an unusable current observation.
        cold = IdentityReliability(IdentityConfig(enabled=True))
        cold.begin_frame(1)
        self.assertEqual(cold.evaluate([1], [0], [[.2]], [{}], "high", .5)[0][0].reliability, 0.)

    def test_calibrated_cost_remains_between_reference_and_geometry(self):
        engine = mature((E1, E2))
        packets = [packet(direction(.10)), packet(direction(1.4))]
        decisions = engine.evaluate([1, 2], [0, 1], [[.2, .3], [.3, .2]], packets, "high", .5)
        base = np.array([[.2, .3], [.3, .2]])
        reference = np.array([[.01, .2], [.25, .02]])
        output = engine.calibrate(base, reference, decisions)
        self.assertTrue(np.all(output >= reference))
        self.assertTrue(np.all(output <= base))
        self.assertGreater(output[0, 0], reference[0, 0])
        np.testing.assert_array_equal(base, [[.2, .3], [.3, .2]])

    def test_calibration_limits_and_tolerates_only_roundoff(self):
        engine = mature()
        d = engine.evaluate([1], [0], [[.2]], [packet()], "high", .5)[0][0]
        for r, expected in ((0., .2), (1., .03), (None, .03)):
            self.assertAlmostEqual(engine.calibrate([[.2]], [[.03]], [[replace(d, reliability=r)]])[0, 0], expected)
        self.assertLessEqual(engine.calibrate([[.2]], [[.2 + 1e-14]], [[d]])[0, 0], .2)
        for a, b, ds in (([[.2]], [[.3]], [[d]]), ([[.2]], [[float("nan")]], [[d]]),
                         ([[.2]], [[.1, .2]], [[d]]), ([[.2]], [[.1]], [[]])):
            with self.assertRaises(ValueError):
                engine.calibrate(a, b, ds)

    def test_singleton_has_no_fabricated_infinite_margin(self):
        engine = mature()
        vector = direction(.4)
        d = engine.evaluate([1], [0], [[.1]], [packet(vector, .8)], "high", .5)[0][0]
        h = 1 - (1 - np.cos(.4)) / .4
        self.assertAlmostEqual(d.reliability, .8 * h * h)
        self.assertIsNone(d.appearance_margin)
        self.assertIsNone(d.geometry_margin)
        self.assertEqual(d.competitor_count, 0)
        self.assertFalse(d.competition_known)
        self.assertTrue(d.allow_write)
        self.assertLess(d.reliability, .8)

    def test_singleton_requires_strong_geometry_for_memory(self):
        engine = mature()
        d = engine.evaluate([1], [0], [[.4]], [packet()], "high", .5)[0][0]
        self.assertFalse(d.allow_write)
        self.assertEqual(d.reason, "singleton_geometry_weak")
        self.assertGreater(d.reliability, 0.)

    def test_bidirectional_distinct_competitors_support_correct_pair(self):
        engine = mature((E1, E2))
        decisions = engine.evaluate([1, 2], [0, 1], [[.2, .2], [.2, .2]],
                                    [packet(E1), packet(E2)], "high", .5)
        self.assertTrue(decisions[0][0].allow_write)
        self.assertTrue(decisions[1][1].allow_write)
        self.assertEqual(decisions[0][0].appearance_margin, 1.)
        self.assertEqual(decisions[0][0].reliability, 1.)
        self.assertFalse(decisions[0][1].allow_write)
        self.assertEqual(decisions[0][0].competitor_count, 2)

    def test_row_tie_and_column_tie_both_freeze(self):
        for tracks, indices, geometry, vectors in (
                ([1], [0, 1], [[.1, .15]], (E1, E1)),
                ([1, 2], [0], [[.1], [.15]], (E1,))):
            engine = mature((E1, E1))
            d = engine.evaluate(tracks, indices, geometry, [packet(v) for v in vectors], "high", .5)[0][0]
            self.assertEqual(d.appearance_margin, 0.)
            self.assertFalse(d.allow_write)
            self.assertEqual(d.reliability, 0.)
            self.assertEqual(d.reason, "appearance_ambiguous")

    def test_missing_rival_stays_in_competition(self):
        engine = mature()
        d = engine.evaluate([1], [0, 1], [[.1, .15]], [packet(), {}], "high", .5)[0][0]
        self.assertEqual(d.competitor_count, 1)
        self.assertFalse(d.competition_known)
        self.assertIsNone(d.appearance_margin)
        self.assertEqual(d.reason, "competition_unknown")
        self.assertEqual(d.reliability, 0.)
        self.assertFalse(d.allow_write)

    def test_immature_rival_track_stays_unknown(self):
        engine = mature()
        d = engine.evaluate([1, 2], [0], [[.1], [.2]], [packet()], "high", .5)[0][0]
        self.assertEqual(d.competitor_count, 1)
        self.assertFalse(d.competition_known)
        self.assertFalse(d.allow_write)
        self.assertEqual(d.reliability, 0.)

    def test_geometrically_impossible_rival_does_not_create_ambiguity(self):
        engine = mature()
        d = engine.evaluate([1], [0, 1], [[.1, .9]], [packet(), {}], "high", .5)[0][0]
        self.assertEqual(d.competitor_count, 0)
        self.assertTrue(d.allow_write)

    def test_own_outside_geometry_never_gains_or_writes(self):
        engine = mature()
        d = engine.evaluate([1], [0], [[.6]], [packet()], "high", .5)[0][0]
        self.assertEqual(d.reason, "outside_geometry")
        self.assertEqual(d.reliability, 0.)
        self.assertFalse(d.allow_write)

    def test_history_conflict_freezes_without_geometry_rejection(self):
        engine = mature()
        previous = engine.states[1].vector.copy()
        d = engine.evaluate([1], [0], [[0.]], [packet(E2)], "high", .5)[0][0]
        self.assertEqual(d.reason, "history_conflict")
        self.assertFalse(engine.commit(1, packet(E2), d, 4))
        np.testing.assert_array_equal(engine.states[1].vector, previous)
        self.assertEqual(engine.states[1].updates, 3)
        # The core returns evidence only; it never decides to terminate a track.
        self.assertEqual(d.geometry_cost, 0.)

    def test_provisional_conflict_cannot_self_certify_reset(self):
        engine = IdentityReliability(IdentityConfig(enabled=True))
        engine.begin_frame(1)
        engine.seed(1, packet(), 1)
        engine.begin_frame(2)
        d = engine.evaluate([1], [0], [[0.]], [packet(E2)], "high", .5)[0][0]
        self.assertIsNone(d.reliability)
        self.assertFalse(d.allow_write)
        self.assertEqual(d.reason, "history_conflict")

    def test_cold_seed_requires_unique_strong_geometry(self):
        for geometry in ([[.4]], [[.1, .15]]):
            engine = IdentityReliability(IdentityConfig(enabled=True))
            engine.begin_frame(1)
            packets = [packet() for _ in geometry[0]]
            d = engine.evaluate([1], list(range(len(packets))), geometry, packets, "high", .5)[0][0]
            self.assertFalse(d.allow_write)
            self.assertEqual(d.reason, "warmup_geometry_ambiguous")

    def test_low_cannot_seed_or_mature_but_can_update_trusted_identity(self):
        cold = IdentityReliability(IdentityConfig(enabled=True))
        cold.begin_frame(1)
        d = cold.evaluate([1], [0], [[0.]], [packet()], "low", .5)[0][0]
        self.assertFalse(d.allow_write)
        cold.seed(2, packet(), 1)
        cold.begin_frame(2)
        d = cold.evaluate([2], [0], [[0.]], [packet()], "low", .5)[0][0]
        self.assertFalse(d.allow_write)
        engine = mature()
        d = engine.evaluate([1], [0], [[.05]], [packet(direction(.1))], "low", .5)[0][0]
        self.assertTrue(d.allow_write)
        self.assertTrue(engine.commit(1, packet(direction(.1)), d, 4))
        self.assertEqual(engine.states[1].updates, 4)

    def test_recovery_quarantine_independent_of_geometry(self):
        for geometry in (0., .9):
            engine = mature()
            d = engine.evaluate([1], [0], [[geometry]], [packet()], "recovery", .5)[0][0]
            self.assertEqual(d.reason, "recovered_quarantine")
            self.assertFalse(d.allow_write)
            self.assertFalse(engine.commit(1, packet(), d, 4))
            self.assertEqual(engine.states[1].updates, 3)

    def test_frame_snapshot_is_unchanged_by_an_earlier_commit(self):
        engine = mature((E1, E2))
        observations = [packet(direction(.3)), packet(E2)]
        geometry = [[.1, .2], [.2, .1]]
        before = engine.evaluate([1, 2], [0, 1], geometry, observations, "high", .5)
        self.assertTrue(before[0][0].allow_write)
        engine.commit(1, observations[0], before[0][0], 4)
        after = engine.evaluate([1, 2], [0, 1], geometry, observations, "unconfirmed", .5)
        self.assertEqual(before[0][0].distance, after[0][0].distance)
        self.assertEqual(before[1][1].appearance_margin, after[1][1].appearance_margin)
        engine.begin_frame(5)
        tomorrow = engine.evaluate([1, 2], [0, 1], geometry, observations, "high", .5)
        self.assertLess(tomorrow[0][0].distance, before[0][0].distance)

    def test_current_feature_is_never_its_own_history(self):
        engine = IdentityReliability(IdentityConfig(enabled=True))
        engine.begin_frame(1)
        engine.seed(1, packet(), 1)
        d = engine.evaluate([1], [0], [[0.]], [packet()], "high", .5)[0][0]
        self.assertEqual(d.history_updates, 0)
        self.assertIsNone(d.distance)
        self.assertIsNone(d.reliability)

    def test_gradual_legitimate_change_continues_updating(self):
        engine = mature(ema=.8)
        start = engine.states[1].vector.copy()
        for frame in range(4, 34):
            if frame != 4:
                engine.begin_frame(frame)
            observation = packet(direction(.015 * (frame - 3)))
            d = engine.evaluate([1], [0], [[.1]], [observation], "high", .5)[0][0]
            self.assertTrue(d.allow_write)
            self.assertTrue(engine.commit(1, observation, d, frame))
        self.assertEqual(engine.states[1].updates, 33)
        self.assertGreater(np.linalg.norm(engine.states[1].vector - start), .3)
        self.assertAlmostEqual(np.linalg.norm(engine.states[1].vector), 1.)

    def test_one_commit_per_identity_per_frame(self):
        engine = mature()
        d = engine.evaluate([1], [0], [[.1]], [packet()], "high", .5)[0][0]
        engine.commit(1, packet(), d, 4)
        with self.assertRaises(ValueError):
            engine.commit(1, packet(), d, 4)
        with self.assertRaises(ValueError):
            engine.seed(1, packet(), 4)

    def test_wrong_frame_wrong_identity_and_mutated_packet_rejected(self):
        engine = mature()
        d = engine.evaluate([1], [0], [[.1]], [packet()], "high", .5)[0][0]
        for tid, observation, frame in ((1, packet(), 3), (2, packet(), 4), (1, packet(E2), 4)):
            with self.assertRaises(ValueError):
                engine.commit(tid, observation, d, frame)
        with self.assertRaises(ValueError):
            engine.evaluate([1], [0], [[.1]], [packet(E2)], "low", .5)

    def test_features_are_copied_and_anchor_is_read_only(self):
        engine = IdentityReliability(IdentityConfig(enabled=True))
        engine.begin_frame(1)
        vector = np.array([2., 0.])
        observation = {"rgb": Feature(vector, 1.)}
        engine.seed(1, observation, 1)
        np.testing.assert_array_equal(vector, [2., 0.])
        vector[:] = 7
        np.testing.assert_array_equal(engine.states[1].vector, E1)
        with self.assertRaises(ValueError):
            engine.states[1].vector[0] = 3

    def test_missing_bad_quality_birth_does_not_create_history(self):
        engine = IdentityReliability(IdentityConfig(enabled=True))
        engine.begin_frame(1)
        for tid, observation in enumerate(({}, packet([0., 0.]), packet(E1, .1)), 1):
            self.assertFalse(engine.seed(tid, observation, 1))
        self.assertFalse(engine.states)

    def test_dimensions_and_malformed_embeddings_fail(self):
        for vector in ([float("nan"), 0.], [float("inf"), 0.], [], [[1., 0.]], [1., 0., 0.]):
            engine = mature()
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                engine.evaluate([1], [0], [[.1]], [packet(vector)], "high", .5)
        for quality in (True, -.1, 1.1, float("nan"), "0.5"):
            engine = mature()
            with self.subTest(quality=quality), self.assertRaises(ValueError):
                engine.evaluate([1], [0], [[.1]], [packet(E1, quality)], "high", .5)

    def test_empty_matrices_and_global_detection_indices(self):
        engine = mature()
        self.assertEqual(engine.evaluate([], [], np.empty((0, 0)), [], "high", .5), [])
        self.assertEqual(engine.calibrate(np.empty((0, 2)), np.empty((0, 2)), []).shape, (0, 2))
        d = engine.evaluate([1], [2], [[.1]], [{}, packet(E2), packet()], "high", .5)[0][0]
        self.assertTrue(d.allow_write)
        self.assertEqual(d.det_index, 2)

    def test_invalid_matrix_indices_and_stage_are_rejected(self):
        invalid = (([1, 1], [0], [[.1], [.1]], "high"),
                   ([1], [0, 0], [[.1, .1]], "high"),
                   ([1], [1], [[.1]], "high"),
                   ([1], [0], [[-1.]], "high"),
                   ([1], [0], [[float("nan")]], "high"),
                   ([1], [0], [[.1, .2]], "high"),
                   ([1], [0], [[.1]], "unknown"))
        for ids, indices, matrix, stage in invalid:
            engine = mature()
            with self.subTest(ids=ids, indices=indices, matrix=matrix, stage=stage), self.assertRaises(ValueError):
                engine.evaluate(ids, indices, matrix, [packet()], stage, .5)

    def test_retain_bounds_lifecycle_memory_and_json_is_finite(self):
        engine = mature((E1, E2))
        d = engine.evaluate([1], [0], [[.1]], [packet()], "high", .5)[0][0]
        json.dumps(d.to_dict(), allow_nan=False)
        engine.retain([2])
        self.assertEqual(set(engine.states), {2})
        engine.retain([])
        self.assertFalse(engine.states)
        with self.assertRaises(ValueError):
            engine.begin_frame(4)

    def test_optimized_aggregation_matches_bruteforce_randomized(self):
        rng = np.random.default_rng(4819)
        checked = 0
        for trial in range(80):
            n, m = int(rng.integers(0, 7)), int(rng.integers(0, 8))
            engine = IdentityReliability(IdentityConfig(enabled=True))
            engine.begin_frame(1)
            vectors = rng.normal(size=(max(n, 1), 9))
            # Deliberate tied prototypes, empty histories, immature histories.
            if n > 1 and trial % 3 == 0:
                vectors[1] = vectors[0]
            for row in range(n):
                if rng.random() < .8:
                    engine.seed(row+1, packet(vectors[row], float(rng.choice([.35, .8, 1.]))), 1)
                    engine.states[row+1] = replace(engine.states[row+1], updates=int(rng.choice([1, 2, 3, 5])))
            engine.begin_frame(2)
            observations = [{}]
            for col in range(m):
                kind = int(rng.integers(0, 5))
                vector = None if kind == 0 else np.zeros(9) if kind == 1 else vectors[0] if kind == 2 else rng.normal(size=9)
                observations.append(packet(vector, float(rng.choice([0., .2, .35, .8, 1.]))))
            ids, indices = list(range(1, n+1)), list(range(1, m+1))
            geometry = rng.choice([0., .1, .2, .3, .5, .6, 1.], size=(n, m))
            stage = ("high", "low", "unconfirmed", "recovery")[trial % 4]
            decisions = engine.evaluate(ids, indices, geometry, observations, stage, .5)
            for i in range(n):
                for j in range(m):
                    expected = brute_force_pair(engine, ids, indices, geometry, observations, i, j, stage, .5)
                    actual = decisions[i][j].to_dict()
                    for key, value in expected.items():
                        with self.subTest(trial=trial, row=i, column=j, key=key):
                            if isinstance(value, (float, np.floating)):
                                self.assertAlmostEqual(actual[key], value, places=12)
                            else:
                                self.assertEqual(actual[key], value)
                    checked += 1
        self.assertGreater(checked, 500)

    def test_large_rectangular_1280d_smoke(self):
        # Shape/performance smoke for dense competition, not a dataset FPS test.
        # Avoid a wall-clock assertion: shared CI load is not algorithm behavior.
        rng = np.random.default_rng(912)
        n, m, dimension = 96, 128, 1280
        engine = IdentityReliability(IdentityConfig(enabled=True, min_history=1))
        engine.begin_frame(1)
        vectors = rng.normal(size=(m, dimension))
        observations = [packet(v) for v in vectors]
        for row in range(n):
            engine.seed(row+1, observations[row], 1)
        engine.begin_frame(2)
        geometry = np.full((n, m), .2)
        decisions = engine.evaluate(list(range(1, n+1)), list(range(m)), geometry, observations, "high", .5)
        self.assertEqual(len(decisions), n)
        self.assertTrue(all(len(row) == m for row in decisions))
        self.assertEqual(decisions[0][0].competitor_count, n+m-2)
        self.assertTrue(decisions[0][0].allow_write)
        costs = engine.calibrate(geometry, np.zeros_like(geometry), decisions)
        self.assertTrue(np.isfinite(costs).all())
        self.assertTrue(np.all((costs >= 0) & (costs <= geometry)))


if __name__ == "__main__":
    unittest.main()
