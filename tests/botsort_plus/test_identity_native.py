"""C2 integration against the real pinned BoxMOT, Kalman filter and SOF GMC.

Textured images and small unit embeddings are SOFTWARE fixtures only. These
tests make no claims about extracted features or VT-Tiny-MOT accuracy.
"""
from collections import deque
from dataclasses import fields, is_dataclass
import unittest
from unittest.mock import patch

import numpy as np

import test_native_adapter as fixture
from botsort_plus.config import PlusConfig
from botsort_plus.identity_reliability import IdentityConfig
from botsort_plus.native_adapter import BoTSORTPlus
from botsort_plus.recovery import RecoveryConfig


E1, E2 = fixture.E1, fixture.E2
IMAGE = fixture.IMAGE
row, detections, packets = fixture.row, fixture.detections, fixture.packets


def tracker(*, identity=True, recovery=True, **options):
    cfg = PlusConfig(adapter_mode="v2", recovery=RecoveryConfig(enabled=recovery),
                     identity=IdentityConfig(enabled=identity, **options))
    return BoTSORTPlus(cfg, with_reid=True, **fixture.PARAMS)


def frozen(value):
    """Detached exact-value representation of the algorithmic state under test."""
    if isinstance(value, np.ndarray):
        return value.dtype.str, value.shape, value.tobytes()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return tuple((f.name, frozen(getattr(value, f.name))) for f in fields(value))
    if isinstance(value, dict):
        return tuple(sorted((str(k), frozen(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple, deque)):
        return tuple(frozen(v) for v in value)
    if isinstance(value, set):
        return tuple(sorted(frozen(v) for v in value))
    return value


def appearance_state(track):
    return frozen({"smooth": track.smooth_feat, "current": track.curr_feat,
                   "history": list(track.features)})


def bank_state(tracker, identity=1):
    memory = tracker.memories.get(identity)
    if memory is None:
        return None
    return frozen({"prototypes": memory.prototypes, "qualities": memory.qualities,
                   "updates": memory.update_counts})


def reference_state(tracker):
    tracks = {}
    for group in (tracker.active_tracks, tracker.lost_stracks, tracker.removed_stracks):
        for item in group:
            tracks[item.id] = frozen({
                "id": item.id, "state": item.state, "frame": item.frame_id,
                "start": item.start_frame, "activated": item.is_activated,
                "tracklet_len": item.tracklet_len, "mean": item.mean,
                "covariance": item.covariance, "conf": item.conf, "class": item.cls,
                "det_index": item.det_ind, "appearance": appearance_state(item),
                "observations": list(item.history_observations),
            })
    return frozen({
        "native_id_counter": fixture.BaseTrack._count,
        "tracks": tracks,
        "active": [t.id for t in tracker.active_tracks],
        "lost": [t.id for t in tracker.lost_stracks],
        "removed": [t.id for t in tracker.removed_stracks],
        "banks": {tid: bank_state(tracker, tid) for tid in tracker.memories},
        "candidates": {cid: vars(c) for cid, c in tracker.candidate_pool.candidates.items()},
        "candidate_next_id": tracker.candidate_pool._next_id,
        "recovery_clock": tracker.recovery_matcher._first_reserved,
        "recovery_frame": tracker.recovery_matcher._last_frame,
        "prediction_history": tracker.prediction_history,
        "origins": tracker.track_origins,
    })


def advance(tr, dets, feature_packets):
    return tr.update(dets, IMAGE.copy(), features=feature_packets).copy()


def mature(tr):
    for _ in range(3):
        result = advance(tr, detections(row(w=20, h=20)), [packets()])
    return int(result[0, 4])


class IdentityNativeTests(unittest.TestCase):
    def test_same_frame_high_commit_cannot_refresh_unconfirmed_read_snapshot(self):
        tr = tracker()
        for _ in range(2):
            advance(tr, detections(row(w=20, h=20)), [packets(E1)])
        advance(tr, detections(row(w=20, h=20), row(x=100, w=20, h=20)),
                [packets(E1), packets(E2)])
        engine = tr.identity_reliability
        self.assertEqual(engine.states[1].updates, 3)
        self.assertEqual(engine.states[2].updates, 1)
        previous_vector = engine.states[1].vector.copy()
        original_evaluate = engine.evaluate
        observed = []

        def inspect_stage(*args, **kwargs):
            stage = kwargs.get("stage", args[4] if len(args) > 4 else None)
            if stage == "unconfirmed":
                observed.append({"track_ids": list(args[0]),
                                 "live_count": engine.states[1].updates,
                                 "snapshot_count": engine._snapshot[1].updates,
                                 "live_vector": engine.states[1].vector.copy(),
                                 "snapshot_vector": engine._snapshot[1].vector.copy()})
            return original_evaluate(*args, **kwargs)

        near = np.array([np.cos(.1), np.sin(.1)], np.float32)
        with patch.object(engine, "begin_frame", wraps=engine.begin_frame) as begin:
            with patch.object(engine, "evaluate", side_effect=inspect_stage):
                out = advance(tr, detections(row(w=20, h=20), row(x=100, w=20, h=20)),
                              [packets(near), packets(E2)])
        begin.assert_called_once_with(4)
        self.assertEqual(out[:, 4].astype(int).tolist(), [1, 2])
        self.assertEqual(len(observed), 1)
        check = observed[0]
        self.assertEqual(check["track_ids"], [2])
        self.assertEqual(check["live_count"], 4)
        self.assertEqual(check["snapshot_count"], 3)
        np.testing.assert_array_equal(check["snapshot_vector"], previous_vector)
        self.assertFalse(np.array_equal(check["live_vector"], previous_vector))

    def test_config_bridge_defaults_and_round_trip(self):
        baseline = PlusConfig.from_dict({"name": "baseline"})
        self.assertFalse(baseline.enabled)
        self.assertFalse(baseline.identity.enabled)
        configured = PlusConfig.from_dict({"name": "C2", "adapter_mode": "v2",
                                           "identity": {"enabled": True}})
        self.assertTrue(configured.enabled)
        self.assertEqual(configured.identity, IdentityConfig(enabled=True))
        self.assertEqual(configured.identity.min_history, 3)
        self.assertEqual(configured.identity.min_quality, .35)
        self.assertTrue(configured.identity.association_enabled)
        self.assertTrue(configured.identity.memory_enabled)
        self.assertFalse(configured.identity.observe_only)
        self.assertEqual(PlusConfig.from_dict(configured.to_dict()), configured)

    def test_config_bridge_rejects_legacy_a_and_old_c_combinations(self):
        combinations = [
            ({"identity": {"enabled": True}}, "requires adapter_mode=v2"),
            ({"adapter_mode": "v2", "birth": {"enabled": True},
              "identity": {"enabled": True}}, "requires A and legacy C disabled"),
            ({"adapter_mode": "v2", "appearance": {"enabled": True},
              "identity": {"enabled": True}}, "requires A and legacy C disabled"),
        ]
        for data, message in combinations:
            with self.subTest(config=data):
                with self.assertRaisesRegex(ValueError, message):
                    PlusConfig.from_dict(data)

    def test_config_bridge_rejects_unknown_section_keys_and_nonobjects(self):
        combinations = [
            ({"identitiy": {"enabled": True}}, "Unknown experiment options"),
            ({"identity": {"enable": True}}, "Unknown identity options"),
            ({"identity": []}, "identity configuration must be an object"),
            ({"identity": None}, "identity configuration must be an object"),
        ]
        for data, message in combinations:
            with self.subTest(config=data):
                with self.assertRaisesRegex(ValueError, message):
                    PlusConfig.from_dict(data)

    def test_native_constructor_rejects_identity_without_reid(self):
        cfg = PlusConfig(adapter_mode="v2", identity=IdentityConfig(enabled=True))
        with self.assertRaisesRegex(ValueError, "requires with_reid=True"):
            BoTSORTPlus(cfg, with_reid=False, **fixture.PARAMS)

    def test_shadow_is_exact_reference_for_outputs_and_all_shared_state(self):
        sequence = [
            (detections(row(x=30, w=20, h=20), row(x=90, w=20, h=20)),
             [packets(E1), packets(E2)]),
            (detections(row(x=30.5, w=20, h=20), row(x=90, w=20, h=20, score=.3),
                        row(x=125, w=12, h=12)), [packets(E1), packets(E2), packets(E1)]),
            (detections(row(x=31, w=20, h=20), row(x=90.5, w=20, h=20),
                        row(x=125, w=12, h=12), row(x=5, score=.3)),
             [packets(E1), packets(E2), packets(E1), {}]),
            (detections(row(x=31, w=20, h=20)), [packets(E2, rgb_quality=.1)]),
            (detections(row(x=31, w=20, h=20, score=.3), row(x=90.5, w=20, h=20, score=.3)),
             [packets(E1), packets(E2)]),
            (detections(row(x=31, w=20, h=20), row(x=90.5, w=20, h=20, score=.3)),
             [packets(E1), packets(E2)]),
            (detections(row(x=31, w=20, h=20), row(x=90.5, w=20, h=20, score=.3)),
             [packets(E1), packets(E2)]),
            (detections(), []),
            (detections(row(x=31, w=20, h=20), row(x=125, w=12, h=12)),
             [packets(E1), packets(E2)]),
        ]
        ref = tracker(identity=False)
        expected, states, reference_events = [], [], []
        for dets, fs in sequence:
            expected.append(advance(ref, dets, fs))
            states.append(reference_state(ref))
            reference_events.extend(ref.last_events)
        self.assertTrue(any(e["event"] == "recovered" for e in reference_events))
        self.assertTrue(any(e.get("stage") == "unconfirmed" for e in reference_events))
        shadow = tracker(observe_only=True)
        shadow_events = []
        for frame, ((dets, fs), output, state) in enumerate(zip(sequence, expected, states), 1):
            actual = advance(shadow, dets, fs)
            with self.subTest(frame=frame):
                self.assertEqual((output.shape, output.dtype, output.tobytes()),
                                 (actual.shape, actual.dtype, actual.tobytes()))
                self.assertEqual(state, reference_state(shadow))
            shadow_events.extend(shadow.last_events)
        self.assertTrue(any(e["event"] == "identity_decision" for e in shadow_events))
        self.assertTrue(any(e.get("reason") == "recovered_quarantine" for e in shadow_events))

    def test_memory_gate_keeps_geometric_track_but_freezes_both_appearance_paths(self):
        tr = tracker(association_enabled=False)
        tid = mature(tr)
        track = tr.active_tracks[0]
        native, custom, prior_position = appearance_state(track), bank_state(tr, tid), track.mean.copy()
        out = advance(tr, detections(row(x=51, w=20, h=20)), [packets(E2)])
        self.assertEqual(out[:, 4].astype(int).tolist(), [tid])
        self.assertFalse(np.array_equal(track.mean, prior_position))
        self.assertEqual(track.frame_id, 4)
        self.assertEqual(native, appearance_state(track))
        self.assertEqual(custom, bank_state(tr, tid))
        event = next(e for e in tr.last_events if e["event"] == "identity_decision")
        self.assertEqual(event["reason"], "history_conflict")
        self.assertFalse(event["native_feature_written"])
        self.assertEqual(event["custom_modalities_written"], [])

    def test_low_matches_preserve_native_no_write_semantics_and_gate_custom_memory(self):
        tr = tracker(association_enabled=False)
        tid = mature(tr)
        track = tr.active_tracks[0]
        native, custom = appearance_state(track), bank_state(tr, tid)
        advance(tr, detections(row(w=20, h=20, score=.3)), [packets(E2)])
        self.assertEqual(native, appearance_state(track))
        self.assertEqual(custom, bank_state(tr, tid))
        before_count = tr.memories[tid].update_counts["rgb"]
        near = np.array([np.cos(.1), np.sin(.1)], np.float32)
        advance(tr, detections(row(w=20, h=20, score=.3)), [packets(near)])
        self.assertEqual(native, appearance_state(track))
        self.assertNotEqual(custom, bank_state(tr, tid))
        self.assertEqual(tr.memories[tid].update_counts["rgb"], before_count + 1)
        event = next(e for e in tr.last_events if e["event"] == "identity_decision")
        self.assertEqual(event["stage"], "low")
        self.assertTrue(event["allow_write"])
        self.assertFalse(event["native_feature_written"])
        self.assertEqual(event["custom_modalities_written"], ["rgb"])

    def test_real_b_recovery_quarantines_memory_then_reliable_native_update_resumes(self):
        tr = tracker()
        tid = mature(tr)
        original = tr.active_tracks[0]
        native, custom = appearance_state(original), bank_state(tr, tid)
        advance(tr, detections(), [])
        near = np.array([np.cos(.1), np.sin(.1)], np.float32)
        outputs = []
        for _ in range(3):
            outputs.append(advance(tr, detections(row(w=20, h=20, score=.3)), [packets(near)]))
        self.assertEqual([len(out) for out in outputs], [0, 0, 1])
        self.assertEqual(outputs[-1][:, 4].astype(int).tolist(), [tid])
        self.assertTrue(any(e["event"] == "recovered" for e in tr.last_events))
        self.assertEqual(native, appearance_state(original))
        self.assertEqual(custom, bank_state(tr, tid))
        event = next(e for e in tr.last_events if e["event"] == "identity_decision")
        self.assertEqual(event["reason"], "recovered_quarantine")
        self.assertFalse(event["native_feature_written"])
        self.assertEqual(event["custom_modalities_written"], [])
        advance(tr, detections(row(w=20, h=20)), [packets(near)])
        self.assertNotEqual(native, appearance_state(original))
        self.assertNotEqual(custom, bank_state(tr, tid))
        event = next(e for e in tr.last_events if e["event"] == "identity_decision")
        self.assertTrue(event["allow_write"])
        self.assertTrue(event["native_feature_written"])

    def test_low_quality_birth_has_no_native_seed_and_can_bootstrap_on_good_high(self):
        tr = tracker()
        out = advance(tr, detections(row(w=20, h=20)), [packets(E1, rgb_quality=.1)])
        tid = int(out[0, 4])
        track = tr.active_tracks[0]
        self.assertIsNone(track.smooth_feat)
        self.assertIsNone(track.curr_feat)
        self.assertEqual(len(track.features), 0)
        self.assertIsNone(bank_state(tr, tid))
        self.assertNotIn(tid, tr.identity_reliability.states)
        for _ in range(3):
            advance(tr, detections(row(w=20, h=20)), [packets()])
        self.assertEqual(tr.identity_reliability.states[tid].updates, 3)
        np.testing.assert_allclose(track.smooth_feat, E1)
        self.assertEqual(tr.memories[tid].update_counts["rgb"], 3)

    def test_association_only_does_not_apply_memory_gate(self):
        tr = tracker(memory_enabled=False)
        tid = mature(tr)
        track = tr.active_tracks[0]
        native, custom = appearance_state(track), bank_state(tr, tid)
        advance(tr, detections(row(w=20, h=20)), [packets(E2)])
        self.assertNotEqual(native, appearance_state(track))
        self.assertNotEqual(custom, bank_state(tr, tid))
        event = next(e for e in tr.last_events if e["event"] == "identity_decision")
        self.assertFalse(event["allow_write"])
        self.assertFalse(event["memory_gate_applied"])
        self.assertTrue(event["native_feature_written"])
        self.assertEqual(event["custom_modalities_written"], ["rgb"])
        self.assertEqual(tr.identity_reliability.states[tid].updates, 3)

    def test_cache_and_detection_inputs_are_not_normalized_in_place(self):
        tr = tracker()
        dets = detections(row(w=20, h=20))
        original_dets = dets.copy()
        fs = [packets(np.array([3., 4.], np.float32))]
        original_feature = fs[0]["rgb"].vector.copy()
        dets.setflags(write=False)
        fs[0]["rgb"].vector.setflags(write=False)
        for _ in range(4):
            advance(tr, dets, fs)
        np.testing.assert_array_equal(dets, original_dets)
        np.testing.assert_array_equal(fs[0]["rgb"].vector, original_feature)

    def test_mature_appearance_ties_remove_bonus_in_actual_high_assignment(self):
        tr = tracker()
        for _ in range(3):
            advance(tr, detections(row(x=50, w=20, h=20), row(x=58, w=20, h=20)),
                    [packets(E1), packets(E1)])
        self.assertEqual([s.updates for s in tr.identity_reliability.states.values()], [3, 3])
        out = advance(tr, detections(row(x=52, w=20, h=20), row(x=56, w=20, h=20)),
                      [packets(E1), packets(E1)])
        self.assertEqual({int(r[4]): int(r[7]) for r in out}, {1: 0, 2: 1})
        summary = next(e for e in tr.last_events
                       if e["event"] == "identity_association" and e["stage"] == "high")
        self.assertEqual(summary["pair_count"], 4)
        self.assertEqual(summary["proposed_changed_pairs"], 4)
        self.assertGreater(summary["proposed_max_cost_increase"], 0)
        events = [e for e in tr.last_events if e["event"] == "identity_decision"]
        self.assertEqual(len(events), 2)
        for event in events:
            self.assertEqual(event["reliability"], 0.)
            self.assertEqual(event["reason"], "appearance_ambiguous")
            self.assertAlmostEqual(event["reference_cost"], 0.)
            self.assertGreaterEqual(event["applied_cost"], event["reference_cost"])
            self.assertLessEqual(event["applied_cost"], event["geometry_cost"] + 1e-12)
            self.assertAlmostEqual(event["applied_cost"], event["geometry_cost"])


if __name__ == "__main__":
    unittest.main()
