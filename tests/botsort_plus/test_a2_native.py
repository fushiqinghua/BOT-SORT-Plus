"""Independent integration contracts against real pinned BoxMOT/KF/SOF.

Images and embeddings are synthetic software fixtures, not dataset measurements.
Assertions concern causal output, old/new identity ownership and native parity.
"""
import copy
import importlib.util
import unittest

import numpy as np

for _dependency in ("boxmot", "torch", "cv2"):
    if importlib.util.find_spec(_dependency) is None:
        raise unittest.SkipTest(f"A2 native integration tests require {_dependency}")

from boxmot.trackers.botsort.basetrack import BaseTrack, TrackState
from boxmot.trackers.botsort.botsort import BotSort

from botsort_plus.birth_identity import BirthIdentityConfig
from botsort_plus.birth_policy import BirthPolicyConfig
from botsort_plus.candidates import BirthConfig
from botsort_plus.config import PlusConfig
from botsort_plus.native_adapter import BoTSORTPlus, CacheOnlyEncoder
from botsort_plus.recovery import RecoveryConfig

from test_native_adapter import PARAMS, IMAGE, E1, row, detections, packets, dense_features


def a2(*, birth=True, recovery=False, policy=None, identity=None, params=None):
    native_params = dict(PARAMS)
    if params:
        native_params.update(params)
    cfg = PlusConfig(
        name="a2_test", adapter_mode="v2", birth=BirthConfig(enabled=birth),
        recovery=RecoveryConfig(enabled=recovery),
        birth_policy=policy or BirthPolicyConfig(),
        birth_identity=identity or BirthIdentityConfig(),
    )
    return BoTSORTPlus(cfg, with_reid=True, **native_params)


def original(params=None):
    native_params = dict(PARAMS)
    if params:
        native_params.update(params)
    tracker = BotSort(with_reid=False, **native_params)
    tracker.with_reid = True
    tracker.model = CacheOnlyEncoder()
    return tracker


def replay(tracker, sequence):
    outputs, events = [], []
    for dets in sequence:
        outputs.append(tracker.update(dets.copy(), IMAGE.copy(), embs=dense_features(dets))
                       .copy().reshape(-1, 8))
        events.append(copy.deepcopy(getattr(tracker, "last_events", [])))
    return outputs, events


class A2NativeTests(unittest.TestCase):
    def assert_same_outputs(self, expected, actual):
        self.assertEqual(len(expected), len(actual))
        for frame, (a, b) in enumerate(zip(expected, actual), 1):
            with self.subTest(frame=frame):
                np.testing.assert_array_equal(a, b)

    def test_low_support_publishes_only_on_fifth_observation(self):
        tracker = a2()
        outputs, frames = replay(tracker, [detections(row(score=.20)) for _ in range(7)])
        self.assertEqual([len(o) for o in outputs], [0, 0, 0, 0, 1, 1, 1])
        commits = [(i, e) for i, events in enumerate(frames, 1) for e in events
                   if e["event"] == "birth_committed"]
        self.assertEqual(len(commits), 1)
        when, commit = commits[0]
        self.assertEqual(when, 5)
        self.assertEqual(commit["candidate_first_frame"], 1)
        self.assertEqual(commit["birth_frame"], 5)
        self.assertEqual(commit["confirmation_delay_frames"], 4)
        track = tracker.active_tracks[0]
        self.assertEqual(track.start_frame, 5)
        self.assertEqual(track.frame_id, 7)
        self.assertTrue(track.is_activated)
        self.assertEqual(BaseTrack._count, 1)
        self.assertEqual([int(o[0, 4]) for o in outputs[4:]], [track.id]*3)
        first_outputs = [e for events in frames for e in events if e["event"] == "first_emitted"]
        self.assertEqual(len(first_outputs), 1)
        self.assertEqual(first_outputs[0]["frame"], 5)
        # No prior outputs are backfilled, and candidate evidence is consumed once.
        self.assertTrue(all(o.shape == (0, 8) for o in outputs[:4]))
        self.assertFalse(tracker.candidate_pool.candidates)

    def test_stable_near_floor_detections_do_not_become_existence_evidence(self):
        tracker = a2()
        outputs, frames = replay(tracker, [detections(row(score=.11)) for _ in range(25)])
        self.assertTrue(all(len(o) == 0 for o in outputs))
        self.assertEqual(BaseTrack._count, 0)
        self.assertFalse(tracker.memories)
        self.assertFalse(any(e["event"] == "birth_committed" for events in frames for e in events))

    def test_a_only_does_not_steal_or_recover_lost_identity_even_after_wait_expires(self):
        tracker = a2()
        first = tracker.update(detections(row()), IMAGE.copy(), features=[packets()])
        old_id = int(first[0, 4])
        tracker.update(detections(), IMAGE.copy(), features=[])
        logs = []
        for _ in range(11):
            output = tracker.update(detections(row(score=.2)), IMAGE.copy(), features=[{}])
            self.assertEqual(len(output), 0)
            logs.extend(copy.deepcopy(tracker.last_events))
        self.assertEqual(BaseTrack._count, 1)
        self.assertFalse(any(e["event"] in ("recovered", "birth_committed") for e in logs))
        self.assertTrue(any(e["event"] == "birth_rejected"
                            and e.get("reason") == "identity_wait_exhausted" for e in logs))
        self.assertIn(old_id, [t.id for t in tracker.lost_stracks if t.state == TrackState.Lost])
        self.assertTrue(tracker.candidate_pool.candidates)

    def test_b_recovers_on_third_weak_observation_before_A_confirmation(self):
        tracker = a2(recovery=True)
        first = tracker.update(detections(row()), IMAGE.copy(), features=[packets()])
        old_id = int(first[0, 4])
        tracker.update(detections(), IMAGE.copy(), features=[])
        weak_outputs = []
        for _ in range(3):
            weak_outputs.append(tracker.update(detections(row(score=.2)), IMAGE.copy(), features=[{}]).copy())
        self.assertEqual([len(o) for o in weak_outputs], [0, 0, 1])
        self.assertEqual(int(weak_outputs[-1][0, 4]), old_id)
        self.assertEqual(BaseTrack._count, 1)
        self.assertFalse(tracker.candidate_pool.candidates)
        self.assertTrue(any(e["event"] == "recovered" for e in tracker.last_events))
        self.assertFalse(any(e["event"] == "birth_committed" for e in tracker.last_events))

    def test_A_evidence_rejection_does_not_remove_B_candidates(self):
        tracker = a2(recovery=True, policy=BirthPolicyConfig(mean_score=.99))
        first = tracker.update(detections(row()), IMAGE.copy(), features=[packets()])
        old_id = int(first[0, 4])
        tracker.update(detections(), IMAGE.copy(), features=[])
        for _ in range(3):
            output = tracker.update(detections(row(score=.2)), IMAGE.copy(), features=[{}])
        self.assertEqual(output[:, 4].astype(int).tolist(), [old_id])
        self.assertTrue(any(e["event"] == "recovered" for e in tracker.last_events))

    def test_native_high_birth_preempts_A_verification_without_A_delay(self):
        tracker = a2()
        for _ in range(3):
            output = tracker.update(detections(row(score=.2)), IMAGE.copy(), features=[{}])
            self.assertEqual(len(output), 0)
        armed = next(iter(tracker.birth_policy.states.values()))
        self.assertEqual(armed.armed_frame, 3)
        fourth = tracker.update(detections(row()), IMAGE.copy(), features=[packets()])
        self.assertEqual(len(fourth), 0)  # Native high births after f1 are unconfirmed.
        self.assertEqual(len(tracker.active_tracks), 1)
        track = tracker.active_tracks[0]
        self.assertEqual(track.start_frame, 4)
        self.assertFalse(track.is_activated)
        self.assertFalse(tracker.candidate_pool.candidates)
        self.assertTrue(any(e["event"] == "native_birth" for e in tracker.last_events))
        self.assertFalse(any(e["event"] == "birth_committed" for e in tracker.last_events))
        fifth = tracker.update(detections(row()), IMAGE.copy(), features=[packets()])
        self.assertEqual(fifth[:, 4].astype(int).tolist(), [track.id])
        self.assertEqual(BaseTrack._count, 1)

    def test_missing_weak_features_do_not_forbid_existence_confirmation(self):
        tracker = a2()
        for _ in range(5):
            output = tracker.update(detections(row(score=.2)), IMAGE.copy(), features=[{}])
        self.assertEqual(len(output), 1)
        self.assertIsNone(tracker.active_tracks[0].smooth_feat)
        tid = int(output[0, 4])
        continued = tracker.update(detections(row()), IMAGE.copy(), features=[packets()])
        self.assertEqual(continued[:, 4].astype(int).tolist(), [tid])
        np.testing.assert_allclose(tracker.active_tracks[0].smooth_feat, E1)

    def test_native_duplicate_preflight_does_not_consume_or_write_identity(self):
        tracker = a2(identity=BirthIdentityConfig(enabled=False))
        for _ in range(2):
            tracker.update(detections(row()), IMAGE.copy(), features=[packets()])
        old_id = tracker.active_tracks[0].id
        tracker.update(detections(), IMAGE.copy(), features=[])
        original_counts = dict(tracker.memories[old_id].update_counts)
        candidate_id = None
        for i in range(6):
            output = tracker.update(detections(row(score=.2)), IMAGE.copy(), features=[packets()])
            self.assertEqual(len(output), 0)
            current = next(iter(tracker.candidate_pool.candidates.values()))
            candidate_id = current.candidate_id if candidate_id is None else candidate_id
            self.assertEqual(current.candidate_id, candidate_id)
            self.assertEqual(current.first_frame, 4)
            if i >= 4:
                self.assertTrue(any(e["event"] == "birth_rejected"
                                    and e.get("reason") == "native_duplicate_preflight"
                                    for e in tracker.last_events))
                self.assertFalse(any(e["event"] in ("birth_committed", "weak_birth", "memory_update")
                                     for e in tracker.last_events))
        self.assertEqual(set(tracker.memories), {old_id})
        self.assertEqual(tracker.memories[old_id].update_counts, original_counts)
        self.assertEqual(BaseTrack._count, 1)

    def test_shadow_matches_native_with_duplicate_high_births_and_failed_epochs(self):
        sequence = []
        for f in range(1, 10):
            score = .2 if f <= 3 else .11
            sequence.append(detections(row(x=20+f/10, w=20, h=20),
                                       row(x=20+f/10, w=20, h=20, score=.9),
                                       row(x=100, score=score)))
        sequence += [detections(), detections(row(x=21, w=20, h=20),
                                              row(x=21, w=20, h=20, score=.9))]
        expected, _ = replay(original(), sequence)
        tracker = a2(policy=BirthPolicyConfig(observe_only=True))
        actual, logs = replay(tracker, sequence)
        self.assert_same_outputs(expected, actual)
        self.assertEqual(len(actual[0]), 2)
        self.assertFalse(any(e["event"] == "suppressed_native_birth" for events in logs for e in events))
        self.assertFalse(any(e["event"] == "birth_committed" for events in logs for e in events))
        self.assertTrue(any(e["event"] == "birth_evidence" for events in logs for e in events))

    def test_shadow_ready_has_no_effect_on_native_or_public_id_counter(self):
        sequence = [detections(row(score=.2)) for _ in range(10)]
        sequence += [detections(row()), detections(row())]
        expected, _ = replay(original(), sequence)
        tracker = a2(policy=BirthPolicyConfig(observe_only=True))
        actual, logs = replay(tracker, sequence)
        self.assert_same_outputs(expected, actual)
        self.assertEqual(BaseTrack._count, 1)
        self.assertTrue(any(e["event"] == "birth_shadow_ready" for events in logs for e in events))
        self.assertFalse(any(e["event"] == "birth_committed" for events in logs for e in events))

    def test_shadow_matches_B_reference_with_recovery_and_duplicate_candidates(self):
        sequence = [detections(row()), detections()]
        sequence += [detections(row(score=.2), row(x=100, score=.2)) for _ in range(5)]
        sequence += [detections(row(), row(x=100), row(x=100, score=.9)),
                     detections(row(), row(x=100), row(x=100, score=.9)), detections()]
        expected, expected_logs = replay(a2(birth=False, recovery=True), sequence)
        tracker = a2(recovery=True, policy=BirthPolicyConfig(observe_only=True))
        actual, logs = replay(tracker, sequence)
        self.assert_same_outputs(expected, actual)
        expected_recovered = [(f, e["track_id"], e["det_index"])
                              for f, events in enumerate(expected_logs, 1)
                              for e in events if e["event"] == "recovered"]
        actual_recovered = [(f, e["track_id"], e["det_index"])
                            for f, events in enumerate(logs, 1)
                            for e in events if e["event"] == "recovered"]
        self.assertEqual(expected_recovered, actual_recovered)
        self.assertTrue(actual_recovered)

    def test_native_removed_list_enumeration_parity_is_restored_in_shadow(self):
        params = {"track_buffer": 4}  # At 15 FPS the actual retention is two frames.
        sequence = [detections(row()), detections(), detections(), detections(),
                    detections(row()), detections(row())]
        expected, _ = replay(original(params), sequence)
        tracker = a2(params=params, identity=BirthIdentityConfig(max_age=2),
                     policy=BirthPolicyConfig(observe_only=True))
        actual, _ = replay(tracker, sequence)
        self.assert_same_outputs(expected, actual)
        # This tests compatibility with native enumeration, not permission for B
        # to reactivate Removed tracks through its additional recovery branch.
        self.assertEqual(actual[4][:, 4].astype(int).tolist(), [1])

    def test_strict_native_half_score_boundary_survives_A2(self):
        tracker = a2()
        below_low = float(np.nextafter(np.float32(.1), np.float32(0.)))
        for _ in range(10):
            output = tracker.update(detections(row(score=.5), row(x=100, score=below_low)),
                                    IMAGE.copy(), features=[{}, {}])
            self.assertEqual(len(output), 0)
        self.assertFalse(tracker.candidate_pool.candidates)
        self.assertEqual(BaseTrack._count, 0)

    def test_commit_alias_and_lifetime_do_not_count_candidate_as_public_history(self):
        tracker = a2()
        _, logs = replay(tracker, [detections(row(score=.2)) for _ in range(7)])
        committed = [e for events in logs for e in events if e["event"] == "birth_committed"]
        aliases = [e for events in logs for e in events if e["event"] == "weak_birth"]
        self.assertEqual(len(committed), 1)
        self.assertEqual(len(aliases), 1)
        self.assertEqual(aliases[0]["alias_of"], "birth_committed")
        self.assertEqual(aliases[0]["track_id"], committed[0]["track_id"])
        before = copy.deepcopy(tracker.track_origins)
        summaries = tracker.finish_sequence_events()
        self.assertEqual(tracker.track_origins, before)
        self.assertEqual(len(summaries), 1)
        item = summaries[0]
        self.assertEqual(item["origin"], "weak_birth")
        self.assertEqual(item["birth_frame"], 5)
        self.assertEqual(item["candidate_first_frame"], 1)
        self.assertEqual(item["first_emitted"], 5)
        self.assertEqual(item["last_emitted"], 7)
        self.assertEqual(item["output_frames"], 3)


if __name__ == "__main__":
    unittest.main()
