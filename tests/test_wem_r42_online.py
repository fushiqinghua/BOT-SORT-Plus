"""Meaningful state-feedback and anchor-ablation checks without dataset/GPU."""
from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy
import hashlib
import io
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / "tests", ROOT / "tools/research/vttiny_wem_r3/observer",
               ROOT / "tools/research/vttiny_wem_r3", ROOT / "tools/research/vttiny_wem_r4"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from test_wem_r41_online import FakeTrack, fake_native_modules, fixture as r41_fixture, records
from tools.research.wem_r42 import online
from tools.research.wem_r41.online import ClosedLoop as R41ClosedLoop
from capture import encode, track_state


def modules():
    result = fake_native_modules()
    adapter = ModuleType("botsort_plus.adapter_v2")
    adapter.STrack = FakeTrack
    result[adapter.__name__] = adapter
    return result


def add_track_fields(track):
    track.is_activated = True
    track.start_frame = 1
    track.tracklet_len = 2
    return track


def fixture(mode="W0", rows=None):
    original, tracker, lost, raw = r41_fixture("off" if mode == "off" else "W0", rows)
    obj = object.__new__(online.ClosedLoop)
    obj.__dict__ = original.__dict__
    obj.r42_mode = obj.variant = obj.mode = mode
    obj.policy = "W0" if mode != "off" else "off"
    obj.extra_log = io.BytesIO() if mode in (online.RELAXED, online.EMIT_ONLY) else None
    obj.states = io.BytesIO()
    obj.counts = Counter()
    obj.previous = 2
    obj._frame_weak_outputs = 0
    obj.runtime_counts.update(weak_outputs_emitted=0, weak_state_updates=0)
    add_track_fields(lost)
    FakeTrack._count = 23
    tracker.active_tracks = [lost]
    tracker.candidate_pool._next_id = 31
    tracker.candidate_pool._frame = 3
    tracker.memories = {}
    tracker.prediction_history = {}
    tracker.recovery_matcher = SimpleNamespace(_first_reserved={}, _last_frame=3)
    tracker.track_origins = {17: {"origin": "native", "first_emitted": 1, "last_emitted": 2}}
    return obj, tracker, lost, raw


def run_frame(obj, tracker, lost, raw, consumed=(), reserved=(), activated=()):
    with patch.dict(sys.modules, modules()):
        native = obj.finish_before_output(tracker, list(activated), [], [lost], [], set(reserved), set(consumed), raw)
        output = obj.publish(native)
    return output


def state(tracker, output, gmc):
    with patch.dict(sys.modules, modules()):
        return encode(track_state(tracker, output, gmc))


class DelegationTests(unittest.TestCase):
    def test_off_and_W0_delegate_without_changing_events_scores_or_state(self):
        for mode in ("off", "W0"):
            with self.subTest(mode=mode):
                current, tracker, lost, raw = fixture(mode)
                previous = object.__new__(R41ClosedLoop)
                previous.__dict__ = deepcopy(current.__dict__)
                old_tracker, old_lost = deepcopy(tracker), deepcopy(lost)
                output = run_frame(current, tracker, lost, raw)
                with patch.dict(sys.modules, modules()):
                    expected = previous.finish_before_output(old_tracker, [], [], [old_lost], [], set(), set(), raw)
                np.testing.assert_array_equal(output, expected)
                for name in ("events", "score_log", "decision_log", "frame_log"):
                    self.assertEqual(getattr(current, name).getvalue(), getattr(previous, name).getvalue())
                self.assertEqual(state(tracker, output, current.gmc), state(old_tracker, expected, previous.gmc))
                self.assertIs(current.publish(output), output)


class AnchorAblationTests(unittest.TestCase):
    def test_one_regular_observation_is_opened_without_requiring_motion(self):
        obj, tracker, lost, raw = fixture(online.RELAXED)
        obj.anchors[17].ready = False
        obj.start_tracks[0]["raw_history"] = obj.start_tracks[0]["raw_history"][-1:]
        obj.history[17] = deque(obj.start_tracks[0]["raw_history"], maxlen=3)
        output = run_frame(obj, tracker, lost, raw)
        self.assertEqual(len(output), 1)
        self.assertEqual(obj.runtime_counts["exposure_motion_unavailable"], 2)
        self.assertEqual(obj.runtime_counts["exposure_anchor_not_ready_bypassed_tracks"], 1)
        self.assertEqual(obj.runtime_counts["anchor_support_bypassed_committed"], 1)
        self.assertEqual(obj.runtime_counts["weak_state_updates"], 1)
        self.assertEqual(obj.runtime_counts["exposure_anchor_guard_added_selected"], 1)
        committed, = [row for row in records(obj.events) if row["event"] == "committed"]
        self.assertTrue(committed["state_committed"])
        self.assertTrue(committed["anchor_not_ready"])
        self.assertIsNone(committed["motion"])
        self.assertEqual([row["frame"] for row in obj.history[17]], [2])

    def test_independent_prediction_and_box_gates_are_opened(self):
        for case in ("prediction", "box_only"):
            with self.subTest(case=case):
                obj, tracker, lost, raw = fixture(online.RELAXED, [[12., 10., 22., 20., .08, 0]])
                # x shift -2 gives anchor/native IoU 2/3 but anchor/weak 3/7.
                obj.anchors[17].mean[0] -= 100 if case == "prediction" else 2
                output = run_frame(obj, tracker, lost, raw)
                self.assertEqual(len(output), 1)
                self.assertEqual(obj.runtime_counts["exposure_anchor_box_bypassed_edges"], 1)
                self.assertEqual(obj.runtime_counts["exposure_anchor_prediction_bypassed_tracks"], int(case == "prediction"))
                self.assertEqual(obj.runtime_counts["anchor_support_bypassed_committed"], 1)

    def test_opened_cases_are_rejected_by_the_unchanged_strict_W0(self):
        for case in ("readiness", "prediction", "box_only"):
            with self.subTest(case=case):
                obj, tracker, lost, raw = fixture("W0", [[12., 10., 22., 20., .08, 0]])
                if case == "readiness": obj.anchors[17].ready = False
                else: obj.anchors[17].mean[0] -= 100 if case == "prediction" else 2
                self.assertEqual(len(run_frame(obj, tracker, lost, raw)), 0)

    def test_when_guard_already_passes_relaxed_matches_W0(self):
        a, tracker_a, lost_a, raw = fixture("W0")
        b, tracker_b, lost_b, _ = fixture(online.RELAXED)
        result_a = run_frame(a, tracker_a, lost_a, raw)
        result_b = run_frame(b, tracker_b, lost_b, raw)
        np.testing.assert_array_equal(result_a, result_b)
        self.assertEqual(state(tracker_a, result_a, a.gmc), state(tracker_b, result_b, b.gmc))
        self.assertEqual(b.runtime_counts["exposure_anchor_guard_changed_track_frames"], 0)
        self.assertEqual(b.runtime_counts["anchor_support_bypassed_committed"], 0)

    def test_freshness_budget_and_native_ownership_remain_active(self):
        for case in ("missing_anchor", "spent", "stale", "unconfirmed", "not_active", "consumed", "reserved", "near_owner", "backup", "native_iou"):
            with self.subTest(case=case):
                obj, tracker, lost, raw = fixture(online.RELAXED, [[10.25, 10.25, 20.25, 20.25, .08, 0]])
                obj.anchors[17].ready = False
                consumed, reserved = set(), set()
                if case == "missing_anchor": obj.anchors.clear()
                if case == "spent": obj.anchors[17].spent = True
                if case == "stale": obj.anchors[17].observed = 1
                if case == "unconfirmed": obj.start_tracks[0]["confirmed"] = False
                if case == "not_active": obj.start_tracks[0]["collection"] = "lost_stracks"
                if case == "consumed": consumed.add(0)
                if case == "reserved":
                    reserved.add(77)
                    tracker.candidate_pool.candidates[77] = SimpleNamespace(latest=SimpleNamespace(det_index=0, frame=3))
                if case == "near_owner":
                    obj.detections.append(dict(obj.detections[0], det_index=1, band="high", raw_score=.9))
                    raw = np.vstack((raw, [*obj.detections[0]["raw_xyxy"], .9, 0, 1]))
                    obj.rows[2]["detections"].append(raw[-1, :6].tolist())
                    consumed.add(1)
                if case == "backup":
                    obj.gmc["backup_guard_active"] = True
                    obj.post_gmc[0]["backup_xyxy"] = [100., 100., 110., 110.]
                if case == "native_iou": obj.post_gmc[0]["selected_xyxy"] = [100., 100., 110., 110.]
                result = run_frame(obj, tracker, lost, raw, consumed, reserved)
                self.assertEqual(len(result), 0)
                self.assertEqual(obj.runtime_counts["weak_state_updates"], 0)

    def test_relaxed_assignment_is_still_one_to_one(self):
        obj, tracker, lost, raw = fixture(online.RELAXED, [[10.25, 10.25, 20.25, 20.25, .08, 0]])
        other = add_track_fields(FakeTrack(tid=18))
        obj.start_tracks.append(dict(deepcopy(obj.start_tracks[0]), track_id=18))
        obj.post_gmc.append(dict(obj.post_gmc[0], track_id=18))
        obj.anchors[18] = deepcopy(obj.anchors[17])
        obj.history[18] = deepcopy(obj.history[17])
        for anchor in obj.anchors.values(): anchor.ready = False
        tracker.active_tracks.append(other)
        with patch.dict(sys.modules, modules()):
            result = obj.finish_before_output(tracker, [], [], [lost, other], [], set(), set(), raw)
        self.assertEqual(len(result), 1)
        self.assertEqual(obj.runtime_counts["weak_outputs_emitted"], 1)
        self.assertEqual(len(tracker.lost_stracks), 1)

    def test_finalizer_rejection_does_not_spend_or_update(self):
        obj, tracker, lost, raw = fixture(online.RELAXED)
        obj.anchors[17].ready = False
        tracker.drop_reactivation = True
        result = run_frame(obj, tracker, lost, raw)
        self.assertEqual(len(result), 0)
        self.assertEqual(obj.runtime_counts["rejected_finalization"], 1)
        self.assertEqual(obj.runtime_counts["anchor_support_bypassed_committed"], 0)
        self.assertFalse(obj.anchors[17].spent)


class StateFeedbackTests(unittest.TestCase):
    def test_emit_only_matches_W0_current_measurement_and_off_canonical_state(self):
        w, tw, lw, raw = fixture("W0")
        e, te, le, _ = fixture(online.EMIT_ONLY)
        o, to, lo, _ = fixture("off")
        original = deepcopy(le)
        wout = run_frame(w, tw, lw, raw)
        with patch.dict(sys.modules, modules()):
            native = e.finish_before_output(te, [], [], [le], [], set(), set(), raw)
            # Native adapter bookkeeping observes this empty native output.
            self.assertEqual(len(native), 0)
            self.assertEqual(te.active_tracks, [])
            self.assertEqual(te.lost_stracks, [le])
            emitted = e.publish(native)
        off = run_frame(o, to, lo, raw)
        np.testing.assert_array_equal(emitted, wout)
        np.testing.assert_array_equal(te.lost_stracks[0].mean, original.mean)
        np.testing.assert_array_equal(te.lost_stracks[0].covariance, original.covariance)
        self.assertEqual(state(te, native, e.gmc), state(to, off, o.gmc))
        self.assertNotEqual(state(tw, wout, w.gmc), state(te, native, e.gmc))
        self.assertEqual(FakeTrack._count, 23)
        self.assertEqual(e.runtime_counts["weak_outputs_emitted"], 1)
        self.assertEqual(e.runtime_counts["weak_state_updates"], 0)
        row, = [r for r in records(e.events) if r["event"] == "committed"]
        self.assertFalse(row["state_committed"])
        self.assertTrue(row["output_only"])
        self.assertFalse(row["prediction"])

    def test_canonical_state_chain_uses_native_output_and_records_emitted_digest(self):
        e, te, le, raw = fixture(online.EMIT_ONLY)
        o, to, lo, _ = fixture("off")
        emitted = run_frame(e, te, le, raw)
        native = run_frame(o, to, lo, raw)
        with patch.dict(sys.modules, modules()):
            e.finish(te, emitted)
            o.finish(to, native)
        self.assertEqual(e.states.getvalue(), o.states.getvalue())
        row, = records(e.extra_log)
        self.assertEqual(row["state_chain_output_basis"], "canonical_native_output")
        self.assertNotEqual(row["canonical_state_sha256"], row["emitted_state_sha256"])
        self.assertEqual(row["canonical_output_rows"], 0)
        self.assertEqual(row["emitted_output_rows"], 1)
        self.assertEqual(row["weak_state_updates"], 0)
        expected = hashlib.sha256(state(te, e._pending_native, e.gmc)).hexdigest()
        self.assertEqual(row["canonical_state_sha256"], expected)

    def test_native_future_dynamics_remain_off_identical(self):
        e, te, le, raw = fixture(online.EMIT_ONLY)
        o, to, lo, _ = fixture("off")
        run_frame(e, te, le, raw)
        run_frame(o, to, lo, raw)
        # The next native prediction/update consumes official Lost state, not
        # the privately updated track used only for the emitted measurement.
        for tracker in (te, to):
            track = tracker.lost_stracks[0]
            track.mean, track.covariance = tracker.kalman_filter.predict(track.mean, track.covariance)
            observation = FakeTrack(np.array([11., 11., 21., 21., .9, 0, 0]))
            track.re_activate(observation, 4, new_id=False)
            tracker.frame_count = 4
            tracker._prepare_output([], [track], [], [])
        self.assertEqual(state(te, np.empty((0, 8)), e.gmc), state(to, np.empty((0, 8)), o.gmc))

    def test_empty_proposal_does_not_mutate_shared_removed_list(self):
        obj, tracker, lost, raw = fixture(online.EMIT_ONLY)
        obj.anchors[17].spent = True
        removed = add_track_fields(FakeTrack(tid=99))
        calls = []
        # Patch the class method: private preview and actual finalization each
        # see one removed row, not a shared append that duplicates the row.
        with patch.object(type(tracker), "_prepare_output", autospec=True) as method:
            def bound(self, activated, refound, lost_list, removed_list):
                self.removed_stracks.extend(removed_list)
                calls.append(len(self.removed_stracks))
                self.active_tracks = list(activated) + list(refound)
                self.lost_stracks = list(lost_list)
                return np.empty((0, 8))
            method.side_effect = bound
            with patch.dict(sys.modules, modules()):
                native = obj.finish_before_output(tracker, [], [], [lost], [removed], set(), set(), raw)
                obj.publish(native)
        self.assertEqual(calls, [1, 1])
        self.assertEqual([track.id for track in tracker.removed_stracks], [99])

    def test_shared_canonical_mutation_is_detected(self):
        obj, tracker, lost, raw = fixture(online.EMIT_ONLY)
        original = R41ClosedLoop.finish_before_output
        def corrupted(observer, proxy, *args):
            proxy.memories[55] = SimpleNamespace(update_counts={}, qualities={}, prototypes={})
            return original(observer, proxy, *args)
        with patch.object(R41ClosedLoop, "finish_before_output", corrupted), patch.dict(sys.modules, modules()):
            with self.assertRaisesRegex(ValueError, "changed canonical"):
                obj.finish_before_output(tracker, [], [], [lost], [], set(), set(), raw)

    def test_publish_requires_native_boundary_and_exactly_once(self):
        obj, tracker, lost, raw = fixture(online.EMIT_ONLY)
        with self.assertRaisesRegex(ValueError, "did not follow"):
            obj.publish(np.empty((0, 8)))
        with patch.dict(sys.modules, modules()):
            native = obj.finish_before_output(tracker, [], [], [lost], [], set(), set(), raw)
            with self.assertRaisesRegex(ValueError, "Post-finalizer"):
                obj.publish(np.zeros((1, 8)))
            with self.assertRaisesRegex(ValueError, "published before finish"):
                obj.finish(tracker, native)
            obj.publish(native)
            with self.assertRaisesRegex(ValueError, "twice"):
                obj.publish(native)


if __name__ == "__main__":
    unittest.main()
