"""Behavioral checks for the R4.1 hook without data/GPU/native dependencies."""
from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy
import gzip
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "tools/research/vttiny_wem_r3/observer",
             ROOT / "tools/research/vttiny_wem_r3", ROOT / "tools/research/vttiny_wem_r4"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tools.research.wem_r41 import online
from tools.research.vttiny_wem_r4 import online as old_online
from score_features import trajectory_context


def mean_for(box):
    box = np.asarray(box, dtype=float)
    return np.r_[(box[:2] + box[2:]) / 2, box[2:] - box[:2], np.zeros(4)]


class FakeKF:
    def initiate(self, measurement):
        return np.r_[measurement, np.zeros(4)], np.eye(8)

    def update(self, mean, covariance, measurement):
        mean = mean.copy()
        mean[:4] = measurement
        return mean, covariance.copy() * .5

    def predict(self, mean, covariance):
        mean = mean.copy()
        mean[:4] += mean[4:]
        return mean, covariance.copy()


class FakeTrack:
    def __init__(self, detection=None, max_obs=50, tid=17, location=(10, 10, 20, 20)):
        self.id = tid
        self.state = 2
        self.mean = mean_for(location)
        self.covariance = np.eye(8)
        self.curr_feat = np.array([.6, .8])
        self.smooth_feat = np.array([.8, .6])
        self.frame_id = 2
        self.det_ind = 0
        self.conf = .9
        if detection is not None:
            self.mean = mean_for(detection[:4])
            self.det_ind = int(detection[6])
            self.conf = float(detection[4])

    def re_activate(self, observation, frame, new_id=False):
        if new_id:
            raise AssertionError("Weak updates may not allocate identities")
        self.mean = observation.mean.copy()
        self.covariance *= .5
        self.state = 1
        self.frame_id = frame
        self.det_ind = observation.det_ind

    @staticmethod
    def multi_gmc(tracks, warp):
        for track in tracks:
            track.mean[:2] = warp[:, :2] @ track.mean[:2] + warp[:, 2]


class FakeTracker:
    def __init__(self):
        self.plus = SimpleNamespace(birth=SimpleNamespace(enabled=False), identity=SimpleNamespace(enabled=False))
        self.candidate_pool = SimpleNamespace(candidates={})
        self.last_events = []
        self.active_tracks, self.lost_stracks, self.removed_stracks = [], [], []
        self.kalman_filter = FakeKF()
        self.frame_count = 3
        self.max_obs = 50
        self.drop_reactivation = False

    def _prepare_output(self, activated, refound, lost, removed):
        refound = [] if self.drop_reactivation else refound
        self.active_tracks = list(activated) + list(refound)
        self.lost_stracks, self.removed_stracks = list(lost), list(removed)
        rows = []
        for track in self.active_tracks:
            rows.append([*online.box(track.mean), track.id, track.conf, 0, track.det_ind])
        return np.asarray(rows, dtype=float).reshape(-1, 8)


def fake_native_modules():
    state = ModuleType("boxmot.trackers.botsort.basetrack")
    state.TrackState = SimpleNamespace(Tracked=1, Lost=2)
    track = ModuleType("boxmot.trackers.botsort.botsort_track")
    track.STrack = FakeTrack
    return {state.__name__: state, track.__name__: track}


def fixture(mode="W0", rows=None):
    obj = object.__new__(online.ClosedLoop)
    obj.enabled = mode != "off"
    obj.variant = obj.mode = mode
    obj.policy = "W3" if mode in ("shadow", "force_reject") else mode
    obj.with_rgb = mode in online.RGB_MODES
    obj.frame = 3
    obj.seq, obj.method = "synthetic", "B_early_moderate"
    obj.image = "unused.jpg"
    obj.images_root = ROOT
    obj.current_image = np.zeros((32, 32, 3), dtype=np.uint8)
    obj.events, obj.score_log, obj.decision_log, obj.frame_log = (io.BytesIO() for _ in range(4))
    obj.runtime_counts = Counter()
    obj.encoder = None
    rows = rows or [[10.25, 10.25, 20.25, 20.25, .08, 0], [10.7, 10.7, 20.7, 20.7, .07, 0]]
    raw = np.asarray(rows, dtype=np.float32)
    obj.rows = [{}, {}, {"frame": 3, "image": obj.image, "detections": raw.tolist()}]
    obj.detections = [dict(det_index=i, raw_xyxy=row[:4].astype(float).tolist(), raw_score=float(row[4]),
                           band="S0", detector_class=0, rgb_valid=False) for i, row in enumerate(raw)]
    history = [dict(track_id=17, frame=frame, det_index=0, stage="high", raw_xyxy=[10., 10., 20., 20.])
               for frame in (1, 2)]
    obj.start_tracks = [dict(track_id=17, collection="active_tracks", confirmed=True, state=1,
                             last_frame=2, raw_history=history)]
    obj.post_gmc = [dict(track_id=17, selected_xyxy=[10., 10., 20., 20.], backup_xyxy=[10., 10., 20., 20.])]
    obj.anchors = {17: online.Anchor(mean_for([10, 10, 20, 20]), np.eye(8), 2, ready=True)}
    obj.history = {17: deque(history, maxlen=3)}
    obj.gmc = {"backup_guard_active": False}
    obj.warps = {i: np.array([[1., 0., 0.], [0., 1., 0.]]) for i in (1, 2, 3)}
    tracker, lost = FakeTracker(), FakeTrack()
    return obj, tracker, lost, np.column_stack((raw, np.arange(len(raw))))


def records(stream):
    return [json.loads(line) for line in stream.getvalue().splitlines()]


class MotionAndScoringTests(unittest.TestCase):
    def test_motion_context_matches_original_with_camera_motion(self):
        history = [dict(frame=frame, det_index=0, raw_xyxy=[frame, 0, frame + 8, 6]) for frame in (1, 2, 4)]
        warps = {frame: np.array([[1., 0., .3 * frame], [0., 1., -.2]]) for frame in range(2, 6)}
        cache = {"offsets": np.arange(6), "rgb_valid": np.ones(5, bool), "rgb": np.tile([1., 0.], (5, 1))}
        _, expected, count = trajectory_context(history, 5, warps, cache)
        actual, kept = online.motion_context(history, 5, warps)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(len(kept), count)

    def test_motion_requires_distinct_recent_real_frames(self):
        rows = [dict(frame=1, raw_xyxy=[0, 0, 10, 10])]
        self.assertIsNone(online.motion_context(rows, 3, {})[0])
        with self.assertRaisesRegex(ValueError, "Repeated"):
            online.motion_context(rows + rows, 3, {})

    def test_history_only_keeps_geometry_utility_and_original_ranks(self):
        obj, _, _, _ = fixture()
        shot = dict(frame=3, sequence=obj.seq, reference=obj.method, detections=obj.detections,
                    post_gmc=obj.post_gmc, start_tracks=obj.start_tracks,
                    s0_edges=[dict(track_id=17, det_index=i, selected_iou=.8 - .1 * i) for i in range(2)])
        rows = online.motion_scores(shot, obj.rows[2], obj.warps)
        rows[0]["motion"] = None
        rows[1]["geometry_group_rank"] = 6
        decisions = online.history_only_decisions(rows)
        self.assertFalse(any(row["selected"] for row in decisions))
        self.assertIn("motion_unavailable", decisions[0]["reasons"])
        self.assertIn("top_k", decisions[1]["reasons"])

    def test_common_score_path_retains_missing_appearance(self):
        obj, tracker, lost, raw = fixture("W1")
        with patch.dict(sys.modules, fake_native_modules()), patch.object(online, "frame_features", side_effect=AssertionError("RGB called")):
            output = obj.finish_before_output(tracker, [], [], [lost], [], set(), set(), raw)
        self.assertEqual(len(output), 1)
        self.assertEqual(records(obj.score_log)[0]["cosine"], None)
        self.assertEqual(obj.runtime_counts["exposure_motion_unavailable"], 0)

    def test_rgb_scorer_preserves_invalid_crop_and_same_motion_pool(self):
        obj, _, _, _ = fixture("W2")
        vectors = np.zeros((4, 1280), dtype=np.float32)
        vectors[:, 0] = 1
        obj.features = dict(offsets=np.array([0, 1, 2, 4]), rgb_valid=np.ones(4, bool), rgb=vectors)
        obj.encoder = object()
        shot = dict(frame=3, sequence=obj.seq, reference=obj.method, image=obj.image,
                    detections=obj.detections, post_gmc=obj.post_gmc, start_tracks=obj.start_tracks,
                    accepted=[], consumed=[], reserved=[],
                    s0_edges=[dict(track_id=17, det_index=i, native_low_feasible=True,
                                   gmc_guard_feasible=True, selected_iou=.8 - .1 * i) for i in range(2)])
        exports = ModuleType("botsort_plus.feature_export")
        exports._valid_rgb_crop = lambda boxes, image: np.array([True, False])
        exports._encode_batches = lambda encoder, boxes, image, modality, batch: (
            vectors[:len(boxes)], np.ones(len(boxes), bool), None,
        )
        cache_io = ModuleType("botsort_plus.io")
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "image.jpg"
            image.write_bytes(b"frozen RGB content")
            cache_io.resolve_image = lambda name, root: image
            with patch.dict(sys.modules, {exports.__name__: exports, cache_io.__name__: cache_io}):
                scored = obj._scores(shot)
        self.assertEqual(len(scored), 2)
        self.assertEqual(scored[0]["cosine"], 1.)
        self.assertIsNone(scored[1]["cosine"])
        self.assertEqual(obj.runtime_counts["rgb_crops"], 2)
        self.assertEqual(obj.runtime_counts["rgb_valid_crops"], 1)
        decisions = online.decision_bundle(scored, True)
        self.assertEqual(decisions["W2"][1]["appearance_status"], "missing_fallback")
        self.assertTrue(decisions["W2"][1]["eligible"])
        self.assertFalse(any(row["eligible"] for row in decisions["W3"]))

    def test_rgb_pool_drift_is_rejected(self):
        obj, _, _, _ = fixture("W2")
        obj.encoder, obj.features = object(), {}
        shot = dict(frame=3, sequence=obj.seq, reference=obj.method, image=obj.image,
                    detections=obj.detections, post_gmc=obj.post_gmc, start_tracks=obj.start_tracks,
                    s0_edges=[dict(track_id=17, det_index=0, selected_iou=.8)])
        cache_io = ModuleType("botsort_plus.io")
        cache_io.resolve_image = lambda name, root: Path(name)
        with patch.dict(sys.modules, {cache_io.__name__: cache_io}), patch.object(online, "frame_features", return_value=([], [], None)):
            with self.assertRaisesRegex(ValueError, "shared weak candidate pool"):
                obj._scores(shot)


class OnlineLifecycleTests(unittest.TestCase):
    def run_frame(self, obj, tracker, lost, raw, consumed=(), reserved=()):
        with patch.dict(sys.modules, fake_native_modules()):
            return obj.finish_before_output(tracker, [], [], [lost], [], set(reserved), set(consumed), raw)

    def test_real_hook_commits_measurement_retains_id_and_appearance(self):
        obj, tracker, old, raw = fixture("W1")
        before = deepcopy(old)
        output = self.run_frame(obj, tracker, old, raw)
        self.assertEqual(output[0, 4], old.id)
        self.assertFalse(np.array_equal(tracker.active_tracks[0].mean, before.mean))
        np.testing.assert_array_equal(old.mean, before.mean)
        np.testing.assert_array_equal(tracker.active_tracks[0].smooth_feat, before.smooth_feat)
        self.assertEqual([item["frame"] for item in obj.history[17]], [1, 2])
        self.assertTrue(obj.anchors[17].spent)
        commit, = [row for row in records(obj.events) if row["event"] == "committed"]
        self.assertFalse(commit["prediction"])
        self.assertEqual(commit["decision_policy"], "W1")

    def test_W0_is_equivalent_to_archived_plain_native_transaction(self):
        new, tracker, lost, raw = fixture("W0")
        old = object.__new__(old_online.ClosedLoop)
        old.__dict__ = deepcopy(new.__dict__)
        old.variant = old.mode = "plain"
        other_tracker, other_lost = deepcopy(tracker), deepcopy(lost)
        actual = self.run_frame(new, tracker, lost, raw)
        with patch.dict(sys.modules, fake_native_modules()):
            expected = old.finish_before_output(other_tracker, [], [], [other_lost], [], set(), set(), raw)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(tracker.active_tracks[0].mean, other_tracker.active_tracks[0].mean)
        self.assertEqual(new.anchors[17].spent, old.anchors[17].spent)

    def test_anchor_and_ownership_rules_prevent_weak_commit(self):
        for case in ("not_ready", "spent", "stale", "diverged", "consumed", "reserved", "near_owner", "backup"):
            with self.subTest(case=case):
                obj, tracker, lost, raw = fixture("W0", [[10.25, 10.25, 20.25, 20.25, .08, 0]])
                consumed, reserved = set(), set()
                if case == "not_ready": obj.anchors[17].ready = False
                if case == "spent": obj.anchors[17].spent = True
                if case == "stale": obj.anchors[17].observed = 1
                if case == "diverged": obj.anchors[17].mean[:2] += 100
                if case == "consumed": consumed.add(0)
                if case == "reserved":
                    reserved.add(77)
                    tracker.candidate_pool.candidates[77] = SimpleNamespace(latest=SimpleNamespace(det_index=0, frame=3))
                if case == "near_owner":
                    other = dict(obj.detections[0], det_index=1, band="high", raw_score=.9)
                    obj.detections.append(other)
                    raw = np.vstack((raw, [*other["raw_xyxy"], .9, 0, 1]))
                    obj.rows[2]["detections"].append(raw[-1, :6].tolist())
                    consumed.add(1)
                if case == "backup":
                    obj.gmc["backup_guard_active"] = True
                    obj.post_gmc[0]["backup_xyxy"] = [100., 100., 110., 110.]
                output = self.run_frame(obj, tracker, lost, raw, consumed, reserved)
                self.assertEqual(len(output), 0)
                self.assertEqual(obj.runtime_counts["committed"], 0)

    def test_native_finalization_rejection_does_not_spend_anchor(self):
        obj, tracker, lost, raw = fixture("W0")
        tracker.drop_reactivation = True
        output = self.run_frame(obj, tracker, lost, raw)
        self.assertEqual(len(output), 0)
        self.assertEqual(obj.runtime_counts["rejected_finalization"], 1)
        self.assertFalse(obj.anchors[17].spent)

    def test_off_does_not_touch_tracking_or_require_candidate_state(self):
        obj, tracker, lost, raw = fixture("off")
        del obj.start_tracks
        output = self.run_frame(obj, tracker, lost, raw)
        self.assertEqual(len(output), 0)
        self.assertEqual(obj.runtime_counts, {})
        self.assertEqual(obj.events.getvalue(), b"")

    def test_readonly_controls_are_identical_and_never_commit(self):
        streams = []
        for mode in ("shadow", "force_reject"):
            obj, tracker, lost, raw = fixture(mode)
            obj.with_rgb = True
            def score(shot, obj=obj):
                rows = online.motion_scores(shot, obj.rows[2], obj.warps)
                for i, row in enumerate(rows):
                    row.update(cosine=.9 - .2 * i, rgb_evaluated=True)
                return rows
            obj._scores = score
            before = deepcopy(lost)
            output = self.run_frame(obj, tracker, lost, raw)
            self.assertEqual(len(output), 0)
            self.assertGreater(obj.runtime_counts["proposed"], 0)
            self.assertEqual(obj.runtime_counts["committed"], 0)
            self.assertFalse(obj.anchors[17].spent)
            np.testing.assert_array_equal(lost.mean, before.mean)
            streams.append(tuple(stream.getvalue() for stream in (obj.events, obj.score_log, obj.decision_log, obj.frame_log)))
        self.assertEqual(streams[0], streams[1])

    def test_spent_episode_needs_two_new_regular_observations(self):
        obj, tracker, lost, raw = fixture("W0")
        self.run_frame(obj, tracker, lost, raw)
        for frame, ready, expected_history in ((4, False, [4]), (5, True, [4, 5])):
            obj.frame = frame
            record = dict(frame=frame, track_id=17, det_index=0, stage="high", raw_xyxy=[10., 10., 20., 20.])
            obj._update_anchors(tracker, [record])
            self.assertEqual(obj.anchors[17].ready, ready)
            self.assertFalse(obj.anchors[17].spent)
            self.assertEqual([row["frame"] for row in obj.history[17]], expected_history)

    def test_broken_anchor_history_is_reported_before_abort(self):
        obj, tracker, lost, raw = fixture("W0")
        obj.start_tracks[0]["raw_history"] = obj.start_tracks[0]["raw_history"][-1:]
        with self.assertRaisesRegex(ValueError, "two-frame motion history"):
            self.run_frame(obj, tracker, lost, raw)
        self.assertGreater(obj.runtime_counts["exposure_motion_unavailable"], 0)
        self.assertEqual(obj.runtime_counts["committed"], 0)

    def test_gzip_logs_have_stable_content_independent_of_path(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = []
            for name in ("first", "second"):
                obj, _, _, _ = fixture()
                obj._raw_audit_streams = []
                stream = obj._audit_stream(Path(folder) / name, "wem_scores")
                stream.write(b'{"frame":1}\n')
                stream.close()
                for raw in obj._raw_audit_streams: raw.close()
                paths.append(Path(folder) / name / "wem_scores" / obj.method / (obj.seq + ".jsonl.gz"))
            self.assertEqual(paths[0].read_bytes(), paths[1].read_bytes())
            self.assertEqual(gzip.decompress(paths[0].read_bytes()), b'{"frame":1}\n')


if __name__ == "__main__":
    unittest.main()
