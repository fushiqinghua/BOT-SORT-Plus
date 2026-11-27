"""Probe isolation and causal contracts; fixtures are not benchmark evidence."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from experiments.vttiny_gmc.round3_joint.probe import (
    JointProbe, common_identity_support, json_value, members, pool_snapshot,
    prediction_disagreement,
)
from botsort_plus.appearance import AppearanceConfig, MemoryBank
from botsort_plus.candidates import BirthConfig, CandidatePool
from botsort_plus.config import PlusConfig
from botsort_plus.geometry import GeometryConfig
from botsort_plus.recovery import RecoveryConfig, RecoveryMatcher
from botsort_plus.types import Feature, Observation, Prediction


def obs(frame, x=10., index=0, feature=True):
    packet = {"rgb": Feature(np.array([1., 0.]))} if feature else {}
    return Observation(frame, index, np.array([x, 10., x+5., 15.]), .3, packet)


class _Track:
    """Deterministic test dynamics, used only for observer ownership tests."""
    def __init__(self):
        self.id = 1
        self.mean = np.array([12.5, 12.5, 5., 5., 0., 0., 0., 0.])
        self.covariance = np.eye(8)
        self.state = 1
        self.end_frame = 1

    @property
    def xyxy(self):
        return np.r_[self.mean[:2] - self.mean[2:4]/2, self.mean[:2] + self.mean[2:4]/2]

    @staticmethod
    def multi_predict(tracks):
        for t in tracks:
            t.mean[:4] += t.mean[4:]
            t.covariance += np.eye(8)

    @staticmethod
    def multi_gmc(tracks, warp):
        r = np.kron(np.eye(4), warp[:, :2])
        for t in tracks:
            t.mean = r @ t.mean
            t.mean[:2] += warp[:, 2]
            t.covariance = r @ t.covariance @ r.T


class _KF:
    def project(self, mean, covariance):
        return mean[:4].copy(), covariance[:4, :4].copy()


class ProbeContractTests(unittest.TestCase):
    def test_missing_prediction_is_unknown_not_zero_support(self):
        result = common_identity_support([obs(2)], {}, None, GeometryConfig(),
                                         AppearanceConfig(), RecoveryConfig())
        self.assertFalse(result["available"])
        self.assertIsNone(result["score"])
        self.assertEqual(result["reason"], "missing_same_frame_prediction")

    def test_common_identity_score_does_not_depend_on_track_age_or_mutate_memory(self):
        memory = MemoryBank()
        memory.update({"rgb": Feature(np.array([1., 0.]))})
        before = deepcopy(memory.__dict__)
        observation = obs(2)
        prediction = {2: Prediction(2, observation.box.copy(), np.eye(2))}
        result = common_identity_support([observation], prediction, memory, GeometryConfig(),
                                         AppearanceConfig(), RecoveryConfig())
        self.assertTrue(result["available"])
        self.assertEqual(result["appearance_hits"], 1)
        self.assertTrue(result["time_weight_excluded"])
        np.testing.assert_array_equal(memory.prototypes["rgb"], before["prototypes"]["rgb"])
        self.assertEqual(memory.update_counts, before["update_counts"])

    def test_no_appearance_is_explicit(self):
        observation = obs(1, feature=False)
        result = common_identity_support([observation],
                                         {1: Prediction(1, observation.box.copy(), np.eye(2))},
                                         None, GeometryConfig(), AppearanceConfig(), RecoveryConfig())
        self.assertTrue(result["available"])
        self.assertIsNone(result["appearance"])
        self.assertEqual(result["appearance_hits"], 0)
        self.assertEqual(result["score"], result["geometry"])

    def test_strict_json_preserves_missing_cost(self):
        converted = json_value({"x": np.array([np.inf, np.nan, 0.])})
        self.assertEqual(converted, {"x": [None, None, 0.]})
        json.dumps(converted, allow_nan=False)

    def test_candidate_clone_transport_never_moves_historical_observations(self):
        pool = CandidatePool(BirthConfig(), GeometryConfig())
        pool.update(1, [obs(1)])
        before = deepcopy(pool_snapshot(pool))
        private = deepcopy(pool)
        private.update(2, [obs(2, x=12.)], warp=np.array([[1., 0., 2.], [0., 1., 0.]]))
        np.testing.assert_array_equal(pool.candidates[1].observations[0].box, obs(1).box)
        np.testing.assert_array_equal(private.candidates[1].observations[0].box, obs(1).box)
        self.assertEqual(members(pool.candidates[1]), [(1, 0)])
        self.assertEqual(members(private.candidates[1]), [(1, 0), (2, 0)])
        self.assertEqual(json_value(pool_snapshot(pool)), json_value(before))

    def test_prediction_is_forward_accumulated_and_reference_untouched(self):
        plus = PlusConfig(adapter_mode="v2", recovery=RecoveryConfig(enabled=True))
        track = _Track()
        tracker = SimpleNamespace(plus=plus, frame_count=1, _packets=[], memories={}, prediction_history={},
                                  candidate_pool=CandidatePool(plus.birth, plus.geometry),
                                  recovery_matcher=RecoveryMatcher(plus.recovery, plus.geometry),
                                  kalman_filter=_KF(), active_tracks=[], lost_stracks=[], last_events=[])
        identity = np.eye(2, 3)
        chosen = np.array([[1., 0., 3.], [0., 1., 0.]])
        with tempfile.TemporaryDirectory() as directory:
            probe = JointProbe(Path(directory) / "probe.jsonl", native_probe=False, candidate_probe=False)
            probe.point("begin", tracker, pool=[], unconfirmed=[], high_detections=[],
                        low_rows=np.empty((0, 7)), all_dets=np.empty((0, 7)))
            probe.point("compensated", tracker, pool=[], unconfirmed=[], warp=identity, native_warp=identity)
            tracker.active_tracks = [track]
            probe.point("end", tracker, output=np.empty((0, 8)))
            for frame in (2, 3):
                tracker.frame_count = frame
                track.state = 2
                tracker.active_tracks = []
                tracker.lost_stracks = [track]
                probe.point("begin", tracker, pool=[track], unconfirmed=[], high_detections=[],
                            low_rows=np.empty((0, 7)), all_dets=np.empty((0, 7)))
                _Track.multi_predict([track]); _Track.multi_gmc([track], chosen)
                actual_mean, actual_cov = track.mean.copy(), track.covariance.copy()
                probe.point("compensated", tracker, pool=[track], unconfirmed=[],
                            warp=chosen, native_warp=identity)
                np.testing.assert_array_equal(track.mean, actual_mean)
                np.testing.assert_array_equal(track.covariance, actual_cov)
                difference = prediction_disagreement(track.xyxy, probe.p_states["native"][1].xyxy, 4.)
                self.assertEqual(difference["center_pixels"], (frame-1)*3.)
                self.assertEqual(probe.anchors[1], 1)
                probe.point("end", tracker, output=np.empty((0, 8)))
            self.assertEqual(probe.close()["status"], "PASS")

    def test_probe_refuses_overwrite_and_nonconsecutive_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"probe.jsonl"
            probe = JointProbe(path)
            with self.assertRaises(FileExistsError):
                JointProbe(path)
            with self.assertRaises(ValueError):
                probe.point("begin", SimpleNamespace(frame_count=2), pool=[], unconfirmed=[],
                            high_detections=[], low_rows=[], all_dets=[])
            probe.close()


NATIVE_AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("boxmot", "torch", "cv2"))


@unittest.skipUnless(NATIVE_AVAILABLE, "requires optional pinned native runtime")
class NativeProbeParityTests(unittest.TestCase):
    def test_bound_evidence_gate_is_not_replayed_and_dual_low_n_is_explicitly_unavailable(self):
        import torch
        from botsort_plus.native_adapter import BoTSORTPlus
        from experiments.vttiny_gmc.b_gmc_recovery.forensics.lost_camera_margin import LostCameraMargin
        from experiments.vttiny_gmc.native_botsort_gmc_0030_fp_v1.guard import capture_priors, iou
        params = dict(reid_weights=Path("unused.pt"), device=torch.device("cpu"), half=False,
                      det_thresh=.5, track_high_thresh=.5, track_low_thresh=.1, new_track_thresh=.6,
                      match_thresh=.8, track_buffer=60, frame_rate=15, cmc_method="sof",
                      proximity_thresh=.5, appearance_thresh=.4, fuse_first_associate=False)
        plus = PlusConfig(name="B", adapter_mode="v2", recovery=RecoveryConfig(enabled=True))
        image = np.zeros((96, 128, 3), np.uint8)
        frames = [[[20.,20.,25.,25.,.9,0.]]] + [[] for _ in range(6)]
        frames += [[[20.,20.,25.,25.,.3,0.]] for _ in range(4)]
        def replay(evidence, path=None):
            class T(BoTSORTPlus):
                def _cost(self, tracks, detections, unconfirmed=False):
                    result = super()._cost(tracks, detections, unconfirmed)
                    if self.margin is not None and not unconfirmed:
                        self.margin.high_cost(result, tracks)
                    return result
                def guard_low_cost(self, cost, tracks, detections):
                    if self.matched_camera:
                        for i,t in enumerate(tracks):
                            for j,d in enumerate(detections):
                                if cost[i,j] <= .5 and iou(self.backup_boxes[int(t.id)], d[:4]) < .5:
                                    cost[i,j] = 1.
            tracker = T(deepcopy(plus), with_reid=True, **params)
            tracker.joint_dual_low = True
            tracker.joint_evidence_gate = evidence
            tracker.margin = LostCameraMargin(tracker, require_observation=True) if evidence else None
            tracker.capture_trace = False
            tracker.gate_calls = 0
            if evidence:
                gated = tracker.margin.recovery_pair(tracker.recovery_matcher._pair)
                def record_call(*args, **kwargs):
                    tracker.gate_calls += 1
                    return gated(*args, **kwargs)
                tracker.recovery_matcher._pair = record_call
            def warp(img, dets):
                tracker.joint_native_warp = np.eye(2,3)
                selected = np.eye(2,3)
                selected[0,2] = .25 if tracker.frame_count == 2 else 0.
                tracker.matched_camera = tracker.frame_count == 2
                if tracker.matched_camera:
                    capture_priors(tracker, selected, tracker.joint_native_warp)
                if tracker.margin is not None:
                    tracker.margin.observe(tracker.frame_count, tracker.joint_native_warp, selected)
                return selected
            tracker.cmc.apply = warp
            if path:
                tracker.joint_probe = JointProbe(path)
            outputs, events = [], []
            for rows in frames:
                dets = np.asarray(rows,np.float32).reshape(-1,6)
                packets = [{"rgb":Feature(np.array([1.,0.],np.float32))} if d[4]>.5 else {} for d in dets]
                outputs.append(tracker.update(dets,image.copy(),features=packets).copy())
                events.append(deepcopy(tracker.last_events))
            counts = [tracker.gate_calls]
            if tracker.margin is not None:
                counts += [tracker.margin.high_rejections, tracker.margin.recovery_rejections,
                           tracker.margin.unsupported_rejections]
            summary = tracker.joint_probe.close() if path else None
            return outputs, events, counts, summary
        with tempfile.TemporaryDirectory() as directory:
            for evidence in (False, True):
                reference = replay(evidence)
                observed = replay(evidence, Path(directory)/f"probe_{evidence}.jsonl.gz")
                self.assertEqual(json_value(reference[:3]), json_value(observed[:3]))
                self.assertEqual(observed[3]["status"], "PASS")
                self.assertEqual(observed[3]["counts"]["native_counterfactual_unavailable"], len(frames))
                if evidence:
                    self.assertGreater(reference[2][0], 0)
                    self.assertGreater(reference[2][-1], 0)
                    self.assertEqual(observed[3]["counts"]["recovery_counterfactual_unavailable"], len(frames))

    def test_real_native_all_probes_preserve_outputs_events_ids_and_memory(self):
        import torch
        from boxmot.trackers.botsort.basetrack import BaseTrack
        from botsort_plus.native_adapter import BoTSORTPlus
        params = dict(reid_weights=Path("unused.pt"), device=torch.device("cpu"), half=False,
                      det_thresh=.5, track_high_thresh=.5, track_low_thresh=.1, new_track_thresh=.6,
                      match_thresh=.8, track_buffer=60, frame_rate=15, cmc_method="sof",
                      proximity_thresh=.5, appearance_thresh=.4, fuse_first_associate=False)
        plus = PlusConfig(name="B", adapter_mode="v2", recovery=RecoveryConfig(enabled=True))
        def row(x, score):
            return [x, 20., x+5., 25., score, 0.]
        frames = [[row(20,.9)], [], [row(20,.3)], [row(20,.3)], [row(20,.3)],
                  [row(20,.9), row(90,.9)], [row(20,.3), row(90,.9)], [],
                  [row(20,.9), row(90,.9)], [row(20,.9), row(90,.9)]]
        image = np.zeros((128, 160, 3), dtype=np.uint8)
        def replay(path=None):
            tracker = BoTSORTPlus(deepcopy(plus), with_reid=True, **params)
            tracker.joint_dual_low = False
            tracker.joint_evidence_gate = False
            def warp(img, dets):
                tracker.joint_native_warp = np.eye(2, 3)
                result = np.eye(2, 3)
                result[0, 2] = .25 if tracker.frame_count in (2, 3, 7) else 0.
                return result
            tracker.cmc.apply = warp
            if path is not None:
                tracker.joint_probe = JointProbe(path)
            outputs, events = [], []
            for rows in frames:
                dets = np.asarray(rows, np.float32).reshape(-1,6)
                packets = [{"rgb": Feature(np.array([1.,0.] if d[0]<70 else [0.,1.], np.float32))}
                           if d[4]>.5 else {} for d in dets]
                outputs.append(tracker.update(dets, image.copy(), features=packets).copy())
                events.append(deepcopy(tracker.last_events))
            summary = tracker.joint_probe.close() if path is not None else None
            memories = {tid: {"prototypes": deepcopy(m.prototypes), "counts": deepcopy(m.update_counts)}
                        for tid,m in tracker.memories.items()}
            return outputs, events, int(BaseTrack._count), memories, summary
        reference = replay()
        with tempfile.TemporaryDirectory() as directory:
            observed = replay(Path(directory)/"probe.jsonl.gz")
            for left,right in zip(reference[0], observed[0]):
                np.testing.assert_array_equal(left,right)
            self.assertEqual(json_value(reference[1:4]), json_value(observed[1:4]))
            self.assertEqual(observed[4]["status"], "PASS")
            self.assertGreater(observed[4]["counts"]["parity_checks"], 0)
            self.assertTrue(any(e.get("event")=="recovered" for frame in reference[1] for e in frame))


if __name__ == "__main__":
    unittest.main()
