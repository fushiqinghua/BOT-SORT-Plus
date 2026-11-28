"""Execute patched upstream bodies to verify the R4.2 publication boundary.

Native tracking/heavy dependencies are doubles; AST function bodies come
from the actual pinned adapter and worker. Full native/GPU parity remains
the user's technical smoke/full check, not a claim of these unit checks.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / "tests"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from test_wem_r41_hooks import compile_function
from tools.research.wem_r42 import batch


class PublicationHookTests(unittest.TestCase):
    def test_real_adapter_bookkeeping_precedes_publish_and_finish(self):
        batch.legacy_imports()
        adapter_source = (ROOT / "botsort_plus/adapter_v2.py").read_text(encoding="utf-8")
        worker_source = (ROOT / "experiments/vttiny_gmc/round3_joint/worker.py").read_text(encoding="utf-8")
        for mode in batch.MODES:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                trace, native_recorded_rows, finish_rows, native_thresholds = [], [], [], []

                class NativeSTrack:
                    @staticmethod
                    def multi_predict(pool): trace.append("predict")
                    @staticmethod
                    def multi_gmc(pool, warp): trace.append("gmc")

                def record_native(tracker, output):
                    trace.append("native_record_output")
                    native_recorded_rows.append(len(output))

                namespace = dict(np=np, STrack=NativeSTrack,
                    TrackState=SimpleNamespace(Tracked=1, Lost=2, Removed=3),
                    joint_stracks=lambda a, b: list(a) + list(b),
                    iou_distance=lambda a, b: np.empty((len(a), len(b))),
                    linear_assignment=lambda cost, thresh: ([], [], []),
                    _forget=lambda tracker, ids: None, _record_output=record_native)
                native_update = compile_function(batch.previous._adapter_source(adapter_source), "update_v2", namespace)

                class NativeTracker:
                    def __init__(self, **kwargs):
                        self.frame_count = 0
                        self.track_low_thresh = .1
                        self.cmc = SimpleNamespace(apply=lambda *args: None)
                        self.plus = SimpleNamespace(raw_ir=SimpleNamespace(enabled=False),
                            identity=SimpleNamespace(enabled=False), birth=SimpleNamespace(enabled=False),
                            birth_identity=SimpleNamespace(enabled=False), appearance=SimpleNamespace(enabled=False),
                            recovery=SimpleNamespace(enabled=True, window=5, output_prediction_frames=0))
                        self.active_tracks, self.lost_stracks, self.removed_stracks = [], [], []
                        self.last_events, self.memories, self.prediction_history = [], {}, {}
                        self.max_obs, self.match_thresh, self.max_time_lost = 50, .8, 30
                        self.candidate_pool = SimpleNamespace(candidates={}, events=[], update=lambda *a, **kw: [])
                        self.recovery_matcher = SimpleNamespace(match=lambda *a: SimpleNamespace(
                            reserved_candidate_ids=set(), diagnostics=[], matches=[]))
                    def _split_detections(self, dets, dense):
                        native_thresholds.append(self.track_low_thresh)
                        return (np.empty((0, 7)),) * 4
                    def _separate_tracks(self): return [], []
                    def _associate(self, *args, **kwargs): return [], []
                    def _update_track_states(self, removed): trace.append("update_track_states")
                    def _prepare_output(self, *args):
                        trace.append("native_finalize")
                        return np.empty((0, 8))
                    def update(self, detections, image, *args, **kwargs):
                        trace.append("native_update")
                        return native_update(self, detections, image, kwargs["features"])
                    def finish_sequence_events(self): trace.append("native_close")

                class Controller:
                    def __init__(self, path, seq, method, output, protocol, actual_mode, settings, encoder):
                        trace.append("controller_init")
                        self.mode, self.encoder = actual_mode, encoder
                    def start(self, tracker, detections, image, features): trace.append("start")
                    def pre_gmc(self, *args): trace.append("pre_gmc")
                    def compensated(self, *args): trace.append("compensated")
                    def finish_before_output(self, tracker, activated, refound, lost, removed, reserved, consumed, all_dets):
                        trace.append("weak_finalize")
                        return tracker._prepare_output(activated, refound, lost, removed)
                    def publish(self, result):
                        trace.append("publish")
                        if self.mode == "W0_emit_only":
                            return np.array([[10., 10., 20., 20., 17, .08, 0, 0]])
                        return result
                    def finish(self, tracker, result):
                        trace.append("finish")
                        finish_rows.append(len(result))
                    def close(self): trace.append("controller_close")

                class Camera:
                    def __init__(self, *args): pass
                    def apply(self, *args):
                        trace.append("camera")
                        warp = np.eye(2, 3)
                        return warp, warp, warp, {"choice": "native"}
                    def close(self): return {}

                adapter = SimpleNamespace(BoTSORTPlus=NativeTracker, verify_upstream=lambda: None)
                cfg = SimpleNamespace(adapter_mode="v2", recovery=SimpleNamespace(enabled=True, early_evidence="moderate"))
                upstream = ModuleType("botsort_plus.adapter_v2")
                upstream.verify_v2_upstream = lambda: None
                observed_mode = "off" if mode == "lower_low" else mode
                namespace = dict(original_prepare=lambda parsed: (cfg, {"S": Path("cache")},
                                     {"status": "READY", "native_config": {"cmc_method": "sof"}}),
                    original_tracker=NativeTracker, adapter=adapter, settings={},
                    policy={"camera": "native", "dual_low": False, "evidence_both": False},
                    setup_matcher=lambda *args: (None, None), np=np,
                    SCHEMA="test", BASE_COMMIT="test", source_hashes=lambda: {},
                    probe_mode="off", preflight=False, output=Path(directory), method="evidence_both",
                    trackers=[], WarpCamera=Camera, json=json,
                    sha256=lambda path: "test", write_json=lambda *args: None,
                    _r41_closed=Controller, _r41_mode=observed_mode, _r41_rgb_modes=frozenset(),
                    _r41_encoder=lambda settings: self.fail("R4.2 must not instantiate an RGB encoder"))
                prepare = compile_function(batch.worker_source(worker_source, mode), "prepare", namespace)
                with patch.dict(sys.modules, {upstream.__name__: upstream}):
                    prepare(None)
                tracker = adapter.BoTSORTPlus()
                result = tracker.update(np.empty((0, 6)), np.zeros((8, 8, 3)), features=[])
                tracker.finish_sequence_events()
                expected_rows = int(mode == "W0_emit_only")
                self.assertEqual(result.shape, (expected_rows, 8))
                self.assertEqual(native_recorded_rows, [0])
                self.assertEqual(finish_rows, [expected_rows])
                self.assertEqual(native_thresholds, [.05 if mode == "lower_low" else .1])
                self.assertEqual(tracker.wem.mode, observed_mode)
                self.assertIsNone(tracker.wem.encoder)
                self.assertEqual(trace, ["controller_init", "start", "native_update", "predict", "pre_gmc",
                    "camera", "gmc", "gmc", "compensated", "update_track_states", "weak_finalize",
                    "native_finalize", "native_record_output", "publish", "finish", "controller_close", "native_close"])
                rows = [json.loads(line) for line in tracker.audit_path.read_text().splitlines()]
                output_event, = [row for row in rows if row["event"] == "output_index"]
                self.assertEqual(len(output_event["rows"]), expected_rows)
                self.assertEqual([row["event"] for row in tracker.last_events], ["frame"])

    def test_child_records_effective_threshold_and_state_feedback_semantics(self):
        for mode in ("lower_low", "W0_emit_only", "W0_relaxed_anchor"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                bundle = root / "bundle.json"
                batch.write(bundle, dict(schema="wem_r42_prediction_bundle_v1", settings={},
                                        code_sha256={}, sequences={"S": 1}, protocol={"fixed": True}))
                output = root / "output"
                def run(settings, method, folder, sequences, probe_mode):
                    self.assertEqual((method, sequences, probe_mode), ("evidence_both", ["S"], "off"))
                    batch.write(folder / "protocols/evidence_both.json",
                                dict(status="COMPLETE", native_config={"track_low_thresh": .1}))
                    return 0
                with patch.object(batch, "predictor", return_value=SimpleNamespace(run=run)) as predictor:
                    self.assertEqual(batch.child(bundle, mode, output, ["S"]), 0)
                    predictor.assert_called_once_with(mode)
                protocol = batch.read(output / "protocols/evidence_both.json")
                self.assertEqual(protocol["native_config"]["track_low_thresh"], .05 if mode == "lower_low" else .1)
                self.assertEqual(protocol["wem_r42"]["weak_state_feedback"], mode == "W0_relaxed_anchor")
                if mode == "lower_low":
                    self.assertEqual(protocol["wem_low_control"], dict(before=.1, effective=.05,
                                                                       camera_fingerprints_use_original_settings=True))
                if mode == "W0_emit_only":
                    self.assertEqual(protocol["wem_r42"]["state_chain_output_basis"], "canonical_native_output")
                    self.assertEqual(protocol["wem_r42"]["output_only_publication_boundary"], "after_complete_native_update")


if __name__ == "__main__":
    unittest.main()
