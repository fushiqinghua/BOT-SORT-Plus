"""Execute the patched upstream worker and adapter on an empty native frame.

The AST bodies are the real pinned source, not reimplemented patch snippets.
Heavy native dependencies are minimal doubles: this proves hook wiring/order,
not full BoxMOT/data performance (the user's smoke run verifies that).
"""
from __future__ import annotations

import ast
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.research.wem_r41 import batch


def compile_function(source, name, namespace):
    tree = ast.parse(source)
    node, = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(module, "<real-patched-upstream>", "exec"), namespace)
    return namespace[name]


class HookWiringTests(unittest.TestCase):
    def test_worker_and_adapter_execute_both_hook_layers_in_order(self):
        batch.legacy_imports()
        adapter_source = (ROOT / "botsort_plus/adapter_v2.py").read_text(encoding="utf-8")
        worker_source = (ROOT / "experiments/vttiny_gmc/round3_joint/worker.py").read_text(encoding="utf-8")
        for mode in ("off", "W1", "W3", "shadow", "force_reject"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                trace = []

                class NativeSTrack:
                    @staticmethod
                    def multi_predict(pool): trace.append("predict")
                    @staticmethod
                    def multi_gmc(pool, warp): trace.append("gmc")

                namespace = dict(np=np, STrack=NativeSTrack,
                                 TrackState=SimpleNamespace(Tracked=1, Lost=2, Removed=3),
                                 joint_stracks=lambda a, b: list(a) + list(b),
                                 iou_distance=lambda a, b: np.empty((len(a), len(b))),
                                 linear_assignment=lambda cost, thresh: ([], [], []),
                                 _forget=lambda tracker, ids: None,
                                 _record_output=lambda tracker, output: trace.append("record_output"))
                native_update = compile_function(batch._adapter_source(adapter_source), "update_v2", namespace)

                class NativeTracker:
                    def __init__(self, **kwargs):
                        self.frame_count = 0
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
                        empty = np.empty((0, 7))
                        return empty, empty, empty, empty
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
                        self.mode = actual_mode
                        self.encoder = encoder
                    def start(self, tracker, detections, image, features): trace.append("start")
                    def pre_gmc(self, *args): trace.append("pre_gmc")
                    def compensated(self, *args): trace.append("compensated")
                    def finish_before_output(self, tracker, activated, refound, lost, removed, reserved, consumed, all_dets):
                        trace.append("weak_finalize")
                        return tracker._prepare_output(activated, refound, lost, removed)
                    def finish(self, tracker, result): trace.append("finish")
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
                encoder = object()
                globals_ = dict(original_prepare=lambda parsed: (cfg, {"S": Path("cache")},
                                   {"status": "READY", "native_config": {"cmc_method": "sof"}}),
                                original_tracker=NativeTracker, adapter=adapter, settings={},
                                policy={"camera": "native", "dual_low": False, "evidence_both": False},
                                setup_matcher=lambda *args: (None, None), np=np,
                                SCHEMA="test", BASE_COMMIT="test", source_hashes=lambda: {},
                                probe_mode="off", preflight=False, output=Path(directory), method="B_early_moderate",
                                trackers=[], WarpCamera=Camera, json=__import__("json"),
                                sha256=lambda path: "test", write_json=lambda *args: None,
                                _r41_closed=Controller, _r41_mode=mode,
                                _r41_rgb_modes={"W2", "W3", "shadow", "force_reject"},
                                _r41_encoder=lambda settings: trace.append("load_encoder") or encoder)
                prepare = compile_function(batch._worker_source(worker_source), "prepare", globals_)
                with patch.dict(sys.modules, {upstream.__name__: upstream}):
                    prepare(None)
                tracker = adapter.BoTSORTPlus()
                result = tracker.update(np.empty((0, 6)), np.zeros((8, 8, 3)), features=[])
                tracker.finish_sequence_events()
                self.assertEqual(result.shape, (0, 8))
                self.assertEqual(tracker.wem.mode, mode)
                self.assertEqual(tracker.wem.encoder is encoder, mode in globals_["_r41_rgb_modes"])
                expected = ["controller_init", "start", "native_update", "predict", "pre_gmc",
                            "camera", "gmc", "gmc", "compensated", "update_track_states",
                            "weak_finalize", "native_finalize", "record_output", "finish",
                            "controller_close", "native_close"]
                self.assertEqual([item for item in trace if item != "load_encoder"], expected)
                self.assertEqual(trace.count("load_encoder"), int(mode in globals_["_r41_rgb_modes"]))

    def test_patch_fails_when_expected_boundary_is_absent_or_repeated(self):
        batch.legacy_imports()
        adapter = (ROOT / "botsort_plus/adapter_v2.py").read_text(encoding="utf-8")
        worker = (ROOT / "experiments/vttiny_gmc/round3_joint/worker.py").read_text(encoding="utf-8")
        with self.assertRaises(ValueError):
            batch._adapter_source(adapter.replace("    STrack.multi_predict(pool)\n", ""))
        with self.assertRaises(ValueError):
            batch._worker_source(worker + "\n        sequence_iter = iter(paths)\n")


if __name__ == "__main__":
    unittest.main()
