"""Synthetic asset, execution-boundary and resume tests (not dataset results)."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

from tools.research.wem_r41 import assets, batch


REAL_PROTOCOL = assets.read(assets.ROOT / "configs/research/wem_r4_1_protocol.json")


class Fixture:
    """A tiny-byte fixture with the exact 30 real names/lengths, no real data."""
    def __init__(self, root):
        self.root = root
        self.lengths = REAL_PROTOCOL["sequences"]
        self.d = root / "data"
        for directory in ("baseline", "features", "images", "warps", "gt", "trackeval/trackeval"):
            (self.d / directory).mkdir(parents=True)
        (self.d / "trackeval/trackeval/__init__.py").write_text("# synthetic evaluator\n")
        self.put("provenance/trackeval.json", {"commit": "synthetic", "package_sha256": {
            "trackeval/__init__.py": assets.sha(self.d / "trackeval/trackeval/__init__.py")}})
        baseline = {"sequences": {name: [None] * n for name, n in self.lengths.items()}}
        self.put("data/baseline/protocol.json", baseline)
        self.weights = self.d / "clip_veri.pt"
        self.weights.write_bytes(b"SYNTHETIC WEIGHTS - NOT A MODEL")
        identity = {"detector_sha256": "d" * 64, "rgb_encoder_sha256": assets.sha(self.weights),
                    "source_protocol_sha256": assets.sha(self.d / "baseline/protocol.json")}
        proto = deepcopy(REAL_PROTOCOL)
        proto["input_identity"] = identity
        proto["upstream_git_blobs"] = {}  # Only the isolated fixture has no installed runtime.
        self.put("configs/research/wem_r4_1_protocol.json", proto)
        self.put("configs/botsort_plus/B_early_moderate.json", {})
        for path in ("tools/run_wem_r41.py", "tools/run_botsort_plus.py", "tools/evaluate_botsort_plus.py"):
            file = root / path
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text("# synthetic launcher source\n")
        features, gt = {}, {}
        for name in self.lengths:
            f = self.d / "features" / (name + ".npz")
            f.write_bytes(("synthetic feature " + name).encode())
            self.put("data/features/" + name + ".json", {"fixture": name})
            features[name] = {"features": assets.sha(f), "metadata": assets.sha(f.with_suffix(".json"))}
            truth = self.d / "gt" / name / "gt/gt.txt"
            truth.parent.mkdir(parents=True)
            truth.write_text("", encoding="utf-8")
            gt[name] = assets.sha(truth)
        warps = {}
        for camera in ("native", "adaptive_gmc"):
            self.put("data/warps/protocols/" + camera + ".json", {"status": "COMPLETE", "sequences": self.lengths})
            hashes = {}
            for name in self.lengths:
                f = self.d / "warps/warps" / camera / (name + ".jsonl.gz")
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_bytes(("synthetic warp " + camera + name).encode())
                hashes[name] = assets.sha(f)
            warps[camera] = {"run_dir": str(self.d / "warps"), "method": camera, "sha256": hashes}
        prediction = {"schema": "v04_joint_batch_v1", "gmc_mode": "cached",
                      "baseline_root": str(self.d / "baseline"), "features_dir": str(self.d / "features"),
                      "images_root": str(self.d / "images"), "warp_sources": warps}
        self.put("prediction.json", prediction)
        inputs = {"sequences": self.lengths, "frames": 12590, **identity, "features": features,
                  "ground_truth": {"root": str(self.d / "gt"), "sha256": gt},
                  "trackeval": {"root": str(self.d / "trackeval")}}
        self.put("evaluation.json", inputs)
        refs, parity = {}, {"status": "PASS", "sequences": list(self.lengths), "references": {}}
        for role, method in assets.METHODS.items():
            path = f"references/{role}/protocols/{method}.json"
            self.put(path, {"status": "COMPLETE", "sequences": self.lengths,
                            "native_config": {"track_low_thresh": .1}, **identity})
            hashes = {name: "1" * 64 for name in self.lengths}
            refs[role] = {"run_dir": str(root / f"references/{role}"), "tracker": method,
                          "protocol_sha256": assets.sha(root / path), "trajectory_hashes": hashes}
            parity["references"][role] = {"files": {name: {"mot_sha256": hashes[name],
                "events_sha256": "2" * 64, "state_chain_sha256": "3" * 64} for name in self.lengths}}
        self.put("references.json", {"references": refs})
        self.put("parity.json", parity)
        settings = deepcopy(assets.SETTINGS)
        settings["paths"] = {"prediction_settings": str(root / "prediction.json"),
            "r0_parity": str(root / "parity.json"), "reference_manifest": str(root / "references.json"),
            "evaluation_manifest": str(root / "evaluation.json"), "rgb_weights": str(self.weights),
            "output_root": str(root / "run")}
        self.settings = root / "settings.json"
        self.put("settings.json", settings)

    def put(self, path, value):
        assets.write(self.root / path, value)

    def update(self, path, mutate):
        value = assets.read(self.root / path)
        mutate(value)
        self.put(path, value)

    def prepare(self):
        with patch.object(assets, "ROOT", self.root):
            return assets.prepare(assets.load(self.settings))


class AssetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = Fixture(Path(self.temp.name))

    def test_complete_scope_and_gt_separation(self):
        context = self.fixture.prepare()
        bundle = context["bundle"]
        self.assertEqual(len(bundle["sequences"]), 30)
        self.assertEqual(sum(bundle["sequences"].values()), 12590)
        self.assertEqual(context["receipt"]["status"], "INPUTS_VERIFIED_NOT_RUN")
        self.assertNotIn(str(self.fixture.d / "gt"), json.dumps(bundle))
        self.assertNotIn("ground_truth", bundle["settings"])
        self.assertEqual(context["receipt"]["input_sha256"]["trackeval_verified"]["verified_source_files"], 1)

    def test_gt_path_in_prediction_is_rejected(self):
        self.fixture.update("prediction.json", lambda x: x.update(gt_root="secret_gt"))
        with self.assertRaisesRegex(ValueError, "prediction-only"):
            self.fixture.prepare()

    def test_cannot_replace_benchmark_with_subset(self):
        self.fixture.update("data/baseline/protocol.json", lambda x: x["sequences"].pop(next(iter(x["sequences"]))))
        with self.assertRaisesRegex(ValueError, "30-sequence"):
            self.fixture.prepare()

    def test_tampered_feature_warp_weights_gt_or_evaluator_are_rejected(self):
        name = next(iter(self.fixture.lengths))
        paths = [self.fixture.d / f"features/{name}.npz", self.fixture.d / f"warps/warps/native/{name}.jsonl.gz",
                 self.fixture.weights, self.fixture.d / f"gt/{name}/gt/gt.txt",
                 self.fixture.d / "trackeval/trackeval/__init__.py"]
        for path in paths:
            original = path.read_bytes()
            with self.subTest(path=path):
                path.write_bytes(original + b"changed")
                try:
                    with self.assertRaises(ValueError):
                        self.fixture.prepare()
                finally:
                    path.write_bytes(original)

    def test_incomplete_parity_is_rejected(self):
        self.fixture.update("parity.json", lambda x: x["sequences"].pop())
        with self.assertRaisesRegex(ValueError, "R0 parity"):
            self.fixture.prepare()

    def test_reference_protocol_drift_is_rejected(self):
        self.fixture.update("references/B/protocols/B_early_moderate.json", lambda x: x.update(status="FAILED"))
        with self.assertRaisesRegex(ValueError, "reference protocol"):
            self.fixture.prepare()

    def test_configuration_errors_are_not_silently_accepted(self):
        with patch.object(assets, "ROOT", self.fixture.root):
            with self.assertRaisesRegex(ValueError, "Unresolved"):
                assets.absolute("${WEM_R41_UNSET_VARIABLE_947}/data")
            self.fixture.update("settings.json", lambda x: x["runtime"].update(workers=True))
            with self.assertRaisesRegex(ValueError, "positive integer"):
                assets.load(self.fixture.settings)

    def test_init_keeps_existing_settings_and_supports_legacy_layout(self):
        path = self.fixture.root / "initialized.json"
        assets.init(path, self.fixture.root)
        data = assets.read(path)
        self.assertIn("vttiny_wem_r01_v1", data["paths"]["prediction_settings"])
        with self.assertRaises(FileExistsError):
            assets.init(path)


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_all_arms_are_fixed_and_include_both_references(self):
        self.assertEqual(len(assets.FULL_ARMS), 10)
        self.assertEqual(set(assets.FULL_ARMS), {(r, m) for r in ("B", "BG") for m in ("off", "W0", "W1", "W2", "W3")})
        self.assertNotIn(("B", "W_H"), assets.FULL_ARMS)

    def test_full_does_not_gate_on_r3_reject_or_manual_review(self):
        called = []
        context = {"settings": {"paths": {"output_root": str(self.root)}},
                   "receipt": {"prior_r3": "REJECT_ALL", "human_reviews": 0}}
        def stage(ctx, name):
            called.append(name)
            return self.root / name
        with patch.object(batch, "load", return_value={}), patch.object(batch, "prepare", return_value=context), \
             patch.object(batch, "stage", side_effect=stage), \
             patch.object(batch, "evaluate", side_effect=lambda *_: called.append("evaluate")), \
             patch.object(batch, "summarize", side_effect=lambda *_: called.append("summarize")):
            self.assertEqual(batch.main(["full", "--settings", "unused.json"]), 0)
        self.assertEqual(called, ["smoke", "full", "evaluate", "summarize"])

    def test_technical_parity_failure_prevents_invalid_full_run(self):
        context = {"settings": {"paths": {"output_root": str(self.root)}}, "receipt": {}}
        with patch.object(batch, "load", return_value={}), patch.object(batch, "prepare", return_value=context), \
             patch.object(batch, "stage", side_effect=ValueError("parity mismatch")), patch.object(batch, "evaluate") as evaluate:
            with self.assertRaisesRegex(ValueError, "parity mismatch"):
                batch.main(["full", "--settings", "unused.json"])
            evaluate.assert_not_called()

    def test_resume_verifies_all_outputs_and_input_identity(self):
        folder = self.root / "B_W0"
        assets.write(folder / "actual.json", {"frames": 30})
        assets.write(folder / "arm_complete.json", {"status": "PREDICTION_COMPLETE", "arm": "B_W0",
            "fingerprint": "fixed", "sequences": ["seq"], "files": batch._file_manifest(folder)})
        self.assertTrue(batch.completed_arm(folder, "B_W0", "fixed", ["seq"]))
        with self.assertRaisesRegex(ValueError, "different experiment"):
            batch.completed_arm(folder, "B_W0", "different", ["seq"])
        assets.write(folder / "actual.json", {"frames": 29})
        with self.assertRaisesRegex(ValueError, "output changed"):
            batch.completed_arm(folder, "B_W0", "fixed", ["seq"])

    def test_incomplete_arm_is_preserved_and_complete_arm_reused(self):
        folder = self.root / "B_W0"
        assets.write(folder / "incomplete.json", {"old": True})
        (self.root / "B_W0.log").write_text("previous failed run\n")
        context = {"fingerprint": "fixed", "bundle": {"sequences": {"seq": 2}},
                   "settings": {"runtime": {"python": "python"}}}
        def predict(*args, **kwargs):
            assets.write(folder / "protocols/B_early_moderate.json", {"status": "COMPLETE", "sequences": {"seq": 2}})
        with patch.object(batch, "environment", return_value={}), patch.object(batch.subprocess, "run", side_effect=predict) as run:
            batch.track(context, self.root, "B", "W0", ["seq"])
            batch.track(context, self.root, "B", "W0", ["seq"])
            self.assertEqual(run.call_count, 1)
        preserved = list((self.root / "superseded").rglob("incomplete.json"))
        self.assertEqual(len(preserved), 1)
        self.assertTrue(assets.read(preserved[0])["old"])

    def test_lock_is_released_and_existing_windows_lock_is_never_killed(self):
        with batch.run_lock(self.root):
            self.assertTrue((self.root / ".wem_r41_running.json").exists())
        self.assertFalse((self.root / ".wem_r41_running.json").exists())
        assets.write(self.root / ".wem_r41_running.json", {"pid": 123, "host": socket.gethostname()})
        # Keep pathlib on the real OS while mocking only the branch check.
        class WindowsOS:
            name = "nt"
            def __getattr__(self, name):
                return getattr(real_os, name)
        real_os = batch.os
        with patch.object(batch, "os", WindowsOS()), patch.object(real_os, "kill") as kill:
            with self.assertRaisesRegex(RuntimeError, "Windows batch lock"):
                with batch.run_lock(self.root):
                    pass
            kill.assert_not_called()

    def test_frozen_patch_boundaries_fail_closed(self):
        batch.legacy_imports()
        adapter = (assets.ROOT / "botsort_plus/adapter_v2.py").read_text()
        worker = (assets.ROOT / "experiments/vttiny_gmc/round3_joint/worker.py").read_text()
        compile(batch._adapter_source(adapter), "patched_adapter", "exec")
        compile(batch._worker_source(worker), "patched_worker", "exec")
        with self.assertRaisesRegex(ValueError, "Expected one pinned"):
            batch._adapter_source(adapter.replace("    STrack.multi_predict(pool)\n", ""))


if __name__ == "__main__":
    unittest.main()
