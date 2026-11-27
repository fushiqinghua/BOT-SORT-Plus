"""Real worker CLI tests with explicit synthetic software inputs.

JPEGs, unit-vector feature caches and prescribed affine matrices are test
fixtures, not VT-Tiny-MOT results or actual LightGlue estimates. The original
fixed BoxMOT implementation is exercised; inference weights are not downloaded.
"""
from copy import deepcopy
import gzip
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

REPO = Path(__file__).resolve().parents[1]
NATIVE_AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("boxmot", "torch", "cv2"))


@unittest.skipUnless(NATIVE_AVAILABLE, "requires optional pinned native runtime")
class JointWorkerCLITests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("joint_runner_test_fixture", REPO / "tests/botsort_plus/test_runner.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fixture = module.RunnerTests("test_missing_image_preflight_exits_two")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.root = fixture.root
        self.sequence = module.SEQ
        self.environment = dict(os.environ)
        self.environment.pop("TRACKING_RUNTIME_ROOT", None)
        self.environment["CUDA_VISIBLE_DEVICES"] = ""
        self.environment["PYTHONPATH"] = os.pathsep.join([
            str(REPO), str(REPO / "tools"), self.environment.get("PYTHONPATH", "")])
        self.environment["OPENBLAS_NUM_THREADS"] = "1"
        self.environment["OMP_NUM_THREADS"] = "1"
        self.environment["PYTHONDONTWRITEBYTECODE"] = "1"
        # The main runner accepts an explicitly selected software subset. It
        # must not masquerade as the official all-sequence experiment.
        protocol = json.loads((fixture.baseline / "protocol.json").read_text())
        protocol["sequences"][self.sequence + "_UNSELECTED"] = deepcopy(protocol["sequences"][self.sequence])
        self.write(fixture.baseline / "protocol.json", protocol)
        self.settings = {"schema": "v04_joint_batch_v1", "baseline_root": str(fixture.baseline),
                         "features_dir": str(fixture.features), "images_root": str(fixture.images),
                         "gmc_mode": "live", "cuda_visible_devices": ""}

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    def invoke(self, method, label, *, probe="off", settings=None, preflight=False):
        output = self.root / label
        settings_path = self.root / (label + "_prediction_settings.json")
        self.write(settings_path, settings or self.settings)
        command = [sys.executable, "-m", "experiments.vttiny_gmc.round3_joint.worker",
                   "--settings", str(settings_path), "--method", method, "--output-dir", str(output),
                   "--sequences", self.sequence, "--probe-mode", probe]
        if preflight:
            command.append("--preflight")
        result = subprocess.run(command, cwd=REPO, env=self.environment, text=True,
                                capture_output=True, timeout=75)
        return result, output

    def success(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + "\n" + result.stderr)

    def manifest(self, output, method):
        from botsort_plus.io import sha256
        protocol = json.loads((output / "protocols" / (method + ".json")).read_text())
        self.assertEqual(protocol["status"], "COMPLETE")
        self.assertTrue(protocol["subset"])
        self.assertEqual(protocol["sequences"], {self.sequence: 6})
        self.assertFalse(protocol["joint_batch"]["ground_truth_read"])
        self.assertFalse(protocol["joint_batch"]["pending_identity"])
        self.assertEqual(protocol["backfilled_frames"], 0)
        mot = output / "trackers" / method / "data" / (self.sequence + ".txt")
        self.assertEqual(protocol["output_prediction_hashes"][self.sequence], sha256(mot))
        camera = protocol["joint_batch"]["camera_manifests"][self.sequence]
        self.assertEqual(camera["frames"], 6)
        self.assertEqual(camera["warp_sha256"], sha256(output / "warps" / method / (self.sequence + ".jsonl.gz")))
        return protocol

    def same_outputs(self, left, right, method):
        for folder, suffix in (("trackers", ".txt"), ("events", ".jsonl"), ("decisions", ".jsonl")):
            middle = Path("data") if folder == "trackers" else Path()
            a = left / folder / method / middle / (self.sequence + suffix)
            b = right / folder / method / middle / (self.sequence + suffix)
            self.assertEqual(a.read_bytes(), b.read_bytes(), f"probe changed {folder}")
        p = self.manifest(right, method)
        probe = p["joint_batch"]["probe_manifests"][self.sequence]
        self.assertEqual(probe["status"], "PASS")
        self.assertGreater(probe["counts"]["parity_checks"], 0)
        from botsort_plus.io import sha256
        self.assertEqual(probe["data_sha256"], sha256(right / "probes" / method / (self.sequence + ".jsonl.gz")))
        return p

    def test_frozen_B_worker_matches_archived_joint_adapter(self):
        method = "B_early_moderate"
        historical = self.root / "historical_snapshot_replay"
        script = self.root / "run_frozen_adapter.py"
        script.write_text(
            "import importlib.util, sys\n"
            "from pathlib import Path\n"
            "import botsort_plus.adapter_v2 as current\n"
            f"source=Path({str(REPO / 'experiments/vttiny_gmc/b_gmc_recovery/joint/adapter_v2.py')!r})\n"
            "spec=importlib.util.spec_from_file_location('botsort_plus._frozen_joint_adapter', source)\n"
            "old=importlib.util.module_from_spec(spec)\n"
            "sys.modules[spec.name]=old\n"
            "spec.loader.exec_module(old)\n"
            "current.update_v2=old.update_v2\n"
            "import run_botsort_plus as runner\n"
            "raise SystemExit(runner.run(runner.parser().parse_args()))\n", encoding="utf-8")
        command = [sys.executable, str(script), "--baseline-root", str(self.fixture.baseline),
                   "--features-dir", str(self.fixture.features), "--images-root", str(self.fixture.images),
                   "--config", str(REPO / "configs/botsort_plus/B_early_moderate.json"),
                   "--output-dir", str(historical), "--sequences", self.sequence, "--trace"]
        old = subprocess.run(command, cwd=REPO, env=self.environment, text=True,
                             capture_output=True, timeout=75)
        self.success(old)
        result, output = self.invoke(method, "worker_B")
        self.success(result)
        self.manifest(output, method)
        for folder, middle, suffix in (("trackers", Path("data"), ".txt"), ("events", Path(), ".jsonl")):
            relative = Path(folder) / method / middle / (self.sequence + suffix)
            self.assertEqual((historical / relative).read_bytes(), (output / relative).read_bytes())

    def test_skip_live_full_probe_preserves_reference(self):
        method = "B_skip_only"
        off, reference = self.invoke(method, "skip_off")
        self.success(off)
        on, observed = self.invoke(method, "skip_full", probe="full")
        self.success(on)
        protocol = self.same_outputs(reference, observed, method)
        self.assertEqual(protocol["joint_batch"]["camera"], "skip_only")
        self.assertFalse(protocol["joint_batch"]["dual_low"])
        camera = protocol["joint_batch"]["camera_manifests"][self.sequence]
        self.assertEqual(camera["mode"], "live")
        self.assertTrue(set(camera["choices"]) <= {"sof", "identity"})
        self.assertIsNone(protocol["joint_batch"]["model_reference"])
        self.assertNotIn("actual_model_runtime", protocol["joint_batch"])

    def synthetic_adaptive_cache(self):
        """Prescribed affine fixture, deliberately not a claimed model estimate."""
        result, reference = self.invoke("B_early_moderate", "cache_input_native")
        self.success(result)
        source = self.root / "SYNTHETIC_AFFINE_CACHE_SOFTWARE_ONLY"
        method = "synthetic_adaptive_fixture"
        protocol = self.manifest(reference, "B_early_moderate")
        contract = json.loads((REPO / "experiments/vttiny_gmc/reference.json").read_text())["gmc_factorial"]
        protocol["gmc_factorial"] = deepcopy(contract)
        protocol["gmc_factorial"].update(skip=True, lightglue=True)
        protocol["synthetic_software_test_only"] = True
        protocol["fixture_note"] = "Prescribed affine matrix to exercise guard; no LightGlue inference performed"
        self.write(source / "protocols" / (method + ".json"), protocol)
        with gzip.open(reference / "warps/B_early_moderate" / (self.sequence + ".jsonl.gz"), "rt") as stream:
            rows = [json.loads(line) for line in stream]
        # B recovered the old track at frame 5. At frame 6 the selected matrix
        # keeps a weak association plausible while the backup moves four px.
        row = rows[-1]
        row.update(native=[[1.,0.,4.],[0.,1.,0.]], used=[[1.,0.,0.],[0.,1.,0.]],
                   backup=[[1.,0.,4.],[0.,1.,0.]], choice="lightglue", eligible=["lightglue"],
                   trigger=True, model_called=True)
        path = source / "warps" / method / (self.sequence + ".jsonl.gz")
        path.parent.mkdir(parents=True)
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        from botsort_plus.io import sha256
        settings = deepcopy(self.settings)
        settings.update(gmc_mode="cached", warp_sources={"adaptive_gmc": {
            "run_dir": str(source), "method": method, "sha256": {self.sequence: sha256(path)}}})
        return settings

    def test_cached_adaptive_guard_and_evidence_probe_parity_without_cuda_or_lightglue(self):
        settings = self.synthetic_adaptive_cache()
        for method in ("B_adaptive_no_dual_low", "B_early_moderate_dual_low", "evidence_both"):
            with self.subTest(method=method):
                off, reference = self.invoke(method, method + "_off", settings=settings)
                self.success(off)
                on, observed = self.invoke(method, method + "_full", settings=settings, probe="full")
                self.success(on)
                protocol = self.same_outputs(reference, observed, method)
                camera = protocol["joint_batch"]["camera_manifests"][self.sequence]
                self.assertEqual(camera["mode"], "cached")
                self.assertEqual(camera["choices"], {"sof": 5, "lightglue": 1})
                self.assertIsNone(protocol["joint_batch"]["model_reference"])
                self.assertNotIn("actual_model_runtime", protocol["joint_batch"])
                self.assertTrue(camera["source"]["method"].startswith("synthetic_"))
                actions = json.loads((observed / "actions" / method / (self.sequence + ".json")).read_text())
                self.assertEqual(actions["lightglue_frames"], 1)
                if method != "B_adaptive_no_dual_low":
                    self.assertGreater(actions["blocked_native_edges"], 0)
                    counts = protocol["joint_batch"]["probe_manifests"][self.sequence]["counts"]
                    self.assertEqual(counts["native_counterfactual_unavailable"], 6)
                    if method == "evidence_both":
                        self.assertEqual(counts["recovery_counterfactual_unavailable"], 6)

    def test_preflight_is_persisted_and_prediction_rejects_gt_settings(self):
        result, output = self.invoke("B_skip_only", "preflight", preflight=True)
        self.success(result)
        preflight = json.loads((output / "preflight.json").read_text())
        self.assertEqual(preflight["status"], "READY")
        self.assertTrue(preflight["subset"])
        self.assertFalse((output / "trackers").exists())
        invalid = dict(self.settings, gt_root=str(self.root / "DO_NOT_READ_LABELS"))
        failed, output = self.invoke("B_skip_only", "invalid_gt_settings", settings=invalid)
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn("prediction must not receive evaluation settings", failed.stderr)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
