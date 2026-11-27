"""Contract tests for local WEM preparation, not tracking/performance validation."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

SOURCE_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "manage_wem_batch_for_tests", SOURCE_ROOT / "tools/manage_wem_batch.py")
batch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch)


class PreparationContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="wem-准备-")
        self.root = Path(self.temp.name) / "repo with spaces"
        self.root.mkdir()
        self.previous_root = batch.REPO_ROOT
        batch.REPO_ROOT = self.root
        (self.root / "tools").mkdir()
        (self.root / "configs/research").mkdir(parents=True)
        self.template = json.loads((SOURCE_ROOT / batch.TEMPLATE).read_text(encoding="utf-8"))
        self.protocol = {"schema": batch.PROTOCOL_SCHEMA, "base_commit": batch.BASE_COMMIT,
                         "status": "DESIGN_ONLY"}
        self.protocol_path = self.root / "configs/research/wem_v0.1_protocol.json"
        self.write(self.protocol_path, self.protocol)
        self.write(self.root / batch.TEMPLATE, self.template)
        self.settings_path = self.root / "wem.local.json"
        self.settings = copy.deepcopy(self.template)

    def tearDown(self):
        batch.REPO_ROOT = self.previous_root
        self.temp.cleanup()

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")

    def configured(self):
        (self.root / "data/detections").mkdir(parents=True)
        self.settings["paths"]["detections_dir"] = "data/detections"
        self.write(self.settings_path, self.settings)
        return batch.inspect(self.settings_path)

    def test_missing_later_assets_do_not_block_inventory_preparation(self):
        info = self.configured()
        self.assertTrue(info["report"]["ready_for_r1a_inventory"])
        self.assertIn("gt_root", info["report"]["later_assets_missing"])
        report, code = batch.prepare(info)
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "PREPARED_NOT_RUN")
        self.assertEqual(report["experiment_acceptance"], "NOT_EVALUATED")
        self.assertEqual(report["next_command_shell"], "powershell" if os.name == "nt" else "posix")
        self.assertIn("--low-thresh", report["next_command"])
        self.assertFalse((self.root / "runs/wem_v0.1/r1a_s0").exists())

    def test_missing_detection_path_never_marks_gate_passed(self):
        self.write(self.settings_path, self.settings)
        info = batch.inspect(self.settings_path)
        self.assertFalse(info["report"]["ready_for_r1a_inventory"])
        report, code = batch.prepare(info)
        self.assertEqual(code, 3)
        self.assertEqual(report["status"], "NOT_PREPARED_MISSING_DETECTIONS")
        self.assertFalse(info["output"].exists())

    def test_exact_snapshots_idempotent_prepare_and_stale_status(self):
        info = self.configured()
        batch.prepare(info)
        manifest_path = info["output"] / "batch_manifest.json"
        original = manifest_path.read_bytes()
        second, code = batch.prepare(batch.inspect(self.settings_path))
        self.assertEqual(code, 0)
        self.assertTrue(second["idempotent"])
        self.assertEqual(original, manifest_path.read_bytes())
        self.protocol["extra_decision"] = "new input contract"
        self.write(self.protocol_path, self.protocol)
        changed = batch.inspect(self.settings_path)
        status, code = batch.status(changed)
        self.assertEqual(code, 3)
        self.assertEqual(status["status"], "STALE_OR_INVALID_PREPARATION")
        with self.assertRaises(batch.BatchError):
            batch.prepare(changed)
        self.assertEqual(original, manifest_path.read_bytes())

    def test_modified_snapshot_is_detected(self):
        info = self.configured()
        batch.prepare(info)
        (info["output"] / "prepared_settings.json").write_text("{}", encoding="utf-8")
        status, code = batch.status(info)
        self.assertEqual(code, 3)
        self.assertTrue(any("prepared_settings" in e for e in status["errors"]))

    def test_malformed_manifest_is_reported_without_overwriting(self):
        info = self.configured()
        batch.prepare(info)
        path = info["output"] / "batch_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["snapshots"] = []
        self.write(path, manifest)
        original = path.read_bytes()
        status, code = batch.status(info)
        self.assertEqual(code, 3)
        with self.assertRaises(batch.BatchError):
            batch.prepare(info)
        self.assertEqual(original, path.read_bytes())

    def test_inputs_sources_and_existing_destinations_are_protected(self):
        self.configured()
        for unsafe in ("data/detections/out", "data", ".", "tools/out", "results/wem"):
            settings = copy.deepcopy(self.settings)
            settings["output_root"] = unsafe
            self.write(self.settings_path, settings)
            with self.subTest(path=unsafe), self.assertRaises(batch.BatchError):
                batch.inspect(self.settings_path)
        self.write(self.settings_path, self.settings)
        info = batch.inspect(self.settings_path)
        info["output"].mkdir(parents=True)
        sentinel = info["output"] / "do-not-overwrite.txt"
        sentinel.write_text("keep", encoding="utf-8")
        with self.assertRaises(batch.BatchError):
            batch.prepare(info)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_settings_under_output_are_rejected(self):
        info = self.configured()
        nested = info["output"] / "settings.json"
        nested.parent.mkdir(parents=True)
        self.write(nested, self.settings)
        with self.assertRaises(batch.BatchError):
            batch.inspect(nested)

    def test_invalid_json_thresholds_and_duplicate_keys_fail_closed(self):
        for value in (True, False, float("nan"), float("inf"), "0.1", 0.05):
            settings = copy.deepcopy(self.settings)
            settings["thresholds"]["low"] = value
            self.write(self.settings_path, settings)
            with self.subTest(value=value), self.assertRaises(batch.BatchError):
                batch.inspect(self.settings_path)
        self.settings_path.write_text('{"schema":"x","schema":"y"}', encoding="utf-8")
        with self.assertRaises(batch.BatchError):
            batch.read_object(self.settings_path)
        self.settings_path.write_text('{"nested":[{"number":1e999}]}', encoding="utf-8")
        with self.assertRaises(batch.BatchError):
            batch.read_object(self.settings_path)

    @unittest.skipIf(os.name == "nt", "Windows paths are native on Windows")
    def test_windows_drive_not_misinterpreted_as_posix_relative(self):
        with self.assertRaises(batch.BatchError):
            batch.resolve_path(r"E:\data\detections", "detections_dir")

    def test_init_never_overwrites_and_cli_resolves_from_repository(self):
        report, code = batch.init("wem.local.json")
        self.assertEqual(code, 0)
        original = self.settings_path.read_bytes()
        with self.assertRaises(batch.BatchError):
            batch.init("wem.local.json")
        self.assertEqual(original, self.settings_path.read_bytes())
        self.configured()
        script = self.root / "tools/manage_wem_batch.py"
        script.write_bytes((SOURCE_ROOT / "tools/manage_wem_batch.py").read_bytes())
        external_cwd = Path(self.temp.name)
        result = subprocess.run(
            [sys.executable, "-X", "utf8", str(script), "check",
             "--settings", "wem.local.json", "--strict"],
            cwd=external_cwd, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["ready_for_r1a_inventory"])
        self.assertEqual(payload["paths"]["detections_dir"]["path"],
                         str(self.root / "data/detections"))

    def test_canonical_local_settings_cli_and_source_protection(self):
        script = self.root / "tools/manage_wem_batch.py"
        script.write_bytes((SOURCE_ROOT / "tools/manage_wem_batch.py").read_bytes())
        canonical = "configs/research/wem_v0.1.local.json"
        result = subprocess.run(
            [sys.executable, "-X", "utf8", str(script), "init", "--output", canonical],
            cwd=Path(self.temp.name), capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        created = self.root / canonical
        self.assertEqual(json.loads(created.read_text(encoding="utf-8")), self.template)
        original = created.read_bytes()
        with self.assertRaises(batch.BatchError):
            batch.init(canonical)
        self.assertEqual(original, created.read_bytes())
        for source in ("configs/research/wem_v0.1.example.json",
                       "configs/research/new_protocol.json",
                       "configs/other.local.json",
                       "tools/example.local.json"):
            with self.subTest(path=source), self.assertRaises(batch.BatchError):
                batch.init(source)

    def test_powershell_copyable_command_quoting(self):
        args = ["python", "-X", "utf8", r"C:\研究 仓库\tools\audit_wem_s0.py",
                "--detections-dir", r"E:\foo'bar$variable" + chr(96) + "literal",
                "--output-dir", r"E:\实验结果\round 1"]
        expected = (
            "& 'python' '-X' 'utf8' 'C:\\研究 仓库\\tools\\audit_wem_s0.py' "
            "'--detections-dir' 'E:\\foo''bar$variable" + chr(96) + "literal' "
            "'--output-dir' 'E:\\实验结果\\round 1'"
        )
        self.assertEqual(batch.format_command(args, windows=True), expected)

    def test_posix_copyable_command_round_trip(self):
        posix_args = ["python", "/repo with spaces/tools/audit_wem_s0.py",
                      "--detections-dir", "/缓存/O'Brien$variable" + chr(96) + " folder"]
        self.assertEqual(shlex.split(batch.format_command(posix_args, windows=False)),
                         posix_args)


if __name__ == "__main__":
    unittest.main()
