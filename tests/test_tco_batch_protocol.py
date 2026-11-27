"""Meaningful preparation-contract checks; no trained TCO model is exercised."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


TOOL = Path(__file__).resolve().parents[1] / "tools" / "manage_tco_batch.py"
SPEC = importlib.util.spec_from_file_location("tco_batch_contract_tool", TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TCOBatchContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "repo"
        (self.root / "tools").mkdir(parents=True)
        (self.root / "configs" / "research").mkdir(parents=True)
        self.foreign = Path(self.temp.name) / "unrelated-cwd"
        self.foreign.mkdir()
        shutil.copyfile(TOOL, self.root / "tools" / TOOL.name)
        self.protocol = self.root / "configs/research/tco_v0.1_protocol.json"
        self.tasks = self.root / "configs/research/tco_v0.1_tasks.json"
        self.config = self.root / "configs/research/local.json"
        self.protocol_contract = {"schema": "tco_v0.1_protocol", "base_commit": MODULE.BASE_COMMIT, "constraints": MODULE.CONSTRAINTS, "status": "DESIGNED_NOT_RUN"}
        self.task_contract = {"schema": "tco_v0.1_tasks", "base_commit": MODULE.BASE_COMMIT}
        self.write(self.protocol, self.protocol_contract)
        self.write(self.tasks, {**self.task_contract, "tasks": [{"id": "T0", "title": "Frozen baseline reproduction", "depends_on": []}, {"id": "T1", "title": "Cohort export", "depends_on": ["T0"]}]})
        self.settings = {
            "schema": MODULE.SCHEMA,
            "protocol_file": "configs/research/tco_v0.1_protocol.json",
            "tasks_file": "configs/research/tco_v0.1_tasks.json",
            "output_root": "outputs/research_batch",
            "paths": {},
            "references": {},
            "constraints": dict(MODULE.CONSTRAINTS),
        }
        for name, kind in MODULE.PATH_KINDS.items():
            path = self.root / "local_assets" / name
            if kind == "directory":
                path.mkdir(parents=True)
            else:
                path = path.with_suffix(".json")
                self.write(path, {"note": "Fixture only; mother-video separation has NOT been checked."})
            self.settings["paths"][name] = str(path.relative_to(self.root))
        for role, tracker in MODULE.REFERENCE_TRACKERS.items():
            path = self.root / "local_assets" / f"reference_{role}"
            self.write(path / "protocols" / f"{tracker}.json", {"fixture": True, "tracker": tracker})
            self.settings["references"][role] = {"run_dir": str(path.relative_to(self.root)), "tracker": tracker}
        self.write(self.config, self.settings)

    @staticmethod
    def write(path: Path, value: object, bom: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8-sig" if bom else "utf-8")

    def invoke(self, *arguments: str) -> tuple[int, dict]:
        run = subprocess.run(
            [sys.executable, "-X", "utf8", str(self.root / "tools" / TOOL.name), *arguments],
            cwd=self.foreign, capture_output=True, text=True, encoding="utf-8", check=False,
        )
        self.assertEqual(run.stderr, "", run.stderr)
        return run.returncode, json.loads(run.stdout)

    def test_init_handles_bom_and_repo_paths_and_refuses_overwrite(self) -> None:
        template = self.root / MODULE.DEFAULT_TEMPLATE
        template_settings = dict(self.settings)
        template_settings["paths"] = {key: None for key in MODULE.PATH_KINDS}
        template_settings["references"] = {key: {"run_dir": None, "tracker": tracker} for key, tracker in MODULE.REFERENCE_TRACKERS.items()}
        template_settings["note"] = "本地配置"
        self.write(template, template_settings, bom=True)
        destination = "configs/research/generated.local.json"
        code, result = self.invoke("init", "--output-config", destination)
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "CONFIG_CREATED")
        created = self.root / destination
        original = created.read_bytes()
        self.assertEqual(json.loads(original)["note"], "本地配置")
        code, _ = self.invoke("init", "--output-config", destination)
        self.assertEqual(code, 2)
        self.assertEqual(created.read_bytes(), original)

    def test_missing_assets_are_reported_without_writes_or_fake_success(self) -> None:
        self.settings["paths"]["images_root"] = None
        self.write(self.config, self.settings, bom=True)
        before = sorted(str(path.relative_to(self.root)) for path in self.root.rglob("*"))
        code, result = self.invoke("check", "--settings", "configs/research/local.json")
        self.assertEqual(code, 3)
        self.assertEqual(result["status"], "INPUTS_MISSING")
        self.assertEqual(result["experimental_execution"], "NOT_RUN")
        code, _ = self.invoke("check", "--settings", "configs/research/local.json", "--strict")
        self.assertEqual(code, 2)
        code, _ = self.invoke("prepare", "--settings", "configs/research/local.json")
        self.assertEqual(code, 3)
        self.assertEqual(sorted(str(path.relative_to(self.root)) for path in self.root.rglob("*")), before)

    def test_prepare_freezes_metadata_and_keeps_all_experiments_planned(self) -> None:
        assets = {path: path.read_bytes() for path in (self.root / "local_assets").rglob("*") if path.is_file()}
        code, result = self.invoke("prepare", "--settings", "configs/research/local.json")
        self.assertEqual(code, 0, result)
        self.assertEqual(result["status"], "PREPARED_NOT_RUN")
        output = self.root / self.settings["output_root"]
        manifest = json.loads((output / "batch_manifest.json").read_text())
        self.assertEqual(manifest["preparation_tool"]["sha256"], MODULE.sha256(self.root / "tools" / TOOL.name))
        self.assertIn("Mother-video separation", " ".join(manifest["not_checked"]))
        for record in manifest["snapshots"].values():
            self.assertEqual(MODULE.sha256(output / record["file"]), record["sha256"])
        ledger = json.loads((output / "task_ledger.json").read_text())
        self.assertEqual([row["state"] for row in ledger["tasks"]], ["PLANNED", "PLANNED"])
        self.assertEqual(assets, {path: path.read_bytes() for path in assets})
        first_manifest = (output / "batch_manifest.json").read_bytes()
        code, _ = self.invoke("prepare", "--settings", "configs/research/local.json")
        self.assertEqual(code, 2)
        self.assertEqual((output / "batch_manifest.json").read_bytes(), first_manifest)
        code, result = self.invoke("status", "--settings", "configs/research/local.json")
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "PREPARED_NOT_RUN")

    def test_invalid_boundaries_and_unsafe_destinations_fail_before_writes(self) -> None:
        mutations = [
            ("constraints", "allow_ir", True),
            ("constraints", "rgb_only", 1),
            ("output", None, "../escaped"),
            ("output", None, "configs/new-output"),
            ("output", None, "local_assets/images_root/new-output"),
        ]
        for kind, key, value in mutations:
            with self.subTest(kind=kind, key=key, value=value):
                modified = json.loads(json.dumps(self.settings))
                if kind == "constraints":
                    modified["constraints"][key] = value
                else:
                    modified["output_root"] = value
                self.write(self.config, modified)
                code, result = self.invoke("prepare", "--settings", "configs/research/local.json")
                self.assertEqual(code, 2, result)
                self.assertFalse((self.root / "outputs").exists())

    def test_invalid_metadata_and_duplicate_tasks_fail_before_preparation(self) -> None:
        self.write(self.tasks, {**self.task_contract, "tasks": [{"id": "T0", "depends_on": []}, {"id": "T0", "depends_on": []}]})
        code, result = self.invoke("prepare", "--settings", "configs/research/local.json")
        self.assertEqual(code, 2)
        self.assertTrue(any("Duplicate" in item for item in result["errors"]))
        self.write(self.tasks, {**self.task_contract, "tasks": [{"id": "T0", "depends_on": []}]})
        self.write(self.root / self.settings["paths"]["split_manifest"], [])
        code, result = self.invoke("prepare", "--settings", "configs/research/local.json")
        self.assertEqual(code, 2)
        self.assertTrue(any("JSON root" in item for item in result["errors"]))
        self.assertFalse((self.root / "outputs").exists())

    def test_contract_identity_and_task_dependencies_are_enforced(self) -> None:
        valid_tasks = {**self.task_contract, "tasks": [{"id": "T0", "depends_on": []}, {"id": "T1", "depends_on": ["T0"]}]}
        bad_documents = [
            (self.protocol, {**self.protocol_contract, "schema": "unrelated_protocol"}),
            (self.protocol, {**self.protocol_contract, "base_commit": "0" * 40}),
            (self.tasks, {**valid_tasks, "schema": "unrelated_tasks"}),
            (self.tasks, {**valid_tasks, "base_commit": "0" * 40}),
            (self.tasks, {**self.task_contract, "tasks": [{"id": "T0", "depends_on": ["MISSING"]}]}),
            (self.tasks, {**self.task_contract, "tasks": [{"id": "T0", "depends_on": ["T1"]}, {"id": "T1", "depends_on": ["T0"]}]}),
        ]
        for path, document in bad_documents:
            with self.subTest(path=path.name, document=document):
                self.write(self.protocol, self.protocol_contract)
                self.write(self.tasks, valid_tasks)
                self.write(path, document)
                code, result = self.invoke("prepare", "--settings", "configs/research/local.json")
                self.assertEqual(code, 2, result)
                self.assertEqual(result["status"], "INVALID")
                self.assertFalse((self.root / "outputs").exists())

    def test_snapshot_tampering_is_detected_and_user_labels_are_not_certified(self) -> None:
        code, _ = self.invoke("prepare", "--settings", "configs/research/local.json")
        self.assertEqual(code, 0)
        output = self.root / self.settings["output_root"]
        ledger_file = output / "task_ledger.json"
        ledger = json.loads(ledger_file.read_text())
        ledger["tasks"][0]["state"] = "PASS"
        self.write(ledger_file, ledger)
        code, result = self.invoke("status", "--settings", "configs/research/local.json")
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "LEDGER_CHANGED_UNVERIFIED")
        self.assertEqual(result["experimental_acceptance"], "NOT_VALIDATED_BY_THIS_TOOL")
        copied_tool = self.root / "tools" / TOOL.name
        with copied_tool.open("a", encoding="utf-8") as stream:
            stream.write("\n# Deliberate software-fixture version drift.\n")
        code, result = self.invoke("status", "--settings", "configs/research/local.json")
        self.assertEqual(code, 2)
        self.assertTrue(result["tool_source_changed"])
        self.assertTrue(any("tool source changed" in item for item in result["mismatches"]))
        (output / "frozen/reference_B.json").write_text("{}\n")
        code, result = self.invoke("status", "--settings", "configs/research/local.json")
        self.assertEqual(code, 2)
        self.assertEqual(result["status"], "SNAPSHOT_MISMATCH")


if __name__ == "__main__":
    unittest.main()
