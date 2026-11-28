"""R4.2 provenance, reference import and batch-boundary tests; synthetic only."""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_wem_r41_batch import Fixture as R41Fixture
from tools.research.wem_r41 import assets as r41
from tools.research.wem_r42 import assets, batch

PROTOCOL = assets.read(assets.ROOT / assets.PROTOCOL_PATH)


class Fixture(R41Fixture):
    def __init__(self, root):
        super().__init__(root)
        self.put(assets.PROTOCOL_PATH, PROTOCOL)
        for relative in ("tools/run_wem_r42.py", "tools/research/wem_r42/fixture.py"):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# SYNTHETIC SOURCE\n")
        empty = hashlib.sha256(b"").hexdigest()
        refs, parity = r41.read(root / "references.json"), r41.read(root / "parity.json")
        inputs = r41.read(root / "evaluation.json")
        for role, method in r41.METHODS.items():
            relative = f"references/{role}/protocols/{method}.json"
            protocol = r41.read(root / relative)
            protocol.update(encoder_contracts={},
                detection_fingerprints={seq: "synthetic_detection_" + seq for seq in self.lengths},
                feature_hashes=inputs["features"],
                decoded_image_hashes={seq: "synthetic_image_" + seq for seq in self.lengths})
            self.put(relative, protocol)
            refs["references"][role]["protocol_sha256"] = r41.sha(root / relative)
            refs["references"][role]["trajectory_hashes"] = {seq: empty for seq in self.lengths}
            parity["references"][role]["files"] = {seq: dict(mot_sha256=empty, events_sha256=empty,
                                                               state_chain_sha256=empty) for seq in self.lengths}
        self.put("references.json", refs)
        self.put("parity.json", parity)
        old = super().prepare()
        self.source = root / "run/full"
        self.put("run/full/prediction_bundle.json", old["bundle"])
        names, fingerprint = list(self.lengths), old["fingerprint"]
        all_keys = [f"{role}_{mode}" for role, mode in r41.FULL_ARMS]
        verified = dict(status="PASS", fingerprint=fingerprint, scope=names, scope_frames=12590,
                        arms={key: {} for key in all_keys})
        for mode in ("off", "W0"):
            key, method = f"BG_{mode}", assets.METHODS["BG"]
            folder = self.source / key
            rows = {}
            protocol = deepcopy(old["references"]["BG"]["protocol"])
            protocol["output_prediction_hashes"] = {seq: empty for seq in names}
            protocol["joint_batch"] = {"camera_manifests": {}}
            protocol["wem_r41"] = dict(role="BG", mode=mode, experiment=old["bundle"]["protocol"],
                code_sha256=old["bundle"]["code_sha256"],
                prediction_bundle_sha256=r41.sha(self.source / "prediction_bundle.json"))
            for seq, n in self.lengths.items():
                for kind, suffix in (("trackers", ".txt"), ("events", ".jsonl"), ("states", ".sha256"),
                                     ("wem_events", ".jsonl"), ("wem_scores", ".jsonl.gz"),
                                     ("wem_decisions", ".jsonl.gz"), ("wem_frame_audit", ".jsonl.gz")):
                    folder_path = folder / kind / method
                    if kind == "trackers":
                        folder_path /= "data"
                    folder_path.mkdir(parents=True, exist_ok=True)
                    (folder_path / (seq + suffix)).write_bytes(b"")
                r41.write(folder / "capture_receipts" / method / (seq + ".json"), {"counts": {"frames": n}})
                r41.write(folder / "timing" / method / (seq + ".json"),
                          dict(association_and_gmc_seconds=1, replay_wall_seconds=2))
                camera = old["bundle"]["settings"]["warp_sources"]["adaptive_gmc"]["sha256"][seq]
                protocol["joint_batch"]["camera_manifests"][seq] = dict(mode="cached", frames=n,
                    source=dict(sha256=camera, decoded_image_sha256=protocol["decoded_image_hashes"][seq]))
                rows[seq] = dict(mot=empty, native_events=empty, states=empty)
            r41.write(folder / "protocols" / (method + ".json"), protocol)
            r41.write(folder / "arm_complete.json", dict(status="PREDICTION_COMPLETE", arm=key,
                fingerprint=fingerprint, sequences=names, files=batch.file_manifest(folder)))
            verified["arms"][key] = dict(per_sequence=rows)
        self.put("run/full/verified.json", verified)
        self.put("run/full/plan.json", dict(fingerprint=fingerprint, sequences=names, arms=all_keys))
        self.put("run/full/report/summary.json", dict(status="FULL_EVALUATION_COMPLETE", fixture=True))
        self.put("run/full/complete.json", dict(status="FULL_BENCHMARK_COMPLETE", sequence_count=30,
            frames_per_arm=12590, arm_count=10, fingerprint=fingerprint,
            checks=r41.sha(self.source / "verified.json"), report=r41.sha(self.source / "report/summary.json")))
        self.local = root / "r42.json"
        self.put("r42.json", dict(schema=assets.SCHEMA, r41_settings=str(self.settings), r41_batch=str(self.source),
            reference_mode="reuse", output_root=str(root / "r42run"), runtime={"workers": 2}))

    def context(self):
        with patch.object(r41, "ROOT", self.root), patch.object(assets, "ROOT", self.root):
            return assets.prepare(assets.load(self.local))


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = Fixture(Path(self.temp.name))

    def test_reuse_locks_inputs_code_scope_and_excludes_gt_from_prediction(self):
        context = self.fixture.context()
        self.assertEqual(set(context["reuse"]), {"BG_off", "BG_W0"})
        self.assertEqual(sum(context["bundle"]["sequences"].values()), 12590)
        self.assertEqual(context["settings"]["runtime"]["workers"], 2)
        self.assertNotIn(str(self.fixture.d / "gt"), json.dumps(context["bundle"]))
        self.assertNotIn("tools/run_wem_r42.py", context["r41_code_sha256"])
        self.assertIn("tools/run_wem_r42.py", context["bundle"]["code_sha256"])
        self.assertEqual(context["receipt"]["status"], "INPUTS_VERIFIED_NOT_RUN")

    def test_reference_changes_and_unreported_source_changes_are_rejected(self):
        method = assets.METHODS["BG"]
        seq = next(iter(self.fixture.lengths))
        paths = [self.fixture.source / "BG_W0/trackers" / method / "data" / (seq + ".txt"),
                 self.fixture.source / "report/summary.json",
                 self.fixture.root / "tools/run_wem_r41.py"]
        for path in paths:
            old = path.read_bytes()
            with self.subTest(path=path):
                path.write_bytes(old + b"changed")
                try:
                    with self.assertRaises(ValueError):
                        self.fixture.context()
                finally:
                    path.write_bytes(old)

    def test_declared_parameters_cannot_drift_from_fixed_code(self):
        for key, bad in (("native_low_control", .04), ("top_geometry_groups", 10)):
            changed = deepcopy(PROTOCOL)
            changed["parameters"][key] = bad
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Fixed R4.2"):
                assets.validate_protocol(changed)
        changed = deepcopy(PROTOCOL)
        changed["controls"]["scientific_acceptance_gate"] = True
        with self.assertRaisesRegex(ValueError, "Fixed R4.2"):
            assets.validate_protocol(changed)

    def test_rerun_does_not_require_old_full_directory(self):
        self.fixture.update("r42.json", lambda x: x.update(reference_mode="rerun", r41_batch=None))
        context = self.fixture.context()
        self.assertEqual(context["reuse"], {})

    def test_output_cannot_overwrite_the_earlier_run(self):
        self.fixture.update("r42.json", lambda x: x.update(output_root=str(self.fixture.root / "run/accidental")))
        with self.assertRaisesRegex(ValueError, "separate"):
            self.fixture.context()

    def test_init_inherits_paths_and_will_not_replace_existing_settings(self):
        path = self.fixture.root / "new.json"
        with patch.object(r41, "ROOT", self.fixture.root):
            data = assets.init(path, self.fixture.settings)
        self.assertEqual(data["r41_batch"], str(self.fixture.source))
        self.assertEqual(data["reference_mode"], "reuse")
        self.assertEqual(data["runtime"], {})
        with self.assertRaises(FileExistsError):
            assets.init(path, self.fixture.settings)

    def test_reference_import_copies_and_preserves_source(self):
        context = self.fixture.context()
        destination = self.fixture.root / "r42run/full"
        destination.mkdir(parents=True)
        original = {p.relative_to(self.fixture.source).as_posix(): r41.sha(p)
                    for p in self.fixture.source.rglob("*") if p.is_file()}
        names = list(self.fixture.lengths)
        batch.import_reference(context, destination, "W0", names)
        batch.import_reference(context, destination, "W0", names)
        self.assertTrue(batch.completed_arm(destination / "BG_W0", "BG_W0", context["fingerprint"], names))
        marker = r41.read(destination / "BG_W0/arm_complete.json")
        self.assertEqual(marker["provenance"]["kind"], "reused_r41")
        self.assertEqual(original, {p.relative_to(self.fixture.source).as_posix(): r41.sha(p)
                                   for p in self.fixture.source.rglob("*") if p.is_file()})


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.context = dict(r42_settings={"output_root": str(self.root)}, fingerprint="synthetic",
            settings={"runtime": {"workers": 1}}, receipt={},
            bundle={"protocol": {}, "sequences": r41.read(r41.ROOT / "configs/research/wem_r4_1_protocol.json")["sequences"]},
            reuse={"BG_off": {}, "BG_W0": {}})

    def test_full_imports_two_references_and_runs_exactly_three_new_full_arms(self):
        with patch.object(batch, "import_reference") as imported, patch.object(batch, "track") as tracked, \
             patch.object(batch, "inspect"):
            batch.stage(self.context, "full")
        self.assertEqual([call.args[2] for call in imported.call_args_list], ["off", "W0"])
        self.assertEqual({call.args[2] for call in tracked.call_args_list}, set(assets.MODES[2:]))
        for call in tracked.call_args_list:
            self.assertEqual(len(call.args[3]), 30)
        plan = r41.read(self.root / "full/plan.json")
        self.assertEqual(plan["frames"], 12590)
        self.assertFalse(plan["scientific_acceptance_gate"])

    def test_smoke_executes_all_five_modes_and_never_imports_reference_outputs(self):
        with patch.object(batch, "import_reference") as imported, patch.object(batch, "track") as tracked, \
             patch.object(batch, "inspect") as inspect:
            def verified(context, folder, names):
                r41.write(folder / "verified.json", {"synthetic": True})
            inspect.side_effect = verified
            batch.stage(self.context, "smoke")
        imported.assert_not_called()
        self.assertEqual({call.args[2] for call in tracked.call_args_list}, set(assets.MODES))
        self.assertTrue(all(call.args[3] == list(assets.SMOKE) for call in tracked.call_args_list))

    def test_full_reports_even_when_no_intervention_or_negative_metrics(self):
        order = []
        with patch.object(batch, "load", return_value={}), patch.object(batch, "prepare", return_value=self.context), \
             patch.object(batch, "stage", side_effect=lambda _, name: order.append(name) or self.root / name), \
             patch.object(batch, "evaluate", side_effect=lambda *_: order.append("evaluate")), \
             patch.object(batch, "summarize", side_effect=lambda *_: order.append("summarize") or {"delta_HOTA": -1}):
            self.assertEqual(batch.main(["full", "--settings", "not_read.json"]), 0)
        self.assertEqual(order, ["smoke", "full", "evaluate", "summarize"])

    def test_technical_failure_prevents_uninterpretable_full_batch(self):
        with patch.object(batch, "load", return_value={}), patch.object(batch, "prepare", return_value=self.context), \
             patch.object(batch, "stage", side_effect=ValueError("state parity differs")), patch.object(batch, "evaluate") as evaluate:
            with self.assertRaisesRegex(ValueError, "state parity"):
                batch.main(["full", "--settings", "not_read.json"])
        evaluate.assert_not_called()

    def test_init_flags_are_not_silently_ignored_by_full(self):
        with self.assertRaisesRegex(ValueError, "apply to init"):
            batch.main(["full", "--settings", "not_read.json", "--reference-mode", "rerun"])


if __name__ == "__main__":
    unittest.main()
