"""Batch transactions and evidence boundaries, without launching model jobs."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from botsort_research import round2
from botsort_research.io import dump_json, load_json, sha256


SOURCE = {"git_commit": "synthetic-test-commit", "working_tree_dirty": False,
          "python_sha256": {"botsort_research/runner.py": "frozen-source"}}


class BatchTests(unittest.TestCase):
    def setUp(self):
        # Other agents may edit unrelated live source during the test. This
        # stable stand-in makes the source-change test intentional and isolated.
        self.source = mock.patch.object(round2, "source_manifest", return_value=copy.deepcopy(SOURCE))
        self.source.start()
        self.addCleanup(self.source.stop)

    def fixture(self, root, features=False):
        cache, images, gt = root / "cache", root / "images", root / "gt"
        cache.mkdir()
        images.mkdir()
        names = ["DJI_9001_1", "DJI_9002_1"]
        for name in names:
            dump_json(cache / (name + ".json"), {"model_sha256": "d" * 64,
                "frames": [{"frame": 1, "detections": []}]})
            path = gt / name / "gt/gt.txt"
            path.parent.mkdir(parents=True)
            path.write_text("")
        split = root / "split.json"
        dump_json(split, {"schema_version": 1, "interpretation": "retrospective_development_only",
                          "calibration": names[:1], "evaluation": names[1:]})
        configs = root / "configs"
        shutil.copytree(round2.CONFIG_ROOT, configs)
        embeddings = None
        if features:
            embeddings = root / "embeddings"
            embeddings.mkdir()
            for name in names:
                # Planning freezes bytes; actual feature loading belongs to a
                # later independently tested CLI job.
                (embeddings / (name + ".npz")).write_bytes(b"plan-fixture-feature-bytes")
                dump_json(embeddings / (name + ".json"), {"fixture": True})
        return dict(cache_dir=cache, image_root=images, gt_root=gt, split_path=split,
                    embeddings_dir=embeddings, config_root=configs)

    def plan(self, root, features=False, doctor_only=False):
        args = self.fixture(root, features=features)
        plan = round2.make_plan(root / "batch", **args)
        if doctor_only:
            plan["jobs"] = [j for j in plan["jobs"] if j["id"] == "doctor"]
            plan["evaluation_runs"] = []
            dump_json(root / "batch/plan.json", plan)
        return plan, args

    @staticmethod
    def successful_job(command, **kwargs):
        if kwargs.get("shell"):
            raise AssertionError("Batch commands must be argument vectors, not shell programs")
        output = Path(command[command.index("--output") + 1])
        if output.suffix == ".json":
            dump_json(output, {"status": "PASS", "fixture": "successful worker"})
        else:
            output.mkdir(parents=True)
            dump_json(output / "artifact.json", {"status": "PASS"})
        return subprocess.CompletedProcess(command, 0)

    def test_plan_closes_phase_dependencies_and_omits_m3_explicitly(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan, _ = self.plan(root)
            jobs = round2._dependency_order(plan["jobs"], {"coverage"})
            identities = [j["id"] for j in jobs]
            self.assertIn("doctor", identities)
            self.assertIn("baseline_compare", identities)
            self.assertIn("baseline_evaluate", identities)
            self.assertIn("baseline_analyze", identities)
            self.assertIn("coverage_calibration", identities)
            self.assertIn("coverage_evaluation", identities)
            self.assertTrue(all(j["phase"] in {"preflight", "coverage"} for j in jobs))
            done = set()
            for job in jobs:
                self.assertTrue(set(job["dependencies"]).issubset(done))
                done.add(job["id"])
            self.assertTrue(plan["disabled"])
            self.assertFalse(any(j["phase"] == "m3" for j in plan["jobs"]))
            with mock.patch.object(round2.subprocess, "run") as worker:
                with self.assertRaisesRegex(ValueError, "omitted M3"):
                    round2.execute_plan(root / "batch/plan.json", phase="m3")
                worker.assert_not_called()

    def test_feature_option_adds_m3_provenance_and_keeps_gt_out_of_replay_commands(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan, args = self.plan(root, features=True)
            self.assertFalse(plan["disabled"])
            identities = [j["id"] for j in round2._dependency_order(plan["jobs"], {"m3"})]
            self.assertLess(identities.index("appearance_native_compare"), identities.index("appearance_calibrate"))
            self.assertLess(identities.index("appearance_calibrate"), identities.index("appearance_experiments"))
            for file in args["embeddings_dir"].iterdir():
                self.assertEqual(plan["input_sha256"][str(file.resolve())], sha256(file))
            for job in plan["jobs"]:
                if job["args"][0] == "run":
                    self.assertNotIn("--gt-root", job["args"])
            self.assertIn("m1_calibration_summary", {j["id"] for j in plan["jobs"]})

    def test_missing_control_config_is_rejected_before_batch_directory_is_created(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self.fixture(root)
            (args["config_root"] / "iou_090.json").unlink()
            with self.assertRaises(FileNotFoundError):
                round2.make_plan(root / "batch", **args)
            self.assertFalse((root / "batch").exists())

    def test_invalid_dependency_graph_cannot_silently_skip_or_duplicate_work(self):
        a = {"id": "a", "phase": "m2", "dependencies": []}
        b = {"id": "b", "phase": "m2", "dependencies": ["a"]}
        for jobs in ([a, a], [b, a], [{**a, "dependencies": ["missing"]}],
                     [{**a, "dependencies": ["b"]}, b]):
            with self.subTest(jobs=jobs), self.assertRaises(ValueError):
                round2._dependency_order(jobs, {"m2"})

    def test_input_snapshot_and_source_guards_reject_changes_before_any_worker(self):
        for changed in ("input", "snapshot", "source"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                plan, _ = self.plan(root, doctor_only=True)
                source_value = copy.deepcopy(SOURCE)
                if changed == "input":
                    filename = next(iter(plan["input_sha256"]))
                    Path(filename).write_bytes(b"changed input bytes")
                elif changed == "snapshot":
                    (root / "batch/config/default.json").write_text("{}")
                else:
                    source_value["python_sha256"]["botsort_research/runner.py"] = "modified-source"
                with mock.patch.object(round2, "source_manifest", return_value=source_value), \
                     mock.patch.object(round2.subprocess, "run") as worker:
                    with self.assertRaises(ValueError):
                        round2.execute_plan(root / "batch/plan.json")
                    worker.assert_not_called()
                self.assertFalse((root / "batch/state.json").exists())

    def test_external_config_changes_do_not_change_frozen_plan_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan, args = self.plan(root, doctor_only=True)
            snapshot = root / "batch/config/default.json"
            original = snapshot.read_bytes()
            (args["config_root"] / "default.json").write_text('{"changed_after_plan": true}')
            with mock.patch.object(round2.subprocess, "run", side_effect=self.successful_job):
                state = round2.execute_plan(root / "batch/plan.json")
            self.assertEqual(state["jobs"]["doctor"]["status"], "COMPLETE")
            self.assertEqual(snapshot.read_bytes(), original)
            self.assertEqual(plan["config_sha256"][str(snapshot.resolve())], sha256(snapshot))

    def test_resume_verifies_receipts_and_rejects_mutated_completed_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.plan(root, doctor_only=True)
            plan = root / "batch/plan.json"
            with mock.patch.object(round2.subprocess, "run", side_effect=self.successful_job) as worker:
                round2.execute_plan(plan)
                self.assertEqual(worker.call_count, 1)
                with self.assertRaises(FileExistsError):
                    round2.execute_plan(plan)
                round2.execute_plan(plan, resume=True)
                self.assertEqual(worker.call_count, 1)
                (root / "batch/doctor.json").write_text('{"changed":true}')
                with self.assertRaisesRegex(ValueError, "Completed job output"):
                    round2.execute_plan(plan, resume=True)
                self.assertEqual(worker.call_count, 1)

    def test_failed_partial_output_is_preserved_then_archived_on_explicit_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.plan(root, doctor_only=True)
            plan = root / "batch/plan.json"
            def failed(command, **kwargs):
                dump_json(root / "batch/doctor.json", {"partial_attempt": 1})
                return subprocess.CompletedProcess(command, 7)
            with mock.patch.object(round2.subprocess, "run", side_effect=failed):
                with self.assertRaisesRegex(RuntimeError, "exit 7"):
                    round2.execute_plan(plan)
            self.assertEqual(load_json(root / "batch/state.json")["jobs"]["doctor"]["status"], "FAILED")
            with mock.patch.object(round2.subprocess, "run", side_effect=self.successful_job) as worker:
                with self.assertRaises(FileExistsError):
                    round2.execute_plan(plan, resume=True)
                worker.assert_not_called()
                self.assertEqual(load_json(root / "batch/doctor.json"), {"partial_attempt": 1})
                round2.execute_plan(plan, resume=True, retry_failed=True)
                self.assertEqual(worker.call_count, 1)
            archived = list((root / "batch/failed").glob("doctor_*/0_doctor.json"))
            self.assertEqual(len(archived), 1)
            self.assertEqual(load_json(archived[0]), {"partial_attempt": 1})
            self.assertEqual(load_json(root / "batch/doctor.json")["status"], "PASS")

    def test_unrecorded_or_missing_outputs_cannot_gain_complete_receipts(self):
        for scenario in ("unrecorded", "worker_wrote_nothing"):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                self.plan(root, doctor_only=True)
                if scenario == "unrecorded":
                    dump_json(root / "batch/doctor.json", {"not_from_executor": True})
                with mock.patch.object(round2.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as worker:
                    with self.assertRaises((FileExistsError, FileNotFoundError, RuntimeError)):
                        round2.execute_plan(root / "batch/plan.json")
                    if scenario == "unrecorded":
                        worker.assert_not_called()
                    else:
                        self.assertEqual(load_json(root / "batch/state.json")["jobs"]["doctor"]["status"], "FAILED")

    @staticmethod
    def evaluated_run(directory, *, variant="iou_relaxed", subset="calibration",
                      hota=30., idf1=40., errors=10, cache_hash="same-cache", sequence=None):
        directory.mkdir(parents=True)
        sequence = sequence or ("cal_sequence" if subset == "calibration" else "eval_sequence")
        experiment = {"status": "COMPLETE", "interpretation": "retrospective_development_only",
            "subset": subset, "split": {"calibration": ["cal_sequence"], "evaluation": ["eval_sequence"]},
            "variants": [variant], "sequences": [sequence], "cmc": "sof", "frames": {sequence: 10},
            "cache_sha256": {sequence: cache_hash}, "image_manifest_sha256": {sequence: "same-images"},
            "timing": [{"variant": variant, "frames": 10, "association_seconds": 1}],
            "config": {"fixture_candidate": directory.name}}
        dump_json(directory / "experiment.json", experiment)
        row = {"tracker": variant, "HOTA": hota, "DetA": 20., "AssA": 45., "IDF1": idf1,
               "MOTA": 25., "FP": errors, "FN": 0, "IDSW": 0}
        summary = {"experiment_manifest_sha256": sha256(directory / "experiment.json"),
            "gt_sha256": {sequence: "same-gt"},
            "results": [{**row, "sequence": "COMBINED_SEQ"}, {**row, "sequence": sequence}]}
        dump_json(directory / "metrics/summary.json", summary)
        return directory

    def test_simple_control_selection_uses_predeclared_calibration_ranking(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidates = [self.evaluated_run(root / "first", idf1=40, errors=1),
                          self.evaluated_run(root / "second", idf1=41, errors=10),
                          self.evaluated_run(root / "third", idf1=41, errors=3),
                          self.evaluated_run(root / "fourth", idf1=41, errors=3)]
            selected = round2.select_control("iou", candidates, root / "selected")
            self.assertEqual(selected["selected_run"], str(candidates[2]))
            frozen = load_json(root / "selected/config.json")
            self.assertEqual(frozen["selection"]["subset"], "calibration")
            self.assertEqual(frozen["fixture_candidate"], "third")

    def test_control_selection_rejects_evaluation_different_inputs_and_stale_metrics(self):
        for scenario in ("evaluation", "different_inputs", "stale_metrics"):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                first = self.evaluated_run(root / "first")
                if scenario == "evaluation":
                    second = self.evaluated_run(root / "second", subset="evaluation")
                else:
                    second = self.evaluated_run(root / "second", cache_hash="changed" if scenario == "different_inputs" else "same-cache")
                    if scenario == "stale_metrics":
                        manifest = load_json(second / "experiment.json")
                        manifest["config"]["modified_after_evaluation"] = True
                        dump_json(second / "experiment.json", manifest)
                with self.assertRaises(ValueError):
                    round2.select_control("iou", [first, second], root / "selected")
                self.assertFalse((root / "selected/selection.json").exists())

    def test_collect_labels_partial_results_and_checks_all_receipts_before_export(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan, _ = self.plan(root)
            batch = root / "batch"
            baseline = self.evaluated_run(batch / "runs/baseline", variant="native_iou", subset="evaluation")
            dump_json(baseline / "analysis/summary.json", {"status": "COMPLETE", "fixture": True})
            data = baseline / "trackers/native_iou/data/eval_sequence.txt"
            data.parent.mkdir(parents=True)
            data.write_text("1,1,0,0,4,4,0.9,-1,-1,-1\n")
            plan["jobs"] = [
                {"id": "baseline", "phase": "preflight", "outputs": [str(baseline)], "dependencies": []},
                {"id": "baseline_evaluate", "phase": "preflight", "outputs": [str(baseline / "metrics")], "dependencies": ["baseline"]},
                {"id": "baseline_analyze", "phase": "preflight", "outputs": [str(baseline / "analysis")], "dependencies": ["baseline_evaluate"]},
                {"id": "coverage_evaluation", "phase": "coverage", "outputs": [str(batch / "coverage_evaluation")], "dependencies": ["baseline_analyze"]},
            ]
            plan["evaluation_runs"] = [str(baseline)]
            plan_path = batch / "plan.json"
            dump_json(plan_path, plan)
            receipts = {job["id"]: {"status": "COMPLETE", "output_sha256": round2._tree_hashes(job["outputs"])}
                        for job in plan["jobs"][:3]}
            dump_json(batch / "state.json", {"plan_sha256": sha256(plan_path), "jobs": receipts})
            # Partial data exists but has no completed receipt; it must not be
            # copied into an apparently authoritative exported evidence folder.
            dump_json(batch / "coverage_evaluation/summary.json", {"status": "RUNNING", "partial": True})
            with self.assertRaisesRegex(ValueError, "Batch incomplete"):
                round2.collect(plan_path, root / "not_allowed")
            self.assertFalse((root / "not_allowed").exists())
            round2.collect(plan_path, root / "export", allow_partial=True)
            execution = load_json(root / "export/execution.json")
            self.assertEqual(execution["status"], "PARTIAL")
            self.assertIn("coverage_evaluation", execution["incomplete_jobs"])
            self.assertFalse((root / "export/coverage_evaluation").exists())
            self.assertTrue((root / "export/comparison/comparison.csv").is_file())
            data.write_text("tampered completed replay\n")
            with self.assertRaises(ValueError):
                round2.collect(plan_path, root / "tampered_export", allow_partial=True)
            self.assertFalse((root / "tampered_export").exists())
            plan["notes"].append("edited after execution")
            dump_json(plan_path, plan)
            with self.assertRaisesRegex(ValueError, "[Pp]lan"):
                round2.collect(plan_path, root / "edited_plan_export", allow_partial=True)


if __name__ == "__main__":
    unittest.main()
