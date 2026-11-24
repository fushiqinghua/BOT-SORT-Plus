"""Evaluation contracts with tiny real files; no fabricated MOT accuracy runs."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


_PATH = Path(__file__).resolve().parents[2] / "tools" / "evaluate_botsort_plus.py"
_SPEC = importlib.util.spec_from_file_location("evaluate_botsort_plus_contract", _PATH)
evaluation = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(evaluation)


class EvaluationContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.sequences = {"seq_a": 5, "seq_empty": 3}

    def tearDown(self):
        self.tmp.cleanup()

    def make_run(self, name="baseline", **changes):
        data = self.root / "trackers" / name / "data"
        data.mkdir(parents=True)
        for sequence in self.sequences:
            text = "1,1,10,10,4,4,0.3,-1,-1,-1\n3,1,10,10,4,4,0.3,-1,-1,-1\n" if sequence == "seq_a" else ""
            (data / f"{sequence}.txt").write_text(text)
        protocol = {"schema": "botsort_plus_run_v1", "status": "COMPLETE", "config": {"name": name},
                    "native_config": {"with_reid": True}, "sequences": self.sequences,
                    "detector_sha256": "a" * 64, "rgb_encoder_sha256": "b" * 64,
                    "source_protocol_sha256": "c" * 64, "subset": True,
                    "detection_fingerprints": {seq: "d" * 64 for seq in self.sequences},
                    "feature_hashes": {seq: {"features": "e" * 64, "metadata": "f" * 64} for seq in self.sequences},
                    "encoder_contracts": {"rgb": {"sha256": "b" * 64, "profile": "baseline_rgb"}},
                    "output_prediction_hashes": {seq: evaluation.sha256(data / f"{seq}.txt") for seq in self.sequences}}
        protocol.update(changes)
        path = self.root / "protocols" / f"{name}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(protocol))
        return protocol

    def summary(self, names):
        rows = []
        for index, name in enumerate(names):
            for sequence in [*self.sequences, "COMBINED_SEQ"]:
                rows.append({"tracker": name, "sequence": sequence,
                             **{key: 30.0 + index for key in evaluation.METRICS if key not in evaluation.COUNTS},
                             "FP": 10 - index, "FN": 20 - index, "IDSW": 3 - index})
        return {"status": "PASS", "modality_reference": "rgb", "split": "test", "results": rows}

    def test_valid_runs_include_empty_prediction_sequence_and_empty_tail(self):
        self.make_run()
        self.make_run("a_b")
        result = evaluation.validate_runs(self.root, ["baseline", "a_b"])
        self.assertEqual(result["sequences"], self.sequences)
        self.assertEqual(result["prediction_files"]["baseline"]["seq_empty"]["rows"], 0)
        self.assertEqual(result["prediction_files"]["baseline"]["seq_a"]["last_output_frame"], 3)

    def test_incomplete_and_wrong_schema_rejected(self):
        self.make_run(status="RUNNING")
        with self.assertRaises(ValueError):
            evaluation.validate_runs(self.root, ["baseline"])

    def test_stale_predictions_are_rejected(self):
        self.make_run()
        path = self.root / "trackers" / "baseline" / "data" / "seq_a.txt"
        path.write_text(path.read_text().replace("0.3", "0.4"))
        with self.assertRaisesRegex(ValueError, "SHA256"):
            evaluation.validate_runs(self.root, ["baseline"])

    def test_extra_or_missing_sequence_file_rejected(self):
        self.make_run()
        path = self.root / "trackers" / "baseline" / "data" / "stale.txt"
        path.write_text("")
        with self.assertRaises(ValueError):
            evaluation.validate_runs(self.root, ["baseline"])

    def test_mixed_feature_or_detector_identity_rejected(self):
        self.make_run()
        self.make_run("other", rgb_encoder_sha256="d" * 64)
        with self.assertRaisesRegex(ValueError, "rgb_encoder_sha256"):
            evaluation.validate_runs(self.root, ["baseline", "other"])

    def test_mixed_sequence_scope_rejected(self):
        self.make_run()
        self.make_run("other", sequences={"seq_a": 5})
        with self.assertRaises(ValueError):
            evaluation.validate_runs(self.root, ["baseline", "other"])

    def test_same_weights_different_detection_content_rejected(self):
        self.make_run()
        self.make_run("other", detection_fingerprints={seq: "e" * 64 for seq in self.sequences})
        with self.assertRaisesRegex(ValueError, "detection content/NMS"):
            evaluation.validate_runs(self.root, ["baseline", "other"])

    def test_same_rgb_weights_different_preprocessing_rejected(self):
        self.make_run()
        self.make_run("other", encoder_contracts={"rgb": {"sha256": "b" * 64, "profile": "different_crop"}})
        with self.assertRaisesRegex(ValueError, "rgb encoder/preprocessing"):
            evaluation.validate_runs(self.root, ["baseline", "other"])

    def test_rgb_only_and_rgb_thermal_runs_allowed_but_shared_thermal_contract_fixed(self):
        self.make_run()
        rgb = {"sha256": "b" * 64, "profile": "baseline_rgb"}
        thermal = {"sha256": "8" * 64, "profile": "thermal_v1"}
        self.make_run("abc", encoder_contracts={"rgb": rgb, "thermal": thermal})
        result = evaluation.validate_runs(self.root, ["baseline", "abc"])
        self.assertEqual(set(result["encoder_contracts"]), {"rgb", "thermal"})
        self.make_run("abc_other", encoder_contracts={"rgb": rgb, "thermal": {**thermal, "profile": "changed"}})
        with self.assertRaisesRegex(ValueError, "thermal encoder/preprocessing"):
            evaluation.validate_runs(self.root, ["baseline", "abc", "abc_other"])

    def test_trackeval_allows_only_mechanical_line_ending_difference(self):
        trackeval = self.root / "trackeval_source"
        (trackeval / "trackeval").mkdir(parents=True)
        source = trackeval / "trackeval" / "__init__.py"
        source.write_bytes(b"VALUE = 1\nSECOND = 2\n")
        original = source.read_bytes()
        digest = evaluation.hashlib.sha256(original.replace(b"\n", b"\r\n")).hexdigest()
        manifest = self.root / "provenance" / "trackeval.json"
        manifest.parent.mkdir()
        manifest.write_text(json.dumps({"commit": "test", "package_sha256": {"trackeval/__init__.py": digest}}))
        result = evaluation.verify_trackeval(trackeval, self.root)
        self.assertTrue(result["eol_normalized_matched"])
        self.assertEqual(result["source_files"]["trackeval/__init__.py"]["match"], "crlf_normalized")
        self.assertEqual(source.read_bytes(), original)
        source.write_bytes(b"VALUE = 3\nSECOND = 2\n")
        with self.assertRaisesRegex(ValueError, "differs"):
            evaluation.verify_trackeval(trackeval, self.root)

    def test_claiming_full_test_with_two_sequences_rejected(self):
        self.make_run(subset=False)
        with self.assertRaisesRegex(ValueError, "30 sequences"):
            evaluation.validate_runs(self.root, ["baseline"])

    def test_name_mismatch_and_path_traversal_rejected(self):
        self.make_run(config={"name": "different"})
        with self.assertRaises(ValueError):
            evaluation.validate_runs(self.root, ["baseline"])
        with self.assertRaises(ValueError):
            evaluation.validate_runs(self.root, ["../baseline"])

    def test_duplicate_output_id_rejected_even_if_hash_matches(self):
        protocol = self.make_run()
        path = self.root / "trackers" / "baseline" / "data" / "seq_a.txt"
        path.write_text("1,1,10,10,4,4,.3,-1,-1,-1\n1,1,20,20,4,4,.3,-1,-1,-1\n")
        protocol["output_prediction_hashes"]["seq_a"] = evaluation.sha256(path)
        (self.root / "protocols" / "baseline.json").write_text(json.dumps(protocol))
        with self.assertRaisesRegex(ValueError, "duplicate ID"):
            evaluation.validate_runs(self.root, ["baseline"])

    def test_frame_outside_declared_video_rejected(self):
        protocol = self.make_run()
        path = self.root / "trackers" / "baseline" / "data" / "seq_a.txt"
        path.write_text("6,1,10,10,4,4,.3,-1,-1,-1\n")
        protocol["output_prediction_hashes"]["seq_a"] = evaluation.sha256(path)
        (self.root / "protocols" / "baseline.json").write_text(json.dumps(protocol))
        with self.assertRaisesRegex(ValueError, "outside"):
            evaluation.validate_runs(self.root, ["baseline"])

    def test_gt_all_sequences_required_empty_gt_allowed(self):
        for seq in self.sequences:
            path = self.root / "gt" / seq / "gt" / "gt.txt"
            path.parent.mkdir(parents=True)
            path.write_text("")
        result = evaluation.validate_gt(self.root / "gt", self.sequences)
        self.assertEqual(set(result["sha256"]), set(self.sequences))
        (self.root / "gt" / "seq_empty" / "gt" / "gt.txt").unlink()
        with self.assertRaises(FileNotFoundError):
            evaluation.validate_gt(self.root / "gt", self.sequences)

    def test_same_run_baseline_deltas_use_combined_row(self):
        self.make_run()
        self.make_run("a_b")
        validation = evaluation.validate_runs(self.root, ["baseline", "a_b"])
        report = evaluation.build_comparison(self.summary(["baseline", "a_b"]), validation)
        self.assertEqual(report["reference"]["kind"], "same_run_replay")
        self.assertEqual(report["results"][1]["delta"]["HOTA"], 1.0)
        self.assertEqual(report["results"][1]["delta"]["FP"], -1)
        evaluation.write_comparison(self.root / "report", report)
        self.assertTrue((self.root / "report" / "comparison.csv").is_file())
        self.assertIn("序列子集", (self.root / "report" / "comparison.md").read_text())

    def test_historical_full_metrics_never_compared_to_subset(self):
        self.make_run("a_b")
        validation = evaluation.validate_runs(self.root, ["a_b"])
        old = self.root / "old.json"
        old.write_text(json.dumps(self.summary(["botsort_reid"])))
        report = evaluation.build_comparison(self.summary(["a_b"]), validation,
                                            "botsort_reid", old, "historical runtime")
        self.assertEqual(report["reference"]["status"], "unavailable")
        self.assertNotIn("delta", report["results"][0])
        self.assertIn("子集", report["reference"]["reason"])

    def test_missing_reference_has_no_invented_delta(self):
        self.make_run("a_b")
        validation = evaluation.validate_runs(self.root, ["a_b"])
        report = evaluation.build_comparison(self.summary(["a_b"]), validation)
        self.assertNotIn("delta", report["results"][0])
        self.assertEqual(report["reference"]["status"], "unavailable")

    def test_historical_reference_requires_full_matching_protocol_and_context(self):
        self.sequences = {f"seq_{index:02d}": 5 for index in range(30)}
        historical_dir = self.root / "historical"
        (historical_dir / "metrics").mkdir(parents=True)
        historical_protocol = historical_dir / "protocol.json"
        original = {"sequences": self.sequences, "weight_sha256": "a" * 64,
                    "reid_sha256": "b" * 64}
        historical_protocol.write_text(json.dumps(original))
        source_sha = evaluation.sha256(historical_protocol)
        self.make_run("a_b", subset=False, source_protocol_sha256=source_sha)
        validation = evaluation.validate_runs(self.root, ["a_b"])
        reference = historical_dir / "metrics" / "summary.json"
        reference.write_text(json.dumps(self.summary(["botsort_reid"])))
        report = evaluation.build_comparison(self.summary(["a_b"]), validation,
                                            "botsort_reid", reference)
        self.assertEqual(report["reference"]["status"], "unavailable")
        self.assertIn("reference-context", report["reference"]["reason"])
        report = evaluation.build_comparison(self.summary(["a_b"]), validation,
                                            "botsort_reid", reference, "historical runtime")
        self.assertEqual(report["reference"]["status"], "comparable")
        self.assertEqual(report["reference"]["kind"], "historical_runtime")
        self.assertIn("delta", report["results"][0])
        original["weight_sha256"] = "d" * 64
        historical_protocol.write_text(json.dumps(original))
        report = evaluation.build_comparison(self.summary(["a_b"]), validation,
                                            "botsort_reid", reference, "historical runtime")
        self.assertEqual(report["reference"]["status"], "unavailable")
        self.assertNotIn("delta", report["results"][0])


if __name__ == "__main__":
    unittest.main()
