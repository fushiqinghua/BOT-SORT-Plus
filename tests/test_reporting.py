"""Reporting must compare equal experimental inputs before computing deltas."""
import tempfile
import unittest
from pathlib import Path

from botsort_research.io import dump_json, load_json, sha256
from botsort_research.reporting import summarize_runs


class ReportingTests(unittest.TestCase):
    def make_run(self, root, name, cmc="sof", tracker="native_iou", hota=40.,
                 fn=100, sequences=("sequence_a",), cache_suffix="", gt_suffix="",
                 frame_count=10, status="COMPLETE"):
        path = Path(root) / name
        path.mkdir()
        experiment = {
            "status": status, "cmc": cmc, "sequences": list(sequences), "variants": [tracker],
            "cache_sha256": {s: "detection-" + s + cache_suffix for s in sequences},
            "frames": {s: frame_count for s in sequences},
            "image_manifest_sha256": {s: "images-" + s for s in sequences},
            "timing": [{"variant": tracker, "sequence": s, "frames": frame_count,
                        "association_seconds": 2.} for s in sequences],
        }
        dump_json(path / "experiment.json", experiment)
        row = {"tracker": tracker, "sequence": "COMBINED_SEQ", "HOTA": hota,
               "DetA": 25., "AssA": 60., "IDF1": 45., "MOTA": 28.,
               "FP": 20, "FN": fn, "IDSW": 5}
        # An unrelated per-sequence row must never be mistaken for the combined result.
        per_sequence = {**row, "sequence": sequences[0], "HOTA": 99.}
        summary = {
            "status": "PASS", "experiment_manifest_sha256": sha256(path / "experiment.json"),
            "gt_sha256": {s: "gt-" + s + gt_suffix for s in sequences},
            "results": [per_sequence, row],
        }
        dump_json(path / "metrics" / "summary.json", summary)
        return path

    def test_equal_inputs_use_sof_native_for_deltas_not_first_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            m2 = self.make_run(root, "m2", tracker="m2_nll_floor", hota=42., fn=90)
            off = self.make_run(root, "native_off", cmc="none", hota=50.)
            native = self.make_run(root, "native_sof", hota=40.)
            result = summarize_runs([m2, off, native], root / "comparison")
            rows = {row["run"]: row for row in result["results"]}
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows["m2"]["delta_HOTA"], 2.)
            self.assertEqual(rows["m2"]["delta_FN"], -10)
            self.assertEqual(rows["native_off"]["delta_HOTA"], 10.)
            self.assertEqual(rows["native_sof"]["delta_HOTA"], 0.)
            self.assertEqual(rows["m2"]["association_fps"], 5.)
            for name in ("comparison.csv", "comparison.md", "comparison.json"):
                self.assertTrue((root / "comparison" / name).is_file())

    def test_different_caches_gt_frames_or_selected_sets_are_refused(self):
        changes = (
            {"cache_suffix": "changed"},
            {"gt_suffix": "changed"},
            {"frame_count": 11},
            {"sequences": ("sequence_b",)},
        )
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                first = self.make_run(root, "first")
                second = self.make_run(root, "second", **change)
                with self.assertRaisesRegex(ValueError, "differ"):
                    summarize_runs([first, second], root / "comparison")
                self.assertFalse((root / "comparison").exists())

    def test_metrics_stamped_for_another_experiment_are_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = self.make_run(root, "run")
            summary_path = run / "metrics" / "summary.json"
            summary = load_json(summary_path)
            summary["experiment_manifest_sha256"] = "stale-manifest"
            dump_json(summary_path, summary)
            with self.assertRaisesRegex(ValueError, "manifest"):
                summarize_runs([run], root / "comparison")

    def test_missing_gt_hashes_cannot_be_treated_as_equal_unknown_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = [self.make_run(root, name) for name in ("first", "second")]
            for run in runs:
                path = run / "metrics" / "summary.json"
                summary = load_json(path)
                del summary["gt_sha256"]
                dump_json(path, summary)
            with self.assertRaises(ValueError):
                summarize_runs(runs, root / "comparison")

    def test_changed_images_are_refused_even_with_identical_detections_and_gt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self.make_run(root, "first")
            second = self.make_run(root, "second")
            path = second / "experiment.json"
            experiment = load_json(path)
            experiment["image_manifest_sha256"]["sequence_a"] = "different-images"
            dump_json(path, experiment)
            summary_path = second / "metrics" / "summary.json"
            summary = load_json(summary_path)
            summary["experiment_manifest_sha256"] = sha256(path)
            dump_json(summary_path, summary)
            with self.assertRaises(ValueError):
                summarize_runs([first, second], root / "comparison")

    def test_incomplete_run_cannot_enter_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = self.make_run(root, "run", status="RUNNING")
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                summarize_runs([run], root / "comparison")

    def test_without_sof_native_no_implicit_reference_is_used(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            off = self.make_run(root, "native_off", cmc="none")
            m2 = self.make_run(root, "m2", tracker="m2_nll_floor", hota=42.)
            result = summarize_runs([off, m2], root / "comparison")
            self.assertTrue(all(row["delta_HOTA"] is None and row["delta_FN"] is None
                                for row in result["results"]))

    def test_existing_comparison_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = self.make_run(root, "run")
            destination = root / "comparison"
            destination.mkdir()
            with self.assertRaises(FileExistsError):
                summarize_runs([run], destination)


if __name__ == "__main__":
    unittest.main()
