"""Verified CLEAR event tracing must agree with official counts and artifacts."""
import csv
import gzip
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from botsort_research.error_analysis import (ClearMatcher, TRACKEVAL_COMMIT, analyze_run,
                                             clear_iou, _mot_frames)
from botsort_research.io import dump_json, load_json, sha256


class ClearMatchingTests(unittest.TestCase):
    def test_empty_gt_or_predictions_preserve_upstream_continuity(self):
        for empty_ground_truth in (False, True):
            matcher = ClearMatcher()
            matcher.update([1], [11, 12], [[.9, .8]])
            if empty_ground_truth:
                matcher.update([], [11, 12], np.empty((0, 2)))
            else:
                matcher.update([1], [], np.empty((1, 0)))
            matches, switches = matcher.update([1], [11, 12], [[.51, .9]])
            np.testing.assert_array_equal(matches, [[0, 0]])
            self.assertFalse(switches.any())

    def test_nonempty_unmatched_frame_clears_timestep_but_not_id_history(self):
        matcher = ClearMatcher()
        matcher.update([1], [11], [[.9]])
        matcher.update([1], [99], [[.1]])
        matches, switches = matcher.update([1], [11, 12], [[.51, .9]])
        np.testing.assert_array_equal(matches, [[0, 1]])
        np.testing.assert_array_equal(switches, [True])

    def test_clear_continuity_can_prefer_fewer_matches_than_cardinality(self):
        matcher = ClearMatcher()
        matcher.update([1], [10], [[1.]])
        matches, _ = matcher.update([1, 2], [10, 20], [[.5, .9], [.9, 0]])
        np.testing.assert_array_equal(matches, [[0, 0]])

    def test_threshold_and_zero_area_iou_match_upstream(self):
        eps = np.finfo(float).eps
        self.assertEqual(len(ClearMatcher().update([1], [10], [[.5 - eps]])[0]), 1)
        self.assertEqual(len(ClearMatcher().update([1], [10], [[.5 - 2 * eps]])[0]), 0)
        result = clear_iou([[0, 0, 4, 4], [0, 0, 0, 0]], [[0, 0, 4, 4]])
        np.testing.assert_array_equal(result, [[1.], [0.]])

    @unittest.skipUnless(os.environ.get("TRACKEVAL_ROOT"), "Set TRACKEVAL_ROOT to verify against upstream CLEAR")
    def test_random_sequences_match_actual_pinned_clear(self):
        sys.path.insert(0, os.environ["TRACKEVAL_ROOT"])
        from trackeval.metrics.clear import CLEAR
        rng = np.random.default_rng(17)
        for _ in range(12):
            matcher, counts = ClearMatcher(), {"CLR_TP": 0, "CLR_FP": 0, "CLR_FN": 0, "IDSW": 0}
            data = {"gt_ids": [], "tracker_ids": [], "similarity_scores": [],
                    "num_gt_ids": 3, "num_tracker_ids": 5, "num_timesteps": 35}
            for frame in range(data["num_timesteps"]):
                g = np.flatnonzero(rng.random(3) > .3)
                p = np.flatnonzero(rng.random(5) > .3)
                if frame % 9 == 2:
                    g = np.empty(0, dtype=int)
                if frame % 7 == 1:
                    p = np.empty(0, dtype=int)
                similarity = rng.uniform(0, 1, (len(g), len(p)))
                data["gt_ids"].append(g)
                data["tracker_ids"].append(p)
                data["similarity_scores"].append(similarity)
                matches, changes = matcher.update(g, p, similarity)
                counts["CLR_TP"] += len(matches)
                counts["CLR_FP"] += len(p) - len(matches)
                counts["CLR_FN"] += len(g) - len(matches)
                counts["IDSW"] += int(changes.sum())
            data["num_gt_dets"] = sum(map(len, data["gt_ids"]))
            data["num_tracker_dets"] = sum(map(len, data["tracker_ids"]))
            reference = CLEAR({"PRINT_CONFIG": False}).eval_sequence(data)
            for key in counts:
                self.assertEqual(counts[key], reference[key])


class AttributionTests(unittest.TestCase):
    @staticmethod
    def fixture(root):
        run, cache, gt_root = root / "run", root / "cache", root / "gt"
        run.mkdir()
        cache.mkdir()
        sequences, counts = ["sequence_a", "sequence_empty"], {"sequence_a": 4, "sequence_empty": 1}
        variant = "native_iou"
        timing, gt_hashes, cache_hashes = [], {}, {}
        for seq in sequences:
            frames, gt_rows, predictions, traces = [], [], [], []
            for frame in range(1, counts[seq] + 1):
                if seq == "sequence_empty":
                    raw, output_box, tid = [20, 20, 24, 24], [20, 20, 24, 24], 1
                elif frame == 1:
                    raw, output_box, tid = [10, 10, 14, 14], [10, 10, 14, 14], 1
                elif frame == 2:
                    raw, output_box, tid = [11, 10, 15, 14], [9, 10, 13, 14], 1
                elif frame == 3:
                    raw, output_box, tid = [40, 40, 44, 44], [40, 40, 44, 44], 2
                else:
                    raw, output_box, tid = [12, 10, 16, 14], [12, 10, 16, 14], 3
                score = float(np.float32(.9))
                frames.append({"frame": frame, "image": f"test2017/{seq}/00/{frame}.png",
                               "detections": [[*raw, score, 2]]})
                if seq != "sequence_empty" and frame != 3:
                    gt_rows.append(f"{frame},1,{raw[0]},{raw[1]},4,4,1,7,1\n")
                x, y, x2, y2 = output_box
                predictions.append(f"{frame},{tid},{x},{y},{x2-x},{y2-y},{score},-1,-1,-1\n")
                traces.append({"frame": frame, "policy": "kalman", "observations": [
                    {"track_id": tid, "det_index": 0, "score": score, "detection_box": raw,
                     "kalman_box": output_box, "output_box": output_box}]})
            cache_path = cache / f"{seq}.json"
            dump_json(cache_path, {"model_sha256": "a" * 64, "frames": frames})
            gt_path = gt_root / seq / "gt/gt.txt"
            gt_path.parent.mkdir(parents=True)
            gt_path.write_text("".join(gt_rows))
            pred_path = run / "trackers" / variant / "data" / f"{seq}.txt"
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            pred_path.write_text("".join(predictions))
            trace_path = run / "outputs" / variant / f"{seq}.jsonl"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text("".join(json.dumps(event) + "\n" for event in traces))
            cache_hashes[seq], gt_hashes[seq] = sha256(cache_path), sha256(gt_path)
            timing.append({"variant": variant, "sequence": seq, "frames": counts[seq],
                           "predictions_sha256": sha256(pred_path), "output_trace_sha256": sha256(trace_path)})
        dump_json(run / "experiment.json", {"status": "COMPLETE", "output_trace": True,
            "variants": [variant], "sequences": sequences, "frames": counts, "cache_dir": str(cache),
            "cache_sha256": cache_hashes, "timing": timing, "effective_variants": {variant: {"output_policy": "kalman"}}})
        rows = [{"tracker": variant, "sequence": "sequence_a", "FP": 2, "FN": 1, "IDSW": 1},
                {"tracker": variant, "sequence": "sequence_empty", "FP": 1, "FN": 0, "IDSW": 0},
                {"tracker": variant, "sequence": "COMBINED_SEQ", "FP": 3, "FN": 1, "IDSW": 1}]
        dump_json(run / "metrics/summary.json", {"status": "PASS", "results": rows,
            "experiment_manifest_sha256": sha256(run / "experiment.json"), "gt_sha256": gt_hashes,
            "trackeval_provenance": {"commit": TRACKEVAL_COMMIT}})
        return run, cache, gt_root

    @staticmethod
    def rows(path):
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    def test_exact_metrics_verified_and_drift_is_a_diagnostic_cohort(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run, _, gt = self.fixture(root)
            result = analyze_run(run, gt, root / "analysis")
            self.assertEqual(result["status"], "COMPLETE")
            self.assertEqual(result["combined"][0]["FP"], 3)
            self.assertEqual(result["combined"][0]["FN"], 1)
            self.assertEqual(result["combined"][0]["IDSW"], 1)
            fp = self.rows(root / "analysis/fp_cases.csv.gz")
            drift = next(row for row in fp if row["cohort"] == "localization_drift_candidate")
            self.assertEqual(float(drift["center_shift_px"]), 2.)
            self.assertEqual(float(drift["detection_max_iou"]), 1.)
            self.assertLess(float(drift["output_max_iou"]), .5)
            gt_rows = self.rows(root / "analysis/gt_frames.csv.gz")
            self.assertEqual(len(gt_rows), 3)
            self.assertEqual(sum(int(row["matched"]) for row in gt_rows), 2)
            self.assertEqual(gt_rows[-1]["missing_frames_before_match"], "2")
            self.assertEqual(gt_rows[-1]["id_changed"], "1")
            self.assertEqual(len(self.rows(root / "analysis/frame_errors.csv")), 5)
            self.assertEqual(len(self.rows(root / "analysis/case_samples.csv")), 3)
            self.assertIn("First 3", result["case_sampling"])

    def test_per_sequence_disagreement_marks_failure_not_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run, _, gt = self.fixture(root)
            path = run / "metrics/summary.json"
            metrics = load_json(path)
            metrics["results"][0]["FP"] = 0
            dump_json(path, metrics)
            with self.assertRaisesRegex(ValueError, "differs from official"):
                analyze_run(run, gt, root / "analysis")
            self.assertEqual(load_json(root / "analysis/summary.json")["status"], "FAILED")

    def test_changed_inputs_are_rejected_before_analysis(self):
        for change in ("cache", "gt", "prediction", "trace", "manifest"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                run, cache, gt = self.fixture(root)
                paths = {"cache": cache / "sequence_a.json", "gt": gt / "sequence_a/gt/gt.txt",
                         "prediction": run / "trackers/native_iou/data/sequence_a.txt",
                         "trace": run / "outputs/native_iou/sequence_a.jsonl", "manifest": run / "experiment.json"}
                with paths[change].open("a") as stream:
                    stream.write("\n")
                with self.assertRaisesRegex(ValueError, "Missing or changed"):
                    analyze_run(run, gt, root / "analysis")
                self.assertFalse((root / "analysis").exists())

    def test_rehashed_but_inconsistent_trace_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run, _, gt = self.fixture(root)
            path = run / "outputs/native_iou/sequence_a.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[1]["observations"][0]["detection_box"] = [30, 30, 34, 34]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            manifest = load_json(run / "experiment.json")
            manifest["timing"][0]["output_trace_sha256"] = sha256(path)
            dump_json(run / "experiment.json", manifest)
            metrics = load_json(run / "metrics/summary.json")
            metrics["experiment_manifest_sha256"] = sha256(run / "experiment.json")
            dump_json(run / "metrics/summary.json", metrics)
            with self.assertRaisesRegex(ValueError, "does not agree"):
                analyze_run(run, gt, root / "analysis")
            self.assertEqual(load_json(root / "analysis/summary.json")["status"], "FAILED")

    def test_clear_loader_retains_nonzero_integer_marks_and_pools_classes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gt.txt"
            path.write_text("1,1,0,0,4,4,-1,7,1\n1,2,10,10,4,4,0,3,1\n1,3,20,20,4,4,.5,2,1\n")
            data = _mot_frames(path, 1, ground_truth=True)
            np.testing.assert_array_equal(data[1][:, 4], [1])

    def test_low_candidate_boundary_uses_runtime_score_promotion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run, cache, gt = self.fixture(root)
            score = float(np.float32(.1))
            payload = load_json(cache / "sequence_a.json")
            payload["frames"][1]["detections"][0][4] = .1
            dump_json(cache / "sequence_a.json", payload)
            trace_path = run / "outputs/native_iou/sequence_a.jsonl"
            traces = [json.loads(line) for line in trace_path.read_text().splitlines()]
            traces[1]["observations"][0]["score"] = score
            trace_path.write_text("".join(json.dumps(row) + "\n" for row in traces))
            pred_path = run / "trackers/native_iou/data/sequence_a.txt"
            rows = [row.split(",") for row in pred_path.read_text().splitlines()]
            rows[1][6] = str(score)
            pred_path.write_text("".join(",".join(row) + "\n" for row in rows))
            manifest = load_json(run / "experiment.json")
            manifest["cache_sha256"]["sequence_a"] = sha256(cache / "sequence_a.json")
            manifest["timing"][0].update(predictions_sha256=sha256(pred_path), output_trace_sha256=sha256(trace_path))
            dump_json(run / "experiment.json", manifest)
            metrics = load_json(run / "metrics/summary.json")
            metrics["experiment_manifest_sha256"] = sha256(run / "experiment.json")
            dump_json(run / "metrics/summary.json", metrics)
            analyze_run(run, gt, root / "analysis")
            second = next(row for row in self.rows(root / "analysis/gt_frames.csv.gz") if row["frame"] == "2")
            self.assertEqual(second["has_low_candidate"], "True")
            fp = next(row for row in self.rows(root / "analysis/fp_cases.csv.gz") if row["frame"] == "2")
            self.assertEqual(fp["score_bin"], "low")


if __name__ == "__main__":
    unittest.main()
