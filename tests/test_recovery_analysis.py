"""Offline accepted-recovery attribution must not count unknowns as wrong IDs."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from botsort_research.error_analysis import TRACKEVAL_COMMIT, analyze_run, classify_recovery
from botsort_research.io import dump_json, load_json, sha256


def accepted(track_id=1, det_index=0, gap=1, age=2):
    return {"track_id": track_id, "det_index": det_index, "feature_gap": gap,
            "lost_age": age, "raw_cosine_distance": .1, "appearance_threshold": .3}


class RecoveryClassificationTests(unittest.TestCase):
    def test_zero_gt_identity_is_known_and_history_is_strictly_prior(self):
        observed = {1: {"det_index": 0, "score": .9}}
        row = classify_recovery(accepted(), 5, observed, {1: 0}, {1: (1, 0)}, {1: 1.})
        self.assertEqual(row["classification"], "correct_same_gt")
        self.assertEqual(row["previous_gt_id"], 0)
        wrong = classify_recovery(accepted(), 5, observed, {1: 2}, {1: (3, 0)}, {1: 1.})
        self.assertEqual(wrong["classification"], "wrong_identity")
        with self.assertRaisesRegex(ValueError, "strictly precede"):
            classify_recovery(accepted(), 5, observed, {1: 0}, {1: (5, 0)}, {1: 1.})

    def test_unknown_unlocalized_competing_and_not_emitted_are_separate(self):
        observed = {1: {"det_index": 0, "score": .9}}
        cases = [({}, {}, {1: 0.}, "unverifiable_no_history"),
                 ({1: (1, 0)}, {}, {1: .2}, "current_unlocalized"),
                 ({1: (1, 0)}, {}, {1: .8}, "current_unmatched_competition")]
        for previous, current, overlap, classification in cases:
            row = classify_recovery(accepted(), 5, observed, current, previous, overlap)
            self.assertEqual(row["classification"], classification)
        row = classify_recovery(accepted(), 5, {}, {}, {1: (1, 0)}, {})
        self.assertEqual(row["classification"], "not_emitted")
        self.assertIsNone(row["current_gt_id"])
        self.assertFalse(row["emitted"])

    def test_emitted_track_must_use_exact_accepted_detection_index(self):
        with self.assertRaisesRegex(ValueError, "track/detection pair"):
            classify_recovery(accepted(), 5, {1: {"det_index": 1, "score": .9}}, {1: 2}, {1: (1, 2)}, {1: 1.})


class RecoveryAttributionTests(unittest.TestCase):
    @staticmethod
    def fixture(root):
        variant, sequence = "reid_recovery", "sequence"
        run, cache, gt_root = root / "run", root / "cache", root / "gt"
        cache.mkdir()
        # (frame -> [(GT ID, x)], [(track ID, x)]), with empty frames between
        # recovery events. Tracker 6 starts as an unmatched false positive, so
        # its later recovery has no historical GT even though it had outputs.
        observations = {
            1: ([(0, 0), (1, 20), (2, 40), (3, 60)], [(1, 0), (2, 20), (3, 40), (4, 60), (6, 100)]),
            3: ([(0, 0)], [(1, 0)]),
            5: ([(1, 20)], [(1, 20)]),
            7: ([(1, 20)], [(1, 20)]),
            9: ([(2, 40)], [(3, 80)]),
            11: ([(3, 60)], [(4, 60), (2, 60)]),
            13: ([(6, 100)], [(6, 100)]),
            15: ([(0, 0)], []),
        }
        recoveries = {3: [accepted(1)], 5: [accepted(1)], 7: [accepted(1)],
                      9: [accepted(3, gap=7, age=8)], 11: [accepted(2, det_index=1, gap=9, age=10)],
                      13: [accepted(6, gap=11, age=12)], 15: [accepted(7, gap=1, age=2)]}
        frames, predictions, output_events, stage_events, truth_rows = [], [], [], [], []
        score = float(np.float32(.9))
        for frame in range(1, 16):
            truth, predicted = observations.get(frame, ([], []))
            detections = [[x, 0, x+4, 4, score, 0] for _, x in predicted]
            if frame == 15:
                detections = [[0, 0, 4, 4, score, 0]]
            frames.append({"frame": frame, "detections": detections})
            rows = []
            for index, (tid, x) in enumerate(predicted):
                predictions.append(f"{frame},{tid},{x},0,4,4,{score},-1,-1,-1\n")
                rows.append({"track_id": tid, "det_index": index, "score": score,
                             "detection_box": [x, 0, x+4, 4], "kalman_box": [x, 0, x+4, 4],
                             "output_box": [x, 0, x+4, 4]})
            output_events.append({"frame": frame, "policy": "kalman", "observations": rows})
            stage_events.append({"frame": frame, "stage": "appearance_recovery", "variant": variant,
                                 "matches": recoveries.get(frame, [])})
            truth_rows.extend(f"{frame},{gid},{x},0,4,4,1,1,1\n" for gid, x in truth)
        cache_path = cache / (sequence + ".json")
        dump_json(cache_path, {"frames": frames, "model_sha256": "d"*64})
        gt_path = gt_root / sequence / "gt/gt.txt"
        gt_path.parent.mkdir(parents=True)
        gt_path.write_text("".join(truth_rows))
        prediction_path = run / "trackers" / variant / "data" / (sequence + ".txt")
        prediction_path.parent.mkdir(parents=True)
        prediction_path.write_text("".join(predictions))
        output_path = run / "outputs" / variant / (sequence + ".jsonl")
        output_path.parent.mkdir(parents=True)
        output_path.write_text("".join(json.dumps(value) + "\n" for value in output_events))
        stage_path = run / "traces" / variant / (sequence + ".jsonl")
        stage_path.parent.mkdir(parents=True)
        stage_path.write_text("".join(json.dumps(value) + "\n" for value in stage_events))
        dump_json(run / "experiment.json", {"status": "COMPLETE", "variants": [variant], "sequences": [sequence],
            "output_trace": True, "cache_dir": str(cache), "cache_sha256": {sequence: sha256(cache_path)},
            "frames": {sequence: 15}, "effective_variants": {variant: {"output_policy": "kalman"}},
            "timing": [{"variant": variant, "sequence": sequence, "frames": 15,
                        "predictions_sha256": sha256(prediction_path), "output_trace_sha256": sha256(output_path),
                        "trace_sha256": sha256(stage_path)}]})
        dump_json(run / "metrics/summary.json", {"status": "PASS", "results": [
            {"tracker": variant, "sequence": scope, "FP": 3, "FN": 2, "IDSW": 1}
            for scope in (sequence, "COMBINED_SEQ")], "experiment_manifest_sha256": sha256(run / "experiment.json"),
            "gt_sha256": {sequence: sha256(gt_path)}, "trackeval_provenance": {"commit": TRACKEVAL_COMMIT}})
        return run, gt_root, stage_path

    @staticmethod
    def rows(path):
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    def test_actual_outputs_and_latest_prior_gt_drive_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, gt, _ = self.fixture(root)
            result = analyze_run(run, gt, root / "analysis")
            info = result["recovery_attribution"]
            self.assertEqual(info["status"], "COMPLETE")
            cases = {int(row["frame"]): row for row in self.rows(root / "analysis/recovery_cases.csv")}
            self.assertEqual(len(cases), 7)
            self.assertEqual(cases[3]["classification"], "correct_same_gt")
            self.assertEqual(cases[3]["previous_gt_id"], "0")
            self.assertEqual(cases[5]["classification"], "wrong_identity")
            self.assertEqual(cases[7]["classification"], "correct_same_gt")
            self.assertEqual(cases[7]["previous_gt_frame"], "5")
            self.assertEqual(cases[9]["classification"], "current_unlocalized")
            self.assertEqual(cases[11]["classification"], "current_unmatched_competition")
            self.assertEqual(cases[13]["classification"], "unverifiable_no_history")
            self.assertEqual(cases[15]["classification"], "not_emitted")
            combined = next(row for row in info["groups"] if row["sequence"] == "COMBINED_SEQ" and row["gap"] == "ALL")
            self.assertEqual(combined["accepted"], 7)
            self.assertEqual(combined["correct_same_gt"], 2)
            self.assertEqual(combined["wrong_identity"], 1)
            self.assertEqual(combined["identity_verifiable"], 3)
            self.assertAlmostEqual(combined["conditional_identity_precision"], 2/3)
            self.assertAlmostEqual(combined["correct_fraction_all_acceptances"], 2/7)
            self.assertIn("recovery_cases.csv", result["artifacts_sha256"])

    def test_changed_stage_trace_is_refused_before_attribution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, gt, stage = self.fixture(root)
            stage.write_text(stage.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "recovery stage trace"):
                analyze_run(run, gt, root / "analysis")

    def test_absent_instrumentation_is_skipped_not_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, gt, stage = self.fixture(root)
            stage.unlink()
            manifest = load_json(run / "experiment.json")
            del manifest["timing"][0]["trace_sha256"]
            dump_json(run / "experiment.json", manifest)
            metrics = load_json(run / "metrics/summary.json")
            metrics["experiment_manifest_sha256"] = sha256(run / "experiment.json")
            dump_json(run / "metrics/summary.json", metrics)
            result = analyze_run(run, gt, root / "analysis")
            info = result["recovery_attribution"]
            self.assertEqual(info["status"], "SKIPPED")
            self.assertIsNone(info["traces"][0]["accepted"])
            self.assertEqual(info["groups"], [])
            self.assertEqual(self.rows(root / "analysis/recovery_cases.csv"), [])

    def test_verified_empty_or_incomplete_stage_trace_is_not_zero_recoveries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, gt, stage = self.fixture(root)
            stage.write_text("")
            manifest = load_json(run / "experiment.json")
            manifest["timing"][0]["trace_sha256"] = sha256(stage)
            dump_json(run / "experiment.json", manifest)
            metrics = load_json(run / "metrics/summary.json")
            metrics["experiment_manifest_sha256"] = sha256(run / "experiment.json")
            dump_json(run / "metrics/summary.json", metrics)
            with self.assertRaisesRegex(ValueError, "cover every frame"):
                analyze_run(run, gt, root / "analysis")


if __name__ == "__main__":
    unittest.main()
