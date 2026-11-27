"""Offline joins of causal J3 evidence. Synthetic data are software checks only."""
from collections import Counter
from copy import deepcopy
import gzip
import json
import os
from pathlib import Path
import tempfile
import unittest

from experiments.vttiny_gmc.round3_joint import analysis, probe_analysis as p


def clear_frame(number=3):
    return {"frame": number, "gt_ids": (0,), "gt_boxes": {0: [0., 0., 10., 10.]},
            "coverage": {0: 7}, "outputs": {7: {"matched_gt": 0, "box_xywh": [0., 0., 10., 10.],
                                               "status": "TP", "gt_ious": {0: 1.}}}}


def event_frame(number=3):
    return [{"event": "prediction_state", "frame": number, "track_id": 7,
             "anchor_frame": 1, "last_seen": 1, "state": 2, "center_pixels": 3.,
             "center_normalized": .3, "size_normalized": 0., "normalization_scale": 10.},
            {"event": "actual_recovery", "frame": number, "pairs": [[7, 20, 0, .2]],
             "current_candidates": {"20": [[2, 0], [number, 0]], "21": [[number, 1]]},
             "diagnostics": [{"track_id": 7, "candidate_id": 20, "reason": "accepted", "cost": .2},
                             {"track_id": 7, "candidate_id": 21, "reason": "geometry_gate"}]},
            {"event": "identity_competition", "frame": number, "candidate_id": 20,
             "current_det_index": 0, "scores": [
                {"track_id": 7, "role": "lost", "available": True, "score": .3, "appearance_hits": 0, "history_hits": 2},
                {"track_id": 8, "role": "active", "available": True, "score": .1, "appearance_hits": 0, "history_hits": 2},
                {"track_id": 9, "role": "active", "available": False, "score": None}]},
            {"event": "frame_end", "frame": number, "outputs": [[0, 0, 10, 10, 7, .7, 0, 0]],
             "reference_events": [{"event": "recovered", "track_id": 7, "candidate_id": 20, "det_index": 0}]}]


class ProbeAnalysisTests(unittest.TestCase):
    def test_gt_zero_raw_support_and_missing_detection(self):
        label = p.raw_detection_label(clear_frame(), [[0, 0, 10, 10, .7, 0]], 0, 0)
        self.assertEqual(label["raw_detection_status"], "supports_prior_GT")
        self.assertEqual(label["raw_matched_gt"], 0)
        self.assertEqual(label["raw_gt_already_covered"], 1)
        self.assertEqual(p.raw_detection_label(clear_frame(), None, 0, 0)["raw_detection_status"],
                         "unknown_missing_exact_detection")
        frame = clear_frame()
        frame["gt_boxes"][1] = [0., 0., 10., 10.]
        self.assertEqual(p.raw_detection_label(frame, [[0, 0, 10, 10]], 0, 0)["raw_detection_status"],
                         "unknown_multiple_GT_support")

    def test_join_preserves_unknown_and_observation_key_for_rejections(self):
        records, frame = event_frame(), clear_frame()
        before = deepcopy(records)
        routes, motion, proposals, competition = p.analyze_frame(
            records, frame, {7: (0, 1)}, [[0, 0, 10, 10], [20, 0, 30, 10]])
        self.assertEqual(proposals[0]["actual_outcome"], "same_GT_identity")
        self.assertEqual(proposals[1]["det_index"], 1)
        self.assertEqual(proposals[1]["raw_detection_status"], "below_CLEAR_IoU_threshold")
        self.assertEqual(competition[0]["active_available"], 1)
        self.assertEqual(competition[0]["active_identities"], 2)
        self.assertAlmostEqual(competition[0]["lost_minus_best_active"], .2)
        self.assertEqual(competition[0]["active_stronger"], 1)
        self.assertEqual(motion[0]["accepted_B"], 1)
        self.assertTrue(all(row["available"] == 0 for row in routes))
        self.assertEqual(records, before)

    def test_mixed_appearance_coverage_is_not_a_calibrated_margin(self):
        records = event_frame()
        records[2]["scores"][1]["appearance_hits"] = 2
        _, _, _, competition = p.analyze_frame(records, clear_frame(), {7: (0, 1)})
        self.assertEqual(competition[0]["support_coverage_status"], "coverage_mismatch_or_unknown")
        self.assertIsNone(competition[0]["lost_minus_best_active"])
        self.assertIsNone(competition[0]["active_stronger"])
        self.assertAlmostEqual(competition[0]["unadjusted_lost_minus_best_active"], .2)

    def test_route_comparison_does_not_compare_private_candidate_ids(self):
        records = [{"event": "candidate_recovery", "branch": "selected", "pairs": [[7, 11, 0, .2]]},
                   {"event": "candidate_recovery", "branch": "native", "pairs": [[7, 99, 0, .4]]},
                   {"event": "candidate_counterfactual", "branch": "selected", "candidates": {"11": [[1, 0], [2, 0]]}},
                   {"event": "candidate_counterfactual", "branch": "native", "candidates": {"99": [[1, 0], [2, 0]]}}]
        rows = {row["route"]: row for row in p.route_rows(records)}
        self.assertEqual(rows["C_B_assignment"]["changed"], 0)
        self.assertEqual(rows["C_observation_members"]["changed"], 0)

    def test_truncation_and_failed_parity_rejected(self):
        contract = {"event": "contract", "frame": 0, "schema": p.SCHEMA,
                    "reads_gt": False, "affects_reference": False}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"a.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                for value in [contract, {"event": "prediction_state", "frame": 1}]:
                    stream.write(json.dumps(value)+"\n")
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                list(p.probe_frames(path, 1))
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                for value in [contract, {"event": "parity_failure", "frame": 1, "check": "test"}]:
                    stream.write(json.dumps(value)+"\n")
            with self.assertRaisesRegex(ValueError, "parity"):
                list(p.probe_frames(path, 1))

    def test_exact_guard_membership_and_native_lost_entry(self):
        records = event_frame()
        # Keep the native reactivation fixture separate from the B candidate.
        records[-1]["outputs"].append([20, 0, 30, 10, 8, .6, 0, 1])
        records.insert(0, {"event": "prediction_state", "frame": 3, "track_id": 8,
                           "state": 2, "anchor_frame": 1, "center_normalized": .4,
                           "size_normalized": .1})
        frame = clear_frame()
        frame["outputs"][8] = {"matched_gt": None, "box_xywh": [20., 0., 10., 10.],
                               "status": "FP", "gt_ious": {0: 0.}}
        decisions = [{"event": "native_accepted", "frame": 3, "stage": "high", "track_id": 8,
                      "det_index": 1, "state_before": 2, "last_seen": 1, "gap": 2, "cost": .4},
                     {"event": "output_index", "frame": 3, "rows": [{"track_id": 7, "det_index": 0},
                                                                                {"track_id": 8, "det_index": 1}]}]
        guard = {"event": "dual_low_rejected_edge", "frame": 2, "track_id": 8, "det_index": 0,
                 "selected_iou": .7, "backup_iou": .2}
        native, joined, indices = p.decision_joins(records, frame, {7: (0, 1), 8: (0, 1)},
                                                  decisions, {(2, 0): [guard]})
        self.assertEqual(native[0]["actual_Lost_state2"], 1)
        self.assertEqual(native[0]["center_normalized"], .4)
        self.assertEqual(native[0]["actual_outcome"], "FP_localization_or_wrong_identity_unresolved")
        self.assertEqual(len(joined), 1)
        self.assertEqual(joined[0]["guard_track_id"], 8)
        self.assertEqual(joined[0]["recovered_track_id"], 7)
        self.assertEqual(joined[0]["same_identity_as_guard"], 0)
        self.assertEqual(joined[0]["later_observations_after_guard"], 1)
        self.assertIsNone(joined[0]["appearance_hits"])
        self.assertEqual(len(indices), 2)

    def test_decision_log_hash_and_output_frame_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            path = run/"decisions/test/clip.jsonl"
            path.parent.mkdir(parents=True)
            row = {"sequence": "clip", "frame": 1, "event": "output_index", "rows": []}
            path.write_text(json.dumps(row)+"\n", encoding="utf-8")
            protocol = {"joint_batch": {"decision_manifests": {"clip": {"data_sha256": analysis.digest(path), "frames": 1}}}}
            _, _, source = p.load_decisions(run, "test", "clip", 1, protocol)
            self.assertTrue(source["bound_to_protocol_hash"])
            path.write_text(path.read_text()+"\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs"):
                p.load_decisions(run, "test", "clip", 1, protocol)
            _, _, missing = p.load_decisions(run, "test", "absent", 1, protocol)
            self.assertEqual(missing["status"], "UNKNOWN_MISSING_DECISIONS")


@unittest.skipUnless(os.environ.get("J1_TRACKEVAL_ROOT"), "Set J1_TRACKEVAL_ROOT for full offline join")
class ProbeSummaryIntegration(unittest.TestCase):
    def test_complete_public_summary_with_real_CLEAR_and_hash_bound_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gt = root/"gt/clip/gt/gt.txt"
            gt.parent.mkdir(parents=True)
            gt.write_text("1,0,0,0,10,10,1,1,1\n2,0,0,0,10,10,1,1,1\n3,0,0,0,10,10,1,1,1\n")
            run = root/"run"
            mot = run/"trackers/test/data/clip.txt"
            mot.parent.mkdir(parents=True)
            mot.write_text("1,7,0,0,10,10,0.7,-1,-1,-1\n3,7,0,0,10,10,0.7,-1,-1,-1\n")
            trace = run/"probes/test/clip.jsonl.gz"
            trace.parent.mkdir(parents=True)
            contract = {"event": "contract", "frame": 0, "schema": p.SCHEMA,
                        "reads_gt": False, "affects_reference": False}
            first = [{"event": "actual_recovery", "frame": 1, "pairs": [], "diagnostics": []},
                     {"event": "frame_end", "frame": 1, "outputs": [[0, 0, 10, 10, 7, .7, 0, 0]], "reference_events": []}]
            second = [{"event": "actual_recovery", "frame": 2, "pairs": [], "diagnostics": []},
                      {"event": "frame_end", "frame": 2, "outputs": [], "reference_events": []}]
            records = [contract, *first, *second, *event_frame()]
            with gzip.open(trace, "wt", encoding="utf-8") as stream:
                for row in records:
                    stream.write(json.dumps(row)+"\n")
            summary = {"schema": p.SCHEMA, "status": "PASS", "mode": "signals", "frames": 3,
                       "counts": dict(Counter(row["event"] for row in records)), "data_sha256": analysis.digest(trace)}
            analysis.dump(trace.with_name(trace.name+".summary.json"), summary)
            zero = "0"*64
            protocol = {"schema": "botsort_plus_run_v1", "status": "COMPLETE", "config": {"name": "test"},
                        "native_config": {"fixture": True}, "sequences": {"clip": 3}, "subset": True,
                        "detector_sha256": zero, "rgb_encoder_sha256": zero, "source_protocol_sha256": zero,
                        "detection_fingerprints": {"clip": zero},
                        "feature_hashes": {"clip": {"features": zero, "metadata": zero}},
                        "encoder_contracts": {"rgb": {"sha256": zero}},
                        "output_prediction_hashes": {"clip": analysis.digest(mot)},
                        "joint_batch": {"probe_manifests": {"clip": summary}}}
            (run/"protocols").mkdir()
            analysis.dump(run/"protocols/test.json", protocol)
            settings = {"gt_root": str(root/"gt"), "trackeval_root": os.environ["J1_TRACKEVAL_ROOT"],
                        "original_video_groups": {"clip": "source"}}
            result = p.summarize(settings, run, "test", root/"summary")
            self.assertEqual(result["status"], "COMPLETE")
            self.assertEqual(result["actual_B_outcomes"], {"same_GT_identity": 1})
            self.assertEqual(result["active_stronger_accepted"], 1)
            self.assertEqual(result["raw_detection_status"], {"unknown_missing_exact_detection": 2})
            self.assertTrue(result["probe_sources"]["clip"]["bound_to_protocol_hash"])


if __name__ == "__main__":
    unittest.main()
