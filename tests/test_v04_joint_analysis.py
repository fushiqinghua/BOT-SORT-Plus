"""J1 software checks; synthetic identities are not VT-Tiny-MOT results.

Set J1_TRACKEVAL_ROOT to the pinned TrackEval checkout to include the genuine
loader/CLEAR integration test.  Pure ledger tests always run.
"""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import tempfile
import unittest

from experiments.vttiny_gmc.round3_joint import analysis


def frame(index, gt_ids=(0,), coverage=None, outputs=None):
    coverage = {} if coverage is None else coverage
    outputs = {} if outputs is None else outputs
    return {"frame": index, "gt_ids": gt_ids, "coverage": coverage,
            "outputs": outputs, "counts": {"TP": len(coverage), "FP": len(outputs)-len(coverage),
                                             "FN": len(gt_ids)-len(coverage), "IDSW": 0}}


def output(tid, gt):
    return {"track_id": tid, "matched_gt": gt, "status": "TP" if gt is not None else "FP",
            "box_xywh": [0., 0., 10., 10.], "nearest_gt": 0, "max_iou": .9,
            "gt_ious": {0: .9}}


class LedgerTests(unittest.TestCase):
    def test_coverage_uses_gt_zero_and_overlap_is_not_gain_sum(self):
        arms = {}
        states = {"native": (False, True, False), "B": (True, True, False),
                  "G": (True, True, False), "BG": (True, False, True)}
        for role in analysis.ROLES:
            arms[role] = {"frames": [frame(i+1, coverage={0: 100+i} if hit else {})
                                     for i, hit in enumerate(states[role])]}
        rows = analysis.coverage_ledger("clip", arms, "original")
        self.assertEqual(sum(r["B_repairs_native_miss"] for r in rows), 1)
        self.assertEqual(sum(r["G_repairs_native_miss"] for r in rows), 1)
        self.assertEqual(sum(r["both_repair_native_miss"] for r in rows), 1)
        self.assertEqual(sum(r["joint_only_repair"] for r in rows), 1)
        self.assertEqual(sum(r["native_coverage_lost_by_BG"] for r in rows), 1)
        self.assertEqual(rows[0]["gt_id"], 0)

    def test_mixed_gt_frames_fail(self):
        arms = {role: {"frames": [frame(1)]} for role in analysis.ROLES}
        arms["G"] = {"frames": [frame(1, gt_ids=(1,))]}
        with self.assertRaisesRegex(ValueError, "GT IDs"):
            analysis.coverage_ledger("clip", arms, "original")

    def test_tail_censoring_gt_absence_and_unknown_anchor(self):
        frames = [frame(1, coverage={0: 7}, outputs={7: output(7, 0)}),
                  frame(2, gt_ids=(), outputs={7: output(7, None)}), frame(3)]
        result = analysis.followup(frames, 1, 7, 0, 5)
        self.assertEqual(result["observed_frames"], 2)
        self.assertTrue(result["right_censored"])
        self.assertEqual(result["track_FP"], 1)
        self.assertEqual(result["anchor_present_frames"], 1)
        self.assertEqual(result["anchor_FN"], 1)
        self.assertIsNone(analysis.followup(frames, 1, 7, None, 3)["anchor_FN"])

    def test_events_use_past_anchor_and_never_infer_detector_from_score(self):
        frames = [frame(1, coverage={0: 7}, outputs={7: output(7, 0)}), frame(2),
                  frame(3, coverage={0: 7}, outputs={7: output(7, 0)}),
                  frame(4, coverage={0: 8}, outputs={8: output(8, 0), 7: output(7, None)})]
        result = {"frames": frames}
        arms = {role: result for role in analysis.ROLES}
        events = [{"frame": 3, "event": "recovered", "track_id": 7, "candidate_id": 2,
                   "det_index": 12, "age": 2},
                  {"frame": 3, "event": "recovery_check", "track_id": 7,
                   "candidate_id": 2, "reason": "accepted"}]
        original = deepcopy(frames)
        accepted, proposals, rows = analysis.event_ledger("clip", "source", "BG", result, events, arms)
        self.assertEqual(accepted[0]["prior_anchor_gt"], 0)
        self.assertEqual(accepted[0]["prior_anchor_frame"], 1)
        self.assertEqual(accepted[0]["verdict"], "same_gt_identity")
        self.assertEqual(accepted[0]["det_index"], 12)
        self.assertEqual(accepted[0]["next3_anchor_other_id_TP"], 1)
        self.assertEqual(accepted[0]["next3_track_FP"], 1)
        self.assertTrue(accepted[0]["next3_right_censored"])
        self.assertIsNone(rows[0]["det_index"])
        self.assertEqual(rows[0]["stage"], "unknown")
        self.assertEqual(proposals[0]["reason"], "accepted")
        self.assertEqual(frames, original)

    def test_no_prior_identity_is_unknown_even_with_current_tp(self):
        result = {"frames": [frame(1, coverage={0: 7}, outputs={7: output(7, 0)})]}
        event = {"frame": 1, "event": "recovered", "track_id": 7, "det_index": 0}
        rows, _, _ = analysis.event_ledger("clip", "source", "B", result, [event],
                                            {role: result for role in analysis.ROLES})
        self.assertEqual(rows[0]["verdict"], "unknown_prior_identity")
        self.assertIsNone(rows[0]["prior_anchor_gt"])


@unittest.skipUnless(os.environ.get("J1_TRACKEVAL_ROOT"), "Set J1_TRACKEVAL_ROOT for pinned CLEAR integration")
class PinnedCLEARTests(unittest.TestCase):
    def test_real_loader_zero_gt_id_ignored_gt_empty_frames_and_switches(self):
        trackeval, audit = analysis.load_trackeval(os.environ["J1_TRACKEVAL_ROOT"])
        self.assertEqual(audit["commit"], "12c8791b303e0a0b50f753af204249e622d0281a")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gt_path = root / "gt/clip/gt/gt.txt"
            gt_path.parent.mkdir(parents=True)
            rows = []
            for i in range(1, 5):
                rows += [f"{i},0,0,0,10,10,1,1,1", f"{i},1,30,0,10,10,1,1,1"]
            rows.append("1,9,100,0,10,10,0,1,1")
            gt_path.write_text("\n".join(rows)+"\n", encoding="utf-8")
            pred = root / "trackers/test/data/clip.txt"
            pred.parent.mkdir(parents=True)
            # Repeated confidence is intentional: score-based row matching
            # would be ambiguous and must not be part of this audit.
            pred.write_text("\n".join([
                "1,10,0,0,10,10,0.5,-1,-1,-1", "1,20,30,0,10,10,0.5,-1,-1,-1",
                "2,10,0,0,10,10,0.5,-1,-1,-1", "2,21,30,0,10,10,0.5,-1,-1,-1",
                "4,11,0,0,10,10,0.5,-1,-1,-1", "5,99,100,0,10,10,0.5,-1,-1,-1"
            ])+"\n", encoding="utf-8")
            result = analysis.evaluate_sequence(trackeval, root/"gt", root/"trackers", "test", "clip", 5, root/"eval")
            self.assertEqual(result["counts"], {"TP": 5, "FP": 1, "FN": 3, "IDSW": 2})
            self.assertEqual(result["frames"][0]["coverage"][0], 10)
            self.assertNotIn(9, result["frames"][0]["gt_ids"])
            self.assertEqual([(r["frame"], r["gt_id"]) for r in result["switches"]], [(2, 1), (4, 0)])
            self.assertEqual(result["frames"][4]["counts"]["FP"], 1)

            # Run the public API with complete synthetic protocol contracts.
            # All four fixtures intentionally have the same predictions; the
            # archive comparison must remain valid even when every gain is 0.
            run = root / "run"
            protocols = run / "protocols"
            protocols.mkdir(parents=True)
            refs, metric_rows = {}, []
            zero = "0" * 64
            for role in analysis.ROLES:
                name = analysis.ALIASES[role]
                target = run / "trackers" / name / "data/clip.txt"
                target.parent.mkdir(parents=True)
                target.write_bytes(pred.read_bytes())
                protocol = {
                    "schema": "botsort_plus_run_v1", "status": "COMPLETE",
                    "config": {"name": name}, "native_config": {"fixture": True},
                    "sequences": {"clip": 5}, "subset": True,
                    "detector_sha256": zero, "rgb_encoder_sha256": zero,
                    "source_protocol_sha256": zero,
                    "detection_fingerprints": {"clip": zero},
                    "feature_hashes": {"clip": {"features": zero, "metadata": zero}},
                    "encoder_contracts": {"rgb": {"sha256": zero}},
                    "decoded_image_hashes": {"clip": zero},
                    "output_prediction_hashes": {"clip": analysis.digest(target)}}
                analysis.dump(protocols / f"{name}.json", protocol)
                refs[role] = {"run_dir": str(run), "tracker": name}
                metric_rows.append({"tracker": name, "sequence": "clip", "FP": 1, "FN": 3, "IDSW": 2})
            (run/"metrics").mkdir()
            analysis.dump(run/"metrics/summary.json", {"results": metric_rows})
            settings = {"references": refs, "gt_root": str(root/"gt"),
                        "trackeval_root": os.environ["J1_TRACKEVAL_ROOT"],
                        "original_video_groups": {"clip": "fixture_video"}}
            report = analysis.analyze(settings, root/"J1")
            self.assertEqual(report["status"], "COMPLETE")
            self.assertEqual(report["missing_event_sequences"], 4)
            self.assertEqual(report["event_analysis_status"], "PARTIAL_MISSING_EVENT_LOGS")
            self.assertEqual(report["coverage_overlap"]["both_repair_native_miss"], 0)
            self.assertTrue((root/"J1/output_rows_CLEAR.csv.gz").is_file())
            with self.assertRaises(FileExistsError):
                analysis.analyze(settings, root/"J1")


if __name__ == "__main__":
    unittest.main()
