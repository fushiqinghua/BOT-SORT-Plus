"""Sample-label scope and evidence-integrity checks for the E3 closure helper."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


_PATH = (Path(__file__).resolve().parents[1]
         / "experiments/vttiny_gmc/round3_joint/e3_review.py")
_SPEC = importlib.util.spec_from_file_location("v04_joint_e3_review_test", _PATH)
review = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(review)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class E3ClosureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "frozen_E3"
        (self.source / "B").mkdir(parents=True)
        self.rows = [
            {"case_id": "E3_001", "audit_stratum": "persistent_no_gt_support",
             "sequence": "seq_a", "original_video": "video_a", "frame": "2", "track_id": "0",
             "det_index": "0", "center_matched_gt": "", "center_iou_max": "0.0",
             "center_score": "0.4", "card_png": "E3_001/review.png",
             "human_label": "pending", "human_reviewer": "", "human_reason": ""},
            {"case_id": "E3_002", "audit_stratum": "small_static_match",
             "sequence": "seq_b", "original_video": "video_b", "frame": "5", "track_id": "3",
             "det_index": "1", "center_matched_gt": "0", "center_iou_max": "0.7",
             "center_score": "0.5", "card_png": "E3_002/review.png",
             "human_label": "pending", "human_reviewer": "", "human_reason": ""},
        ]
        self.source_csv = self.source / "B/E3_manual_review.csv"
        write_csv(self.source_csv, self.rows)
        contract = {
            "samples": 2, "source_videos": 2, "labels": list(review.LABELS),
            "manual_review_sha256": hashlib.sha256(self.source_csv.read_bytes()).hexdigest(),
        }
        (self.source / "B/review_contract.json").write_text(json.dumps(contract), encoding="utf-8")
        machine = [
            {"case_id": "E3_001", "reviewer_kind": "machine_assisted_RGB",
             "proposed_visual_label": "background_like", "visual_reason": "像背景"},
            {"case_id": "E3_002", "reviewer_kind": "machine_assisted_RGB",
             "proposed_visual_label": "uncertain", "visual_reason": "无法判定"},
        ]
        write_csv(self.source / "B/E3_machine_assisted_review.csv", machine)
        self.input = self.root / "human.csv"
        write_csv(self.input, self.rows)

    def tearDown(self):
        self.temporary.cleanup()

    def fill(self, labels=("possible_unlabelled_object", "uncertain")):
        rows = [dict(row) for row in self.rows]
        for row, label in zip(rows, labels):
            row.update(human_label=label, human_reviewer="reviewer_A", human_reason="已查看原图与相邻帧。")
        write_csv(self.input, rows)
        return rows

    def check(self):
        return review.run("check", self.input, source_root=self.source)

    def test_pending_is_incomplete_without_machine_autofill(self):
        result = self.check()
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(result["pending_count"], 2)
        self.assertEqual(result["completed_count"], 0)
        self.assertEqual(result["label_counts"]["confirmed_background"], 0)

    def test_uncertain_is_completed_but_not_background(self):
        self.fill()
        result = self.check()
        self.assertEqual(result["status"], "COMPLETE_SAMPLE_REVIEW")
        self.assertEqual(result["label_counts"]["uncertain"], 1)
        self.assertEqual(result["label_counts"]["possible_unlabelled_object"], 1)
        self.assertEqual(result["label_counts"]["confirmed_background"], 0)
        self.assertEqual(result["research_decision"], "INCONCLUSIVE_NO_ONLINE_FILTER_VALIDATION")

    def test_completed_label_requires_reviewer_and_reason(self):
        rows = self.fill()
        rows[0]["human_reviewer"] = " "
        rows[1]["human_reason"] = ""
        write_csv(self.input, rows)
        result = self.check()
        self.assertEqual(result["status"], "INVALID_REVIEW")
        self.assertEqual(len(result["errors"]), 2)
        self.assertTrue(all(row["type"] == "missing_review_metadata" for row in result["errors"]))

    def test_identity_and_frozen_metadata_cannot_change(self):
        rows = self.fill()
        rows[0]["frame"] = "3"
        rows[1]["center_matched_gt"] = "9"
        write_csv(self.input, rows)
        result = self.check()
        self.assertEqual(result["status"], "INVALID_REVIEW")
        self.assertEqual({r["case_id"] for r in result["errors"]}, {"E3_001", "E3_002"})

    def test_missing_duplicate_unknown_cases_are_rejected(self):
        rows = self.fill()
        rows[1] = dict(rows[0])
        write_csv(self.input, rows)
        result = self.check()
        self.assertEqual(result["status"], "INVALID_REVIEW")
        self.assertEqual({e["type"] for e in result["errors"]}, {"duplicate_case_ids", "case_set_changed"})

    def test_csv_reordering_is_allowed_but_blank_label_is_not(self):
        rows = self.fill()
        write_csv(self.input, rows[::-1])
        self.assertEqual(self.check()["status"], "COMPLETE_SAMPLE_REVIEW")
        rows[0]["human_label"] = ""
        write_csv(self.input, rows)
        self.assertEqual(self.check()["status"], "INVALID_REVIEW")

    def test_prepare_never_overwrites_and_never_modifies_archive(self):
        before = self.source_csv.read_bytes()
        out = self.root / "editable"
        result = review.run("prepare", output_dir=out, source_root=self.source)
        self.assertEqual(result["status"], "PREPARED")
        self.assertEqual(result["missing_cards"], ["E3_001", "E3_002"])
        with self.assertRaises(FileExistsError):
            review.run("prepare", output_dir=out, source_root=self.source)
        with self.assertRaises(ValueError):
            review.run("prepare", output_dir=self.source / "new", source_root=self.source)
        self.assertEqual(self.source_csv.read_bytes(), before)

    def test_report_joins_only_exact_center_frame_and_preserves_machine_label(self):
        self.fill()
        signals = self.root / "signals.csv"
        signal_rows = [
            {"sequence": "seq_a", "original_video": "video_a", "frame": str(frame),
             "track_id": "0", "output_score": str(score), "future_lifetime_label": "DO_NOT_EXPORT"}
            for frame, score in ((1, .1), (2, .4), (3, .8))
        ]
        write_csv(signals, signal_rows)
        out = self.root / "report"
        result = review.run("report", self.input, out, self.source, signals)
        self.assertEqual(result["causal_signals"]["matched_samples"], 1)
        self.assertEqual(result["causal_signals"]["missing_case_ids"], ["E3_002"])
        with (out / "human_machine_comparison.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["signal_output_score"], "0.4")
        self.assertEqual(rows[1]["signal_output_score"], "")
        self.assertNotIn("signal_future_lifetime_label", rows[0])
        self.assertEqual(rows[0]["machine_proposed_label"], "background_like")
        self.assertEqual(rows[0]["human_label"], "possible_unlabelled_object")
        self.assertEqual(rows[0]["comparison"], "different_category")
        self.assertEqual(rows[1]["comparison"], "both_uncertain")
        self.assertEqual(result["label_counts"]["confirmed_background"], 0)
        self.assertIn("不传播至整条轨迹", (out / "E3_REVIEW_REPORT.md").read_text(encoding="utf-8"))

    def test_duplicate_signal_center_is_rejected(self):
        self.fill()
        signals = self.root / "signals.csv"
        row = {"sequence": "seq_a", "original_video": "video_a", "frame": "2", "track_id": "0",
               "output_score": ".4"}
        write_csv(signals, [row, row])
        with self.assertRaisesRegex(ValueError, "Multiple causal rows"):
            review.run("report", self.input, self.root / "report", self.source, signals)

    def test_frozen_source_checksum_is_enforced(self):
        with self.source_csv.open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.check()


if __name__ == "__main__":
    unittest.main()
