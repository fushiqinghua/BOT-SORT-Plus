"""Synthetic tests for paired reporting; fixtures are not dataset results."""
from collections import Counter
from copy import deepcopy
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("wem_r41_reporting", ROOT / "tools/research/wem_r41/reporting.py")
reporting = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporting)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def fixture(root, sequences=("sequence_a", "sequence_b")):
    """Make internally consistent official-style metrics and event ledgers."""
    profiles = {"off": (0.0, 0, 0, 0), "W0": (1.0, 2, -3, 0),
                "W1": (2.0, 1, -4, 0), "W2": (2.0, 1, -4, 0), "W3": (-0.5, 0, -2, 1)}
    metrics, per_sequence, commits, variants = [], [], [], {}
    verified = dict(status="PASS", scope=list(sequences), arms={})
    for arm in reporting.DEFAULT_ARMS:
        role, policy = arm.split("_")
        shift, fp, fn, idsw = profiles[policy]
        records, details, totals, attr_totals = [], {}, Counter(), Counter()
        for index, seq in enumerate(sequences):
            # Deliberately not an average-compatible set of percentage metrics:
            # official COMBINED_SEQ is a separate evaluator calculation.
            role_shift = 6 if role == "BG" else 0
            raw = dict(HOTA=20.0 + 60 * (index % 2) + role_shift + shift / (2 if role == "BG" else 1),
                       DetA=20.0 + index, AssA=50.0 + index,
                       IDF1=30.0 + index + shift, MOTA=-10.0 + index + shift / 10,
                       FP=10 + index - (2 if role == "BG" else 0) + fp,
                       FN=100 + index - (2 if role == "BG" else 0) + fn,
                       IDSW=2 + idsw)
            records.append(raw)
            metrics.append(dict(tracker=arm, sequence=seq, **raw))
            counts = dict(frames=20, committed=0 if policy == "off" else 2,
                          exposure_motion_unavailable=0, exposure_common_candidate_edges=5,
                          compare_W0_to_W_H_changed_track_frames=0)
            totals.update(counts)
            details[seq] = dict(counts=counts)
            if policy != "off":
                entry = dict(arm=arm, sequence=seq, gained=-fn, lost=0, fp_delta=fp, idsw_delta=idsw,
                             B_reference=2, B_actual=2, unmatched_FP=1, TP_same_old_identity=1,
                             TP_other_identity=0, TP_past_identity_unconfirmed=0, newly_covered=1,
                             reference_B_not_matched=0, reference_B_coverage_lost=0,
                             same_id_covered_with_recent_W=0, coverage_retained=0)
                per_sequence.append(entry)
                attr_totals.update({key: value for key, value in entry.items() if isinstance(value, int)})
                for frame, outcome in ((5, "TP_same_old_identity"), (9, "unmatched_FP")):
                    commits.append(dict(arm=arm, sequence=seq, frame=frame, track_id=8, det_index=3,
                                        output_kind="measurement", matched_gt="2" if frame == 5 else "",
                                        preceding_gt="2", newly_covered=int(frame == 5), outcome=outcome,
                                        max_iou=0.8 if frame == 5 else 0.2))
        combined = dict(HOTA=53.0 + role_shift + shift / (2 if role == "BG" else 1),
                        DetA=35.0, AssA=65.0, IDF1=45.0 + shift, MOTA=-3.0 + shift / 10,
                        **{name: sum(row[name] for row in records) for name in reporting.COUNTS})
        metrics.append(dict(tracker=arm, sequence="COMBINED_SEQ", **combined))
        verified["arms"][arm] = dict(variant=policy, method="evidence_both" if role == "BG" else "B_early_moderate",
                                     totals=dict(totals), per_sequence=details)
        if policy != "off":
            variants[arm] = dict(attr_totals)
    write_json(root / "metrics/summary.json", dict(status="PASS", results=metrics))
    write_json(root / "verified.json", verified)
    write_json(root / "attribution/summary.json", dict(status="COMPLETE", variants=variants))
    write_csv(root / "attribution/per_sequence.csv", per_sequence)
    write_csv(root / "attribution/committed_clear.csv", commits)
    return metrics, verified, per_sequence, commits


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.rows, self.verified, self.attribution, self.commits = fixture(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def run_report(self, **kwargs):
        return reporting.report(self.root, expected_sequences=("sequence_a", "sequence_b"), **kwargs)

    def test_uses_official_combined_and_each_own_reference(self):
        result = self.run_report()
        rows = {row["arm"]: row for row in result["comparison"]}
        self.assertEqual(rows["B_off"]["HOTA"], 53)
        self.assertNotEqual(rows["B_off"]["HOTA"], (20 + 80) / 2)
        self.assertEqual(rows["BG_W0"]["reference"], "BG_off")
        self.assertEqual(rows["BG_W0"]["delta_HOTA"], 0.5)
        self.assertEqual(rows["B_W0"]["delta_HOTA"], 1)
        self.assertEqual(rows["BG_W0"]["delta_FN"], -6)
        self.assertEqual(result["frames"], 40)
        self.assertEqual(result["status"], "SUBSET_EVALUATION_COMPLETE")
        self.assertEqual(result["scientific_acceptance"], "NOT_AUTOMATICALLY_DECIDED")

    def test_component_and_interaction_are_actual_differences(self):
        result = self.run_report()
        components = {row["comparison"]: row for row in result["component_deltas"]}
        self.assertEqual(components["B_W1_minus_B_W0"]["delta_HOTA"], 1)
        self.assertEqual(components["B_W2_minus_B_W1"]["direction_hint"], "NO_METRIC_CHANGE")
        interactions = {row["policy"]: row for row in result["interactions"]}
        self.assertEqual(interactions["W0"]["interaction_HOTA"], -0.5)
        self.assertEqual(interactions["W1"]["interaction_HOTA"], -1)

    def test_full_refuses_subset(self):
        with self.assertRaisesRegex(ValueError, "30"):
            self.run_report(full=True)
        self.assertFalse((self.root / "report").exists())

    def test_full_all_30_sequences_and_all_10_arms(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = tuple(f"sequence_{i:02}" for i in range(30))
            fixture(root, names)
            result = reporting.report(root, expected_sequences=names, full=True)
            self.assertEqual(result["status"], "FULL_EVALUATION_COMPLETE")
            self.assertEqual(result["frames"], 600)
            self.assertEqual(len(reporting.csv_rows(root / "report/paired_per_sequence.csv")), 300)

    def test_missing_main_arm_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "ten main"):
            self.run_report(expected_arms=reporting.DEFAULT_ARMS[:-1])

    def test_missing_or_duplicate_sequence_is_rejected(self):
        path = self.root / "metrics/summary.json"
        document = reporting.read(path)
        document["results"].pop(0)
        write_json(path, document)
        with self.assertRaisesRegex(ValueError, "sequence scope"):
            self.run_report()
        document["results"] = self.rows + [self.rows[0]]
        write_json(path, document)
        with self.assertRaisesRegex(ValueError, "Repeated"):
            self.run_report()

    def test_nonfinite_metrics_and_negative_counts_are_rejected(self):
        path = self.root / "metrics/summary.json"
        for value in (float("nan"), float("inf"), -1):
            rows = deepcopy(self.rows)
            rows[0]["FP"] = value
            write_json(path, dict(status="PASS", results=rows))
            with self.assertRaisesRegex(ValueError, "numeric"):
                self.run_report()

    def test_combined_counts_must_equal_sequence_sums(self):
        rows = deepcopy(self.rows)
        next(row for row in rows if row["tracker"] == "BG_W1" and row["sequence"] == "COMBINED_SEQ")["FP"] += 1
        write_json(self.root / "metrics/summary.json", dict(status="PASS", results=rows))
        with self.assertRaisesRegex(ValueError, "sum"):
            self.run_report()

    def test_attribution_must_reconcile_against_own_reference(self):
        rows = deepcopy(self.attribution)
        next(row for row in rows if row["arm"] == "BG_W1")["gained"] += 1
        write_csv(self.root / "attribution/per_sequence.csv", rows)
        with self.assertRaisesRegex(ValueError, "coverage"):
            self.run_report()

    def test_committed_events_must_match_prediction_receipts(self):
        write_csv(self.root / "attribution/committed_clear.csv", self.commits[:-1])
        with self.assertRaisesRegex(ValueError, "commits"):
            self.run_report()

    def test_exposure_counters_are_preserved_and_receipts_reconciled(self):
        result = self.run_report()
        rows = reporting.csv_rows(self.root / "report/exposure.csv")
        self.assertTrue(any(row["counter"] == "compare_W0_to_W_H_changed_track_frames" for row in rows))
        self.assertEqual(result["checks"]["prediction_receipts"], "PASS")
        changed = deepcopy(self.verified)
        changed["arms"]["B_W1"]["totals"]["exposure_common_candidate_edges"] += 1
        write_json(self.root / "verified.json", changed)
        with self.assertRaisesRegex(ValueError, "receipt total"):
            self.run_report()

    def test_optional_detailed_csv_must_agree(self):
        arm = "B_off"
        rows = []
        for row in self.rows:
            if row["tracker"] != arm:
                continue
            converted = {field: row[name] / 100 if name in reporting.PERCENT else row[name]
                         for name, field in reporting.DETAIL_COLUMNS.items()}
            rows.append(dict(seq="COMBINED" if row["sequence"] == "COMBINED_SEQ" else row["sequence"], **converted))
        rows[0]["HOTA___AUC"] += 0.001
        write_csv(self.root / f"metrics/{arm}/object_detailed.csv", rows)
        with self.assertRaisesRegex(ValueError, "detailed"):
            self.run_report()

    def test_actual_evaluator_comparison_schema_keeps_bg_pairing_correct(self):
        spec = importlib.util.spec_from_file_location("r41_baseline_evaluation", ROOT / "tools/evaluate_botsort_plus.py")
        evaluator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluator)
        document = reporting.read(self.root / "metrics/summary.json")
        document.update(modality_reference="rgb", split="test")
        validation = dict(sequences={"sequence_a": 20, "sequence_b": 20}, subset=True,
                          trackers=list(reporting.DEFAULT_ARMS))
        comparison = evaluator.build_comparison(document, validation, reference_tracker="B_off")
        evaluator.write_comparison(self.root / "metrics", comparison)
        # The upstream evaluator deliberately compares every arm to B_off.
        upstream_bg = next(row for row in comparison["results"] if row["tracker"] == "BG_W0")
        self.assertEqual(upstream_bg["delta"]["HOTA"], 6.5)
        result = self.run_report()
        ours = next(row for row in result["comparison"] if row["arm"] == "BG_W0")
        self.assertEqual(ours["delta_HOTA"], 0.5)

    def test_unfavorable_metrics_do_not_block_full_reporting(self):
        result = self.run_report()
        rows = {row["arm"]: row for row in result["comparison"]}
        self.assertEqual(rows["B_W3"]["direction_hint"], "UNFAVORABLE_DIRECTION")
        self.assertTrue((self.root / "report/REPORT.md").exists())

    def test_report_is_idempotent_but_detects_changed_deliverables(self):
        expected = self.run_report()
        self.assertEqual(self.run_report(), expected)
        path = self.root / "report/comparison.csv"
        path.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.run_report()

    def test_attribution_summary_cannot_hide_sequence_error(self):
        path = self.root / "attribution/summary.json"
        document = reporting.read(path)
        document["variants"]["B_W1"]["FP"] = 0
        write_json(path, document)
        with self.assertRaisesRegex(ValueError, "summary fields"):
            self.run_report()


if __name__ == "__main__":
    unittest.main()
