"""Synthetic R4.2 reporting/attribution checks; these are not benchmark scores."""
from collections import Counter
from copy import deepcopy
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

from tools.research.wem_r41.reporting import DETAIL_COLUMNS
from tools.research.wem_r42 import offline, reporting


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def detailed_files(root, rows):
    for arm in reporting.DEFAULT_ARMS:
        converted = []
        for row in rows:
            if row["tracker"] == arm:
                values = {field: row[key] / 100 if key in reporting.PERCENT else row[key]
                          for key, field in DETAIL_COLUMNS.items()}
                converted.append(dict(seq="COMBINED" if row["sequence"] == "COMBINED_SEQ" else row["sequence"], **values))
        write_csv(root / "metrics" / arm / "object_detailed.csv", converted)


def fixture(root, lengths=None, details=False):
    lengths = lengths or {"sequence_a": 20, "sequence_b": 20}
    profiles = {reporting.REFERENCE: (0, 0, 0), reporting.MAIN: (1, 2, -3),
                reporting.LOWER_LOW: (2, 8, -5), reporting.RELAXED: (-0.5, 5, -2),
                reporting.EMIT_ONLY: (0.25, 1, -1)}
    metrics, attrs, commits, variants = [], [], [], {}
    verified = dict(status="PASS", scope=list(lengths), scope_frames=sum(lengths.values()), arms={})
    for arm in reporting.DEFAULT_ARMS:
        shift, fp, fn = profiles[arm]
        rows, receipts, totals, atotal = [], {}, Counter(), Counter()
        weak = arm not in (reporting.REFERENCE, reporting.LOWER_LOW)
        for index, (sequence, frames) in enumerate(lengths.items()):
            row = dict(HOTA=20 + 60 * (index % 2) + shift, DetA=30 + shift,
                       AssA=50 + shift, IDF1=40 + index / 10 + shift, MOTA=-5 + shift / 10,
                       FP=10 + index + fp, FN=100 + index + fn, IDSW=2)
            rows.append(row)
            metrics.append(dict(tracker=arm, sequence=sequence, **row))
            counts = dict(frames=frames, committed=2 if weak else 0,
                          exposure_common_candidate_edges=5 if weak else 0)
            if arm in (reporting.RELAXED, reporting.EMIT_ONLY):
                counts.update(weak_outputs_emitted=2, weak_state_updates=0 if arm == reporting.EMIT_ONLY else 2)
            if arm == reporting.RELAXED:
                counts.update({key: 1 for key in reporting.ANCHOR_COUNTERS})
            totals.update(counts)
            native_key = "native" if arm in (reporting.REFERENCE, reporting.EMIT_ONLY) else arm
            receipts[sequence] = dict(counts=counts, mot=f"mot:{arm}:{sequence}",
                                       native_events=f"events:{native_key}:{sequence}", states=f"states:{native_key}:{sequence}")
            if arm != reporting.REFERENCE:
                entry = dict(arm=arm, sequence=sequence, gained=-fn, lost=0, fp_delta=fp, idsw_delta=0,
                             B_reference=2, B_actual=2, unmatched_FP=int(weak), TP_same_old_identity=int(weak),
                             TP_other_identity=0, TP_past_identity_unconfirmed=0, newly_covered=int(weak),
                             reference_B_not_matched=0, reference_B_coverage_lost=0,
                             same_id_covered_with_recent_W=0, coverage_retained=0)
                attrs.append(entry)
                atotal.update({key: value for key, value in entry.items() if isinstance(value, int)})
                if weak:
                    for frame, outcome in ((5, "TP_same_old_identity"), (9, "unmatched_FP")):
                        output_only = arm == reporting.EMIT_ONLY
                        commits.append(dict(arm=arm, sequence=sequence, frame=frame, track_id=8, det_index=3,
                                            output_kind="emitted_measurement_no_state_feedback" if output_only else "measurement",
                                            state_committed=int(not output_only), matched_gt="2" if frame == 5 else "",
                                            preceding_gt="2", newly_covered=int(frame == 5), outcome=outcome,
                                            max_iou=0.8 if frame == 5 else 0.2))
        combined = dict(HOTA=53 + shift, DetA=35 + shift, AssA=65 + shift,
                        IDF1=45 + shift, MOTA=-3 + shift / 10,
                        **{key: sum(row[key] for row in rows) for key in reporting.COUNTS})
        metrics.append(dict(tracker=arm, sequence="COMBINED_SEQ", **combined))
        verified["arms"][arm] = dict(variant=arm[3:], method="evidence_both", totals=dict(totals), per_sequence=receipts,
                                     provenance=dict(kind="reused_r41" if arm in (reporting.REFERENCE, reporting.MAIN) else "executed_r42"))
        if arm != reporting.REFERENCE:
            variants[arm] = dict(atotal)
    write_json(root / "verified.json", verified)
    write_json(root / "metrics/summary.json", dict(status="PASS", results=metrics))
    write_json(root / "attribution/summary.json", dict(status="COMPLETE", variants=variants))
    write_csv(root / "attribution/per_sequence.csv", attrs)
    write_csv(root / "attribution/committed_clear.csv", commits)
    if details:
        detailed_files(root, metrics)
    return metrics, verified, attrs, commits


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.metrics, self.verified, self.attrs, self.commits = fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def run_report(self, **kwargs):
        kwargs.setdefault("full", False)
        return reporting.report(self.root, expected_sequences=("sequence_a", "sequence_b"), **kwargs)

    def test_official_combined_not_sequence_average_and_bg_reference(self):
        result = self.run_report()
        rows = {row["arm"]: row for row in result["comparison"]}
        self.assertEqual(rows[reporting.REFERENCE]["HOTA"], 53)
        self.assertNotEqual(rows[reporting.REFERENCE]["HOTA"], (20 + 80) / 2)
        self.assertTrue(all(row["reference"] == reporting.REFERENCE for row in rows.values()))
        self.assertEqual(rows[reporting.MAIN]["delta_HOTA"], 1)
        self.assertEqual(rows[reporting.MAIN]["delta_FN"], -6)
        self.assertEqual(result["scientific_acceptance"], "NOT_AUTOMATICALLY_DECIDED")

    def test_fixed_contrast_directions_preserve_stronger_low_threshold(self):
        result = self.run_report()
        rows = {row["question"]: row for row in result["component_deltas"]}
        self.assertEqual(rows["complete_W0_effect"]["delta_HOTA"], 1)
        self.assertEqual(rows["constrained_vs_global_low_threshold"]["delta_HOTA"], -1)
        self.assertEqual(rows["constrained_vs_global_low_threshold"]["delta_FP"], -12)
        self.assertEqual(rows["relaxed_anchor_support"]["delta_HOTA"], -1.5)
        self.assertEqual(rows["future_tracker_state_feedback"]["delta_HOTA"], 0.75)

    def test_emitted_rows_and_state_updates_are_separate(self):
        result = self.run_report()
        self.assertEqual(result["weak_activity"][reporting.EMIT_ONLY]["weak_outputs_emitted"], 4)
        self.assertEqual(result["weak_activity"][reporting.EMIT_ONLY]["weak_state_updates"], 0)
        self.assertEqual(result["weak_activity"][reporting.MAIN]["weak_state_updates"], 4)
        self.assertEqual(result["weak_activity"][reporting.LOWER_LOW]["weak_outputs_emitted"], 0)
        self.assertEqual(result["checks"]["emit_only_canonical_native_state_chain"], "PASS")
        self.assertIn("derived_from", result["weak_activity"][reporting.MAIN]["count_basis"])

    def test_output_only_canonical_state_drift_is_rejected(self):
        changed = deepcopy(self.verified)
        changed["arms"][reporting.EMIT_ONLY]["per_sequence"]["sequence_a"]["states"] = "changed"
        write_json(self.root / "verified.json", changed)
        with self.assertRaisesRegex(ValueError, "canonical native states"):
            self.run_report()

    def test_output_only_state_update_count_is_rejected(self):
        changed = deepcopy(self.verified)
        changed["arms"][reporting.EMIT_ONLY]["per_sequence"]["sequence_a"]["counts"]["weak_state_updates"] = 1
        changed["arms"][reporting.EMIT_ONLY]["totals"]["weak_state_updates"] = 1
        write_json(self.root / "verified.json", changed)
        with self.assertRaisesRegex(ValueError, "weak_state_updates"):
            self.run_report()

    def test_output_only_ledger_cannot_claim_state_feedback(self):
        changed = deepcopy(self.commits)
        next(row for row in changed if row["arm"] == reporting.EMIT_ONLY)["state_committed"] = 1
        write_csv(self.root / "attribution/committed_clear.csv", changed)
        with self.assertRaisesRegex(ValueError, "no state feedback"):
            self.run_report()

    def test_scope_requires_five_prescribed_arms(self):
        with self.assertRaisesRegex(ValueError, "five fixed"):
            self.run_report(expected_arms=reporting.DEFAULT_ARMS[:-1])
        with self.assertRaisesRegex(ValueError, "30 sequences"):
            self.run_report(full=True)

    def test_full_requires_exact_names_lengths_and_official_detail_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lengths = reporting.read(ROOT / "configs/research/wem_r4_1_protocol.json")["sequences"]
            fixture(root, lengths, details=True)
            result = reporting.report(root, expected_sequences=list(lengths), full=True)
            self.assertEqual(result["frames"], 12590)
            self.assertEqual(result["status"], "FULL_EVALUATION_COMPLETE")
            self.assertEqual(len(reporting.csv_rows(root / "report/paired_per_sequence.csv")), 150)
            self.assertEqual(len(reporting.csv_rows(root / "report/component_per_sequence.csv")), 120)
            (root / "metrics" / reporting.LOWER_LOW / "object_detailed.csv").unlink()
            with self.assertRaisesRegex(ValueError, "detailed CSV"):
                reporting.report(root, expected_sequences=list(lengths), full=True)

    def test_official_detailed_values_are_checked(self):
        detailed_files(self.root, self.metrics)
        path = self.root / "metrics" / reporting.MAIN / "object_detailed.csv"
        rows = reporting.csv_rows(path)
        rows[0]["HOTA___AUC"] = float(rows[0]["HOTA___AUC"]) + 0.01
        write_csv(path, rows)
        with self.assertRaisesRegex(ValueError, "detailed"):
            self.run_report()

    def test_attribution_reconciles_coverage_and_emitted_count(self):
        rows = deepcopy(self.attrs)
        rows[0]["gained"] += 1
        write_csv(self.root / "attribution/per_sequence.csv", rows)
        with self.assertRaisesRegex(ValueError, "coverage"):
            self.run_report()
        write_csv(self.root / "attribution/per_sequence.csv", self.attrs)
        write_csv(self.root / "attribution/committed_clear.csv", self.commits[:-1])
        with self.assertRaisesRegex(ValueError, "commits"):
            self.run_report()

    def test_zero_intervention_and_identical_trajectory_are_reported_not_failed(self):
        metrics, verified, attrs = deepcopy(self.metrics), deepcopy(self.verified), deepcopy(self.attrs)
        main_metrics = {row["sequence"]: row for row in metrics if row["tracker"] == reporting.MAIN}
        for row in metrics:
            if row["tracker"] == reporting.RELAXED:
                row.update({key: main_metrics[row["sequence"]][key] for key in reporting.METRICS})
        main_attr = {row["sequence"]: row for row in attrs if row["arm"] == reporting.MAIN}
        total = Counter()
        for row in attrs:
            if row["arm"] == reporting.RELAXED:
                row.update({key: value for key, value in main_attr[row["sequence"]].items() if key != "arm"})
                total.update({key: value for key, value in row.items() if isinstance(value, int)})
        for seq, detail in verified["arms"][reporting.RELAXED]["per_sequence"].items():
            detail["mot"] = verified["arms"][reporting.MAIN]["per_sequence"][seq]["mot"]
            detail["counts"].update({key: 0 for key in reporting.ANCHOR_COUNTERS})
        verified["arms"][reporting.RELAXED]["totals"].update({key: 0 for key in reporting.ANCHOR_COUNTERS})
        attr = reporting.read(self.root / "attribution/summary.json")
        attr["variants"][reporting.RELAXED] = dict(total)
        write_json(self.root / "metrics/summary.json", dict(status="PASS", results=metrics))
        write_json(self.root / "verified.json", verified)
        write_json(self.root / "attribution/summary.json", attr)
        write_csv(self.root / "attribution/per_sequence.csv", attrs)
        result = self.run_report()
        row = next(row for row in result["component_deltas"] if row["question"] == "relaxed_anchor_support")
        self.assertEqual(row["direction_hint"], "NO_METRIC_CHANGE")
        self.assertEqual(row["decision_evidence"], "TRAJECTORIES_IDENTICAL")
        self.assertEqual(result["anchor_intervention"][reporting.ANCHOR_COUNTERS[-1]], 0)

    def test_reuse_provenance_and_timing_limits_are_preserved(self):
        result = self.run_report()
        self.assertEqual(result["reference_sources"][reporting.MAIN]["kind"], "reused_r41")
        self.assertEqual(result["timing_comparison"], "NOT_COMPARABLE_WITH_IMPORTED_REFERENCE_TIMINGS")
        rows = reporting.csv_rows(self.root / "report/exposure.csv")
        self.assertTrue(any(row["counter"] == reporting.ANCHOR_COUNTERS[-1] for row in rows))

    def test_actual_evaluator_comparison_format_is_supported(self):
        spec = importlib.util.spec_from_file_location("r42_test_evaluation", ROOT / "tools/evaluate_botsort_plus.py")
        evaluator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluator)
        summary = dict(status="PASS", results=self.metrics, modality_reference="rgb", split="test")
        validation = dict(sequences={"sequence_a": 20, "sequence_b": 20}, subset=True,
                          trackers=list(reporting.DEFAULT_ARMS))
        official = evaluator.build_comparison(summary, validation, reference_tracker=reporting.REFERENCE)
        evaluator.write_comparison(self.root / "metrics", official)
        result = self.run_report()
        self.assertEqual(next(row for row in result["comparison"] if row["arm"] == reporting.MAIN)["delta_HOTA"], 1)

    def test_report_reuse_detects_changed_outputs(self):
        expected = self.run_report()
        self.assertEqual(expected, self.run_report())
        (self.root / "report/comparison.csv").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.run_report()


class OfflineTests(unittest.TestCase):
    def test_bg_only_reference_and_output_only_semantics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names, inputs = ["sequence_a"], dict(sequences={"sequence_a": 3}, ground_truth=dict(root="unused_gt"),
                                                trackeval=dict(root="unused_evaluator"))
            write_json(root / "eval_input/protocols/BG_off.json", dict(sequences=inputs["sequences"]))
            method = "evidence_both"
            for arm in (reporting.REFERENCE, reporting.EMIT_ONLY):
                path = root / arm / "events" / method / "sequence_a.jsonl"
                path.parent.mkdir(parents=True)
                path.write_text("", encoding="utf-8")
            weak = root / reporting.EMIT_ONLY / "wem_events" / method / "sequence_a.jsonl"
            weak.parent.mkdir(parents=True)
            event = dict(event="committed", frame=1, track_id=8, det_index=3, prediction=False,
                         state_committed=False, output_only=True)
            weak.write_text(json.dumps(event) + "\n", encoding="utf-8")

            def evaluate_sequence(_trackeval, _gt, _view, arm, _seq, _n, _output):
                matched = arm != reporting.REFERENCE
                frame = dict(coverage={42: 8} if matched else {}, outputs={8: dict(matched_gt=42, max_iou=0.8)} if matched else {})
                return dict(frames=[frame, dict(coverage={}, outputs={}), dict(coverage={}, outputs={})],
                            counts=dict(FP=0, IDSW=0, FN=2 if matched else 3))

            analysis = ModuleType("experiments.vttiny_gmc.round3_joint.analysis")
            analysis.load_trackeval = lambda path: (object(), "synthetic_CLEAR")
            analysis.evaluate_sequence = evaluate_sequence
            common = ModuleType("experiments.vttiny_gmc.round3_joint.common")
            common.read_json = reporting.read
            common.write_json = write_json
            with patch.dict(sys.modules, {analysis.__name__: analysis, common.__name__: common}):
                offline.diagnose(root, (("BG", "off"), ("BG", "W0_emit_only")), names, inputs, {"BG": method})
            rows = reporting.csv_rows(root / "attribution/committed_clear.csv")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["output_kind"], "emitted_measurement_no_state_feedback")
            self.assertEqual(rows[0]["state_committed"], "0")
            self.assertEqual(rows[0]["outcome"], "TP_past_identity_unconfirmed")
            self.assertEqual(rows[0]["newly_covered"], "1")
            self.assertFalse((root / "eval_input/protocols/B_off.json").exists())


if __name__ == "__main__":
    unittest.main()
