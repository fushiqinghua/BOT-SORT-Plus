"""Synthetic, receipt-backed tests for the portable R3.1 cache comparison.

The private training split is replaced only by an in-process unittest mock.
The real CLI has no test-pin bypass. No dataset, images, GT files, or models are
required by these fixtures.
"""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import copy
import csv
import gzip
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.research.wem_r31 import batch, inputs, protocol, reporting
from tools.research.wem_r31.engine import METHODS, edge_key


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return inputs.digest(path.read_bytes())


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows).encode("utf-8")
    path.write_bytes(gzip.compress(content, mtime=0))
    return inputs.digest(path.read_bytes())


def feature(sequence, frame, track, det, *, iou, motion, cosine, rank=1,
            row_margin=None, column_margin=None, challengers=None):
    score = 0.08
    normalized = (score - .05) / .05
    history = None if cosine is None or motion is None else (iou + normalized + (cosine + 1) / 2 + motion) / 4
    return dict(reference=protocol.REFERENCE, sequence=sequence, frame=frame, track_id=track, det_index=det,
                raw_box_xyxy=[20. * det, 0., 20. * det + 10., 10.], raw_score=score,
                native_iou=iou, normalized_score=normalized, cosine=cosine, motion=motion,
                history_score=history, geometry_group_rank=rank,
                row_unknown=False, column_unknown=False, row_margin=row_margin, column_margin=column_margin,
                rgb_weak_valid=True, valid_history_rgb=2 if cosine is not None else 1,
                previous_regular_observations=2, crop_wh=[10, 10], challengers=challengers or [])


class ReceiptFixture:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.features_dir = self.directory / "features"
        self.labels_dir = self.directory / "labels"
        self.output = self.directory / "new_comparison"
        self.split_path = self.directory / "training_split.json"
        self.settings_path = self.directory / "run.local.json"
        self.features, self.labels = {}, {}
        self.split = {"status": "LOCKED", "training_sequence_to_source_video": {},
                      "development_sequence_to_source_video": {"official_test": "test_source"}, "split": {}}
        for part in protocol.PARTITIONS:
            seq, source = part + "_clip", "source_" + part
            self.split["training_sequence_to_source_video"][seq] = source
            self.split["split"][part] = {"sequences": {seq: 3}, "source_videos": [source]}
            rivals = [dict(track_id=99, source="high", history_observations=2, native_iou=.79, cosine=.88)]
            self.features[seq] = [
                feature(seq, 1, 1, 0, iou=.80, motion=.80, cosine=-.8, row_margin=-.2075),
                feature(seq, 1, 1, 1, iou=.79, motion=.79, cosine=.9, rank=2,
                        row_margin=.2075, column_margin=.005, challengers=rivals),
                feature(seq, 1, 2, 2, iou=.70, motion=.80, cosine=None),
                feature(seq, 1, 3, 3, iou=.65, motion=.10, cosine=.95),
                feature(seq, 2, 4, 0, iou=.90, motion=.90, cosine=.95),
            ]
            self.labels[part] = []
            for index, row in enumerate(self.features[seq]):
                known = index in (0, 2, 4)
                label = {key: copy.deepcopy(row[key]) for key in
                         ("reference", "sequence", "frame", "track_id", "det_index", "raw_box_xyxy", "raw_score")}
                label.update(partition=part, source_video=source, formal_geometry_allowed=True,
                             gmc_guard_allowed=True, hard_ownership_conflict=False, near_duplicate_ownership_risk=False,
                             identity_label="same_identity" if known else "unknown", opportunity=known,
                             gt_id=row["track_id"] * 10 if known else None,
                             fn_episode_start_frame=row["frame"] if known else None,
                             identity_rule="synthetic_fixture", anchor_status="synthetic_fixture",
                             best_gt_iou=.8 if known else .2)
                self.labels[part].append(label)
        self.split_sha = write_json(self.split_path, self.split)
        parity = inputs.digest(b"synthetic parity")
        self.feature_manifest = dict(status="COMPLETE", scope="smoke", reference=protocol.REFERENCE,
                                     encoder={"sha256": inputs.PINS["encoder"]},
                                     reference_protocol_sha256=inputs.PINS["reference_protocol"],
                                     input_split_sha256=self.split_sha, parity_sha256=parity, sequences={})
        self.label_manifest = dict(status="TRAINING_DIAGNOSTIC_LABELS_ONLY", stage="smoke", reference=protocol.REFERENCE,
                                   split_sha256=self.split_sha, official_test_gt_used=False,
                                   source={"capture": parity}, partitions={})
        for part in protocol.PARTITIONS:
            seq = part + "_clip"
            snapshot = inputs.digest(seq.encode("utf-8"))
            feature_sha = write_rows(self.features_dir / (seq + ".jsonl.gz"), self.features[seq])
            label_sha = write_rows(self.labels_dir / part / "candidate_edges.jsonl.gz", self.labels[part])
            self.feature_manifest["sequences"][seq] = dict(partition=part, snapshot=snapshot,
                                                           counts={"edges": len(self.features[seq])}, scores_sha256=feature_sha)
            self.label_manifest["partitions"][part] = dict(labels_sha256=label_sha, sequences=[
                dict(sequence=seq, provenance={"snapshot_sha256": snapshot})])
        self.persist_manifests()
        self.settings = {"schema": "wem_r31_settings", "scope": "smoke", "paths": {
            "features_dir": str(self.features_dir), "labels_dir": str(self.labels_dir),
            "split_file": str(self.split_path), "output_dir": str(self.output)}}
        write_json(self.settings_path, self.settings)

    def persist_manifests(self):
        write_json(self.features_dir / "manifest.json", self.feature_manifest)
        write_json(self.labels_dir / "summary.json", self.label_manifest)

    def update_labels(self, part):
        signature = write_rows(self.labels_dir / part / "candidate_edges.jsonl.gz", self.labels[part])
        self.label_manifest["partitions"][part]["labels_sha256"] = signature
        self.persist_manifests()

    def update_features(self, seq):
        signature = write_rows(self.features_dir / (seq + ".jsonl.gz"), self.features[seq])
        self.feature_manifest["sequences"][seq].update(scores_sha256=signature, counts={"edges": len(self.features[seq])})
        self.persist_manifests()


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="wem_r31_中文 space_")
        self.addCleanup(self.temp.cleanup)
        self.fixture = ReceiptFixture(self.temp.name)
        # Tests alone replace the private split pin. There is deliberately no
        # CLI flag that bypasses frozen source validation.
        pins = mock.patch.dict(inputs.PINS, {"split": self.fixture.split_sha})
        pins.start()
        self.addCleanup(pins.stop)

    def bundle(self):
        return inputs.load_bundle(inputs.load_settings(self.fixture.settings_path, ROOT))

    def test_cli_check_and_run_produce_five_methods_and_hashed_outputs(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(batch.main(["check", "--settings", str(self.fixture.settings_path)]), 0)
        self.assertEqual(json.loads(stdout.getvalue())["status"], "INPUTS_VERIFIED_NOT_RUN")
        self.assertFalse(self.fixture.output.exists())
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(batch.main(["run", "--settings", str(self.fixture.settings_path)]), 0)
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["methods"], list(METHODS))
        self.assertFalse(result["enabled_online"])
        summary = inputs.read_json(self.fixture.output / "summary.json")[0]
        self.assertEqual(summary["scientific_acceptance"], "NOT_EVALUATED")
        self.assertEqual(summary["historical_control_parity"]["status"], "NOT_APPLICABLE_SMOKE")
        self.assertEqual(len(summary["comparison"]), 15)
        manifest = inputs.read_json(self.fixture.output / "manifest.json")[0]
        outputs = {path.name for path in self.fixture.output.iterdir()} - {"manifest.json"}
        self.assertEqual(set(manifest["output_sha256"]), outputs)
        for name, fingerprint in manifest["output_sha256"].items():
            self.assertEqual(inputs.digest((self.fixture.output / name).read_bytes()), fingerprint)
        self.assertTrue(manifest["input_files"])
        self.assertIn("tools/research/wem_r31/engine.py", manifest["code_sha256"])
        decoded = gzip.decompress((self.fixture.output / "decisions.jsonl.gz").read_bytes()).decode()
        decisions = [json.loads(line) for line in decoded.splitlines()]
        self.assertEqual(len(decisions), 3 * 5 * 5)

    def test_unknown_review_queue_and_paired_deltas_are_explicit(self):
        bundle = self.bundle()
        decisions = reporting.evaluate(bundle)
        queue = reporting.review_queue(bundle, decisions)
        self.assertEqual(len(queue), 6)
        self.assertTrue(all(item["human_review_status"] == "pending" for item in queue))
        unknown_from_soft = [item for item in queue if item["track_id"] == 1]
        self.assertEqual(len(unknown_from_soft), 3)
        self.assertTrue(all(item["adopted_by"] == "W2" for item in unknown_from_soft))
        deltas = reporting.paired_deltas(decisions)
        changed = [row for row in deltas if row["comparison"] == "W1_to_W2"]
        self.assertEqual(len(changed), 3)
        for item in changed:
            self.assertTrue(item["same_eligible_edges"])
            self.assertEqual(item["added_unknown"], 1)
            self.assertEqual(item["removed_same_identity"], 1)
            self.assertEqual(item["useful_target_frame_delta"], -1)

    def test_relabeling_does_not_change_inference_choices(self):
        bundle = self.bundle()
        before = reporting.evaluate(bundle)
        altered = copy.deepcopy(bundle)
        for label in altered.labels.values():
            label.update(identity_label="unknown", opportunity=False, gt_id=None, fn_episode_start_frame=None)
        after = reporting.evaluate(altered)
        self.assertEqual({(row["method"], edge_key(row)) for row in before if row["selected"]},
                         {(row["method"], edge_key(row)) for row in after if row["selected"]})
        self.assertTrue(any(row["opportunity"] for row in before))
        self.assertFalse(any(row["opportunity"] for row in after))

    def test_absent_negative_denominators_are_null_not_zero_risk(self):
        bundle = self.bundle()
        table, _, _ = reporting.build_tables(bundle, reporting.evaluate(bundle))
        for row in table:
            self.assertIsNone(row["other_identity_acceptance_rate"])
            self.assertIsNone(row["confirmed_background_acceptance_rate"])
            self.assertEqual(row["background_budget_state"], "NOT_AVAILABLE")
            self.assertEqual(row["accepted_confirmed_background"], 0)

    def test_historical_control_parity_checks_all_pinned_metrics(self):
        bundle = self.bundle()
        table, _, _ = reporting.build_tables(bundle, reporting.evaluate(bundle))
        archive = self.fixture.directory / "synthetic_fixed_diagnostics.csv"
        rows = [dict(row, method="geometry" if row["method"] == "W0" else "history_competition", budget=5)
                for row in table if row["method"] in ("W0", "W_old")]
        reporting.write_csv(archive, rows)
        check = reporting.check_historical_controls(table, archive)
        self.assertEqual(check["status"], "PASS")
        self.assertEqual(len(check["checks"]), 6)
        rows[0]["accepted"] += 1
        reporting.write_csv(archive, rows)
        with self.assertRaisesRegex(ValueError, "Historical control parity failed"):
            reporting.check_historical_controls(table, archive)

    def test_corrupt_compressed_bytes_fail_before_creating_output(self):
        path = self.fixture.features_dir / "fit_clip.jsonl.gz"
        path.write_bytes(path.read_bytes() + b"tampered")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            batch.run_comparison(self.fixture.settings_path, root=ROOT)
        self.assertFalse(self.fixture.output.exists())

    def test_failed_output_write_removes_atomic_stage(self):
        with mock.patch.object(reporting, "write_decisions", side_effect=OSError("synthetic disk failure")):
            with self.assertRaisesRegex(OSError, "synthetic disk failure"):
                batch.run_comparison(self.fixture.settings_path, root=ROOT)
        self.assertFalse(self.fixture.output.exists())
        self.assertEqual(list(self.fixture.directory.glob(".wem_r31_*")), [])

    def test_qualification_mismatch_is_not_silently_filtered(self):
        self.fixture.labels["fit"][0]["gmc_guard_allowed"] = False
        self.fixture.update_labels("fit")
        with self.assertRaisesRegex(ValueError, "Unlabeled"):
            self.bundle()
        self.assertFalse(self.fixture.output.exists())

    def test_raw_box_mismatch_is_rejected(self):
        self.fixture.labels["fit"][0]["raw_box_xyxy"][0] += 1
        self.fixture.update_labels("fit")
        with self.assertRaisesRegex(ValueError, "Raw score/coordinates"):
            self.bundle()

    def test_duplicate_offline_edge_is_rejected(self):
        self.fixture.labels["fit"].append(copy.deepcopy(self.fixture.labels["fit"][0]))
        self.fixture.update_labels("fit")
        with self.assertRaisesRegex(ValueError, "Repeated offline"):
            self.bundle()

    def test_missing_sequence_or_unscored_qualified_edge_is_rejected(self):
        old = self.fixture.feature_manifest["sequences"].pop("holdout_clip")
        self.fixture.persist_manifests()
        with self.assertRaisesRegex(ValueError, "one sequence in each partition"):
            self.bundle()
        self.fixture.feature_manifest["sequences"]["holdout_clip"] = old
        self.fixture.features["fit_clip"].pop()
        self.fixture.update_features("fit_clip")
        with self.assertRaisesRegex(ValueError, "one-to-one correspondence"):
            self.bundle()

    def test_existing_output_and_input_overlap_are_rejected(self):
        self.fixture.output.mkdir()
        marker = self.fixture.output / "preserve.txt"
        marker.write_text("keep")
        with self.assertRaisesRegex(ValueError, "already exists"):
            batch.run_comparison(self.fixture.settings_path, root=ROOT)
        self.assertEqual(marker.read_text(), "keep")
        self.fixture.settings["paths"]["output_dir"] = str(self.fixture.features_dir / "new_output")
        write_json(self.fixture.settings_path, self.fixture.settings)
        with self.assertRaisesRegex(ValueError, "overlaps"):
            inputs.load_settings(self.fixture.settings_path, ROOT)

    def test_cli_init_and_failure_exit_codes(self):
        path = self.fixture.directory / "fresh.local.json"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(batch.main(["init", "--settings", str(path)]), 0)
        self.assertEqual(inputs.read_json(path)[0], protocol.SETTINGS)
        with redirect_stderr(io.StringIO()):
            self.assertEqual(batch.main(["init", "--settings", str(path)]), 2)
            self.assertEqual(batch.main(["check", "--settings", str(path)]), 2)

    def test_source_hash_and_pretop_rank_are_validated(self):
        self.fixture.label_manifest["source"]["capture"] = "wrong"
        self.fixture.persist_manifests()
        with self.assertRaisesRegex(ValueError, "provenance"):
            self.bundle()
        self.fixture.label_manifest["source"]["capture"] = self.fixture.feature_manifest["parity_sha256"]
        self.fixture.features["fit_clip"][1]["geometry_group_rank"] = 1
        self.fixture.update_features("fit_clip")
        with self.assertRaisesRegex(ValueError, "top-k rank"):
            self.bundle()


if __name__ == "__main__":
    unittest.main()
