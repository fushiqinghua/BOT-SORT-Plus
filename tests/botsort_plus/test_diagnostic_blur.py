"""D4 diagnostic correctness tests using explicitly synthetic software fixtures.

These do not establish blur prevalence, detector/re-ID quality, or MOT gains.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from botsort_plus.diagnostics.blur import (
    REVIEW_FIELDS, analyze_blur, appearance_evidence, load_manual_labels,
    native_roi_metrics,
)
from botsort_plus.io import detection_fingerprint, sha256, write_json


def write_csv(path, rows, fields=None):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def make_fixture(root):
    """Six frames, an empty frame, missing embeddings, and unmatched detection."""
    root = Path(root)
    baseline, features, images = root/"baseline", root/"features", root/"images"
    run_root, audit = root/"run", root/"audit"
    for folder in (baseline/"detections", features, images/"seq", run_root/"protocols",
                   run_root/"diagnostics"/"fixture", audit):
        folder.mkdir(parents=True)
    write_json(baseline/"protocol.json", {"status": "COMPLETE", "fixture": "synthetic software test"})
    vectors = {1: ([1., 0.], [0., 1.]), 2: ([.8, .6], None),
               3: (None, [0., 1.]), 4: (), 5: ([1., 0.], [0., 1.]),
               6: ([1., 0.], [0., 1.])}
    cache = {"model_sha256": "d"*64, "frames": []}
    rgb, valid, quality, offsets, rows, traces = [], [], [], [0], [], []
    for frame, items in vectors.items():
        image = np.full((32, 64, 3), 128, dtype=np.uint8)
        image_path = Path("seq")/f"{frame:03d}.png"
        assert cv2.imwrite(str(images/image_path), image)
        detections = [[5+30*i, 5, 13+30*i, 13, .8, 0] for i in range(len(items))]
        if frame == 1:
            detections.append([50, 2, 54, 6, .3, 0])
            items = (*items, [1., 0.])
        cache["frames"].append({"frame": frame, "image": str(image_path), "detections": detections})
        for index, (det, vector) in enumerate(zip(detections, items)):
            matched = index < 2
            rgb.append(vector if vector is not None else [0., 0.])
            valid.append(vector is not None)
            quality.append(.8 if vector is not None else 0.)
            row = {"sequence": "seq", "frame": frame, "det_index": index,
                   "gt_id": index+1 if matched else None,
                   "gt_match_status": "matched" if matched else "no_gt",
                   "gt_class": 1 if matched else None, "gt_visibility": 1. if matched else None,
                   **dict(zip(("x1", "y1", "x2", "y2", "score"), det[:5])),
                   "rgb_valid": vector is not None, "rgb_quality": .8 if vector is not None else None,
                   "size_bin": "8to16" if matched else "lt8",
                   "det_center_error_px": frame*.1 if matched else None,
                   "det_center_error_norm": frame*.1/8 if matched else None,
                   "detection_tier": "high" if matched else "low",
                   "accepted_track_id": 1 if matched and index == 1 else None,
                   "accepted_stage": "high" if matched and index == 1 else None}
            rows.append(row)
        offsets.append(len(rgb))
        traces.append({"schema": "botsort_motion_diagnostics_v1", "sequence": "seq", "frame": frame,
                       "image_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                       "image_shape": list(image.shape), "image_dtype": str(image.dtype)})
    write_json(baseline/"detections"/"seq.json", cache)
    feature_path = features/"seq.npz"
    np.savez_compressed(feature_path, frame_ids=np.arange(1, 7), offsets=np.array(offsets),
                        rgb=np.asarray(rgb, np.float32), rgb_valid=np.asarray(valid, bool),
                        rgb_quality=np.asarray(quality, np.float32))
    write_json(features/"seq.json", {"schema": "botsort_plus_features_v1",
               "features_sha256": sha256(feature_path),
               "detection_fingerprint": detection_fingerprint(cache),
               "encoders": {"rgb": {"sha256": "a"*64, "output_dim": 2}}})
    trace_path = run_root/"diagnostics"/"fixture"/"seq.jsonl.gz"
    with gzip.open(trace_path, "wt", encoding="utf-8") as stream:
        for trace in traces:
            stream.write(json.dumps(trace)+"\n")
    trace_manifest = trace_path.with_name("seq.json")
    write_json(trace_manifest, {"schema": "botsort_motion_diagnostics_v1", "status": "COMPLETE",
               "sequence": "seq", "records": 6, "data_sha256": sha256(trace_path)})
    protocol = {"status": "COMPLETE", "source_protocol_sha256": sha256(baseline/"protocol.json"),
                "sequences": {"seq": 6}, "detection_fingerprints": {"seq": detection_fingerprint(cache)},
                "rgb_encoder_sha256": "a"*64,
                "feature_hashes": {"seq": {"features": sha256(feature_path), "metadata": sha256(features/"seq.json")}},
                "diagnostics": {"status": "COMPLETE", "traces": {"seq": {
                    "manifest_sha256": sha256(trace_manifest), "data_sha256": sha256(trace_path),
                    "records": 6, "status": "COMPLETE"}}}}
    write_json(run_root/"protocols"/"fixture.json", protocol)
    write_csv(audit/"detection_audit.csv", rows)
    write_json(audit/"summary.json", {"status": "COMPLETE",
               "run_protocol_sha256": sha256(run_root/"protocols"/"fixture.json"),
               "files_sha256": {"detection_audit.csv": sha256(audit/"detection_audit.csv")}})
    return dict(run_root=run_root, tracker_name="fixture", baseline_root=baseline,
                features_dir=features, images_root=images, audit_dir=audit,
                output_dir=root/"d4", max_review=0)


class ObservationQualityTests(unittest.TestCase):
    def test_native_derivatives_keep_flat_zero_distinct_from_too_small_missing(self):
        image = np.full((10, 10, 3), 128, np.uint8)
        flat = native_roi_metrics(image, [1, 1, 6, 6])
        self.assertEqual(flat["gradient_rms"], 0.)
        self.assertEqual(flat["laplacian_variance"], 0.)
        self.assertIsNone(flat["gradient_axis_deg"])
        tiny = native_roi_metrics(image, [1, 1, 2, 2])
        self.assertIsNone(tiny["gradient_rms"])
        self.assertIsNone(tiny["laplacian_variance"])
        self.assertNotIn("motion_blur", flat)
        cropped = native_roi_metrics(image, [-2, 0, 2, 4])
        self.assertEqual(cropped["in_frame_area_fraction"], .5)
        outside = native_roi_metrics(image, [20, 20, 25, 25])
        self.assertEqual(outside["roi_status"], "outside_image")
        self.assertIsNone(outside["contrast_std"])

    def test_hard_negative_uses_the_same_previous_anchor(self):
        previous = {1: {"frame": 3, "vector": np.array([1., 0.])}}
        current = {1: {"det_index": 0, "vector": np.array([.6, .8])},
                   2: {"det_index": 1, "vector": np.array([0., 1.])}}
        result = appearance_evidence(4, current, previous)[1]
        self.assertAlmostEqual(result["same_gt_cosine_adjacent"], .4)
        self.assertAlmostEqual(result["hard_negative_cosine"], 1.)
        self.assertAlmostEqual(result["appearance_margin"], .6)
        self.assertEqual(result["hard_negative_gt_id"], 2)
        self.assertEqual(result["negative_count"], 1)
        self.assertEqual(previous[1]["frame"], 4)

    def test_nonconsecutive_frames_are_not_adjacent_pairs(self):
        previous = {1: {"frame": 1, "vector": np.array([1., 0.])}}
        result = appearance_evidence(4, {1: {"det_index": 0, "vector": np.array([1., 0.])}}, previous)[1]
        self.assertEqual(result["same_gt_gap_frames"], 3)
        self.assertEqual(result["same_gt_cosine_across_gap"], 0.)
        self.assertIsNone(result["same_gt_cosine_adjacent"])
        self.assertIsNone(result["appearance_margin"])
        self.assertEqual(result["appearance_status"], "nonconsecutive_gt_detection")

    def test_invalid_intervening_feature_cannot_be_skipped(self):
        previous = {1: {"frame": 1, "vector": np.array([1., 0.])}}
        invalid = appearance_evidence(2, {1: {"det_index": 0, "vector": None}}, previous)[1]
        self.assertEqual(invalid["appearance_status"], "missing_current_feature")
        valid = appearance_evidence(3, {1: {"det_index": 0, "vector": np.array([1., 0.])}}, previous)[1]
        self.assertEqual(valid["appearance_status"], "missing_previous_feature")
        self.assertEqual(valid["previous_same_gt_frame"], 2)
        self.assertIsNone(valid["same_gt_cosine_across_gap"])


class BlurEndToEndTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args = make_fixture(self.root)

    def run_d4(self, **kwargs):
        return analyze_blur(**(self.args | kwargs))

    def test_full_run_keeps_all_rows_and_missing_values_without_inferred_blur(self):
        result = self.run_d4()
        self.assertEqual(result["status"], "COMPLETE")
        source_root = Path(__file__).resolve().parents[2]
        for name in ("botsort_plus/diagnostics/blur.py", "botsort_plus/io.py", "botsort_plus/types.py"):
            self.assertEqual(result["offline_implementation_sha256"][name], sha256(source_root/name))
        self.assertEqual(result["overall"]["observations"], 11)
        self.assertFalse(result["quality_is_blur_label"])
        self.assertEqual(result["manual_label_status"], "NEEDS_MORE_COMPARABLE_MANUAL_LABELS")
        rows = {(int(row["frame"]), int(row["det_index"])): row
                for row in read_csv(self.args["output_dir"]/"observation_quality.csv")}
        self.assertEqual(rows[2, 0]["accepted_track_id"], "")
        self.assertAlmostEqual(float(rows[2, 0]["same_gt_cosine_adjacent"]), .2, places=6)
        self.assertEqual(rows[2, 0]["hard_negative_cosine"], "")
        self.assertEqual(rows[2, 1]["rgb_quality"], "")
        self.assertEqual(rows[2, 1]["rgb_quality_bin"], "missing")
        self.assertEqual(rows[3, 1]["same_gt_cosine_adjacent"], "")
        self.assertEqual(rows[5, 1]["same_gt_gap_frames"], "2")
        self.assertEqual(rows[5, 1]["same_gt_cosine_adjacent"], "")
        self.assertEqual(rows[6, 0]["same_gt_cosine_adjacent"], "0.0")
        self.assertEqual(rows[1, 2]["appearance_status"], "unknown_gt_identity")
        self.assertEqual(rows[1, 2]["det_center_error_norm"], "")
        self.assertTrue(all(row["human_label"] == "unreviewed" for row in rows.values()))
        self.assertEqual(read_csv(self.args["output_dir"]/"manual_label_comparisons.csv"), [])
        with self.assertRaises(FileExistsError):
            self.run_d4()

    def test_review_export_and_completed_labels_produce_controlled_comparison(self):
        self.run_d4(max_review=4)
        review_path = self.args["output_dir"]/"manual_review.csv"
        review = read_csv(review_path)
        self.assertEqual(len(review), 4)
        self.assertTrue(all(row["label"] == "unreviewed" for row in review))
        self.assertTrue(all((self.args["output_dir"]/row["contact_sheet"]).is_file() for row in review))
        protocol_hash = sha256(self.args["run_root"]/"protocols"/"fixture.json")
        manual = self.root/"completed_labels.csv"
        # Artificial labels exercise aggregation only; not a scientific blur claim.
        rows = [{"run_protocol_sha256": protocol_hash, "sequence": "seq", "frame": frame,
                 "det_index": 0, "label": label, "occlusion": "no", "reviewer": "unit test",
                 "note": "Synthetic fixture, no real blur evidence"}
                for frame, label in ((1, "clear"), (2, "motion_blur"))]
        write_csv(manual, rows, REVIEW_FIELDS)
        second = self.root/"d4_labelled"
        result = self.run_d4(output_dir=second, manual_labels=manual)
        self.assertEqual(result["manual_label_status"], "LABELLED_DESCRIPTIVE_ANALYSIS")
        comparison = next(row for row in read_csv(second/"manual_label_comparisons.csv")
                          if row["metric"] == "det_center_error_px")
        self.assertAlmostEqual(float(comparison["difference_blur_minus_clear"]), .1)
        self.assertEqual(comparison["clear_n"], "1")
        self.assertEqual(comparison["blur_n"], "1")
        self.assertEqual(comparison["gt_visibility_bin"], "ge0.9")
        obs = read_csv(second/"observation_quality.csv")
        self.assertEqual(next(r for r in obs if r["frame"] == "2" and r["det_index"] == "0")["previous_human_label"], "clear")

    def test_zero_based_gt_identity_is_preserved_in_full_diagnostic(self):
        audit_path = self.args["audit_dir"]/"detection_audit.csv"
        rows = read_csv(audit_path)
        for row in rows:
            if row["gt_id"] == "1":
                row["gt_id"] = "0"
        write_csv(audit_path, rows)
        summary_path = self.args["audit_dir"]/"summary.json"
        summary = json.loads(summary_path.read_text())
        summary["files_sha256"]["detection_audit.csv"] = sha256(audit_path)
        write_json(summary_path, summary)
        result = self.run_d4()
        self.assertEqual(result["status"], "COMPLETE")
        observed = {(int(row["frame"]), int(row["det_index"])): row
                    for row in read_csv(self.args["output_dir"]/"observation_quality.csv")}
        self.assertEqual(observed[2, 0]["gt_id"], "0")
        self.assertEqual(observed[2, 0]["gt_match_status"], "matched")
        self.assertAlmostEqual(float(observed[2, 0]["same_gt_cosine_adjacent"]), .2, places=6)
        self.assertEqual(observed[6, 1]["hard_negative_gt_id"], "0")

    def test_changed_original_image_rejected(self):
        cv2.imwrite(str(self.args["images_root"]/"seq"/"001.png"), np.zeros((32, 64, 3), np.uint8))
        with self.assertRaisesRegex(ValueError, "decoded image differs"):
            self.run_d4()
        summary = json.loads((self.args["output_dir"]/"summary.json").read_text())
        self.assertEqual(summary["status"], "FAILED")

    def test_changed_feature_bytes_rejected(self):
        with (self.args["features_dir"]/"seq.npz").open("ab") as stream:
            stream.write(b"tamper")
        with self.assertRaisesRegex(ValueError, "ReID cache differs"):
            self.run_d4()

    def test_changed_gt_audit_or_wrong_original_row_rejected(self):
        path = self.args["audit_dir"]/"detection_audit.csv"
        rows = read_csv(path)
        rows[0]["det_index"] = 1
        write_csv(path, rows)
        with self.assertRaisesRegex(ValueError, "audit differs"):
            self.run_d4()
        summary_path = self.args["audit_dir"]/"summary.json"
        summary = json.loads(summary_path.read_text())
        summary["files_sha256"]["detection_audit.csv"] = sha256(path)
        write_json(summary_path, summary)
        with self.assertRaisesRegex(ValueError, "row alignment mismatch"):
            self.run_d4()

    def test_unknown_manual_key_is_an_error(self):
        manual = self.root/"unknown.csv"
        write_csv(manual, [{"run_protocol_sha256": sha256(self.args["run_root"]/"protocols"/"fixture.json"),
                           "sequence": "seq", "frame": 999, "det_index": 0, "label": "clear", "occlusion": "no"}])
        with self.assertRaisesRegex(ValueError, "unknown detector rows"):
            self.run_d4(manual_labels=manual)

    def test_manual_labels_validate_hash_label_enum_and_duplicate_key(self):
        path = self.root/"labels.csv"
        base = {"run_protocol_sha256": "a"*64, "sequence": "seq", "frame": 1,
                "det_index": 0, "label": "clear", "occlusion": "no"}
        write_csv(path, [base])
        with self.assertRaisesRegex(ValueError, "different replay"):
            load_manual_labels(path, "b"*64)
        write_csv(path, [base | {"label": "probably_blur"}])
        with self.assertRaisesRegex(ValueError, "Unknown manual"):
            load_manual_labels(path, "a"*64)
        write_csv(path, [base, base])
        with self.assertRaisesRegex(ValueError, "Duplicate manual"):
            load_manual_labels(path, "a"*64)


if __name__ == "__main__":
    unittest.main()
