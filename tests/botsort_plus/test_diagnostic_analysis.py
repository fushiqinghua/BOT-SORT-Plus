"""Diagnostic evidence semantics, not fabricated accuracy/benefit claims."""
from copy import deepcopy
import csv
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from botsort_plus.diagnostics.analysis import analyze_run, recovery_windows, _frame_state, _output_status
from botsort_plus.diagnostics.common import (TRACE_SCHEMA, diagnostic_assignment, iter_trace,
                                              load_gt, read_json, sha256, squared_mahalanobis,
                                              write_json)


def snapshot(frame, x=0, last_seen=1, state=2):
    return {"track_id": 1, "state": state, "is_activated": True, "last_seen": last_seen,
            "det_index": 0, "mean": [x + 5, 5, 10, 10, 1, 0, 0, 0],
            "covariance": np.eye(8).tolist(), "xyxy": [x, 0, x + 10, 10],
            "projected_mean": [x + 5, 5, 10, 10],
            "projected_covariance": (np.eye(4) * 4).tolist()}


def make_records():
    records = []
    last_seen = 1
    for frame in range(1, 8):
        x = 2 * (frame - 1)
        have_det = frame != 6
        score = .05 if frame == 7 else .8
        detections = [{"det_index": 0, "xyxy": [x, 0, x + 10, 10], "score": score,
                       "rgb_quality": .9, "rgb_valid": True}] if have_det else []
        accepted = frame in (1, 3, 4, 5)
        events = []
        if accepted:
            kind = "native_birth" if frame == 1 else "recovered" if frame == 3 else "match"
            event = {"event": kind, "track_id": 1, "det_index": 0}
            if kind == "recovered":
                event.update(candidate_id=2, age=2, hits=2, cost=.2,
                             geometry_cost=.25, appearance_cost=.1)
            if kind == "match":
                event["stage"] = "high"
            events.append(event)
        predictions = [snapshot(frame, max(0, x - 2), last_seen)] if frame > 1 else []
        current = snapshot(frame, x if accepted else max(0, x - 2),
                           frame if accepted else last_seen, 1 if accepted else 2)
        output = [[x, 0, x + 10, 10, 1, score, 0, 0]] if accepted else []
        records.append({"schema": TRACE_SCHEMA, "sequence": "s1", "frame": frame,
                        "detections": detections, "before_prediction": deepcopy(predictions),
                        "after_prediction": deepcopy(predictions), "after_gmc": predictions,
                        "unconfirmed": [], "after_update": [current], "output": output,
                        "events": events, "gmc": {"status": "identity_ambiguous",
                            "quality_method": "deterministic_LK_replay_against_native_warp",
                            "warp": [[1, 0, 0], [0, 1, 0]], "inlier_ratio_proxy": .9,
                            "reprojection_median_px": .2, "spatial_coverage": .5},
                        "image_shape": [512, 640, 3], "image_dtype": "uint8",
                        "image_sha256": "a" * 64})
        if accepted:
            last_seen = frame
    return records


class AuditFixture:
    def __init__(self, root):
        self.root = root
        self.observed = root / "observed"
        self.reference = root / "reference"
        self.gt = root / "gt"
        self.output = root / "analysis"
        self.records = make_records()
        native = {"track_high_thresh": .6, "track_low_thresh": .1, "with_reid": True}
        metadata = {"detection_fingerprint": "d" * 64,
                    "feature_hashes": {"features": "e" * 64, "metadata": "f" * 64},
                    "native_config": native, "source_protocol_sha256": "c" * 64,
                    "implementation_sha256": {"botsort_plus/native_adapter.py": "1" * 64},
                    "boxmot_source_hashes": {"botsort.py": "2" * 64}}
        image_digest = hashlib.sha256()
        for record in self.records:
            image_digest.update((json.dumps({"frame": record["frame"],
                "shape": record["image_shape"], "dtype": record["image_dtype"],
                "sha256": record["image_sha256"]}, sort_keys=True,
                separators=(",", ":")) + "\n").encode("utf-8"))
        base = {"schema": "botsort_plus_run_v1", "status": "COMPLETE",
                "config": {"name": "b"}, "native_config": native, "sequences": {"s1": 7},
                "subset": True, "detector_sha256": "a" * 64, "rgb_encoder_sha256": "b" * 64,
                "source_protocol_sha256": "c" * 64,
                "detection_fingerprints": {"s1": "d" * 64},
                "feature_hashes": {"s1": metadata["feature_hashes"]},
                "encoder_contracts": {"rgb": {"sha256": "b" * 64}},
                "implementation_sha256": metadata["implementation_sha256"],
                "boxmot_source_hashes": metadata["boxmot_source_hashes"],
                "decoded_image_hashes": {"s1": image_digest.hexdigest()},
                "diagnostics": {"enabled": False, "status": "DISABLED"}}
        for run in (self.observed, self.reference):
            path = run / "trackers" / "b" / "data" / "s1.txt"
            path.parent.mkdir(parents=True)
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                for record in self.records:
                    for row in record["output"]:
                        writer.writerow([record["frame"], row[4], row[0], row[1],
                            row[2] - row[0], row[3] - row[1], row[5], -1, -1, -1])
            protocol = deepcopy(base)
            protocol["output_prediction_hashes"] = {"s1": sha256(path)}
            write_json(run / "protocols" / "b.json", protocol)
            events = run / "events" / "b" / "s1.jsonl"
            events.parent.mkdir(parents=True)
            events.write_text("".join(json.dumps({"sequence": "s1", "frame": r["frame"], **e}) + "\n"
                                       for r in self.records for e in r["events"]), encoding="utf-8")
        self.trace = self.observed / "diagnostics" / "b" / "s1.jsonl.gz"
        self.manifest = {"schema": TRACE_SCHEMA, "status": "COMPLETE", "sequence": "s1",
                         "records": 7, "expected_frames": 7, "trace_filename": self.trace.name,
                         "metadata": metadata}
        self.rewrite_trace()
        gt_path = self.gt / "s1" / "gt" / "gt.txt"
        gt_path.parent.mkdir(parents=True)
        gt_path.write_text("".join(f"{f},1,{2*(f-1)},0,10,10,1,7,0.8\n"
                                   for f in range(1, 8) if f != 4), encoding="utf-8")

    def rewrite_trace(self):
        self.trace.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(self.trace, "wt", encoding="utf-8") as stream:
            for row in self.records:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        self.manifest["data_sha256"] = sha256(self.trace)
        mpath = self.trace.with_suffix("").with_suffix(".json")
        write_json(mpath, self.manifest)
        ppath = self.observed / "protocols" / "b.json"
        protocol = read_json(ppath)
        protocol["diagnostics"] = {"enabled": True, "status": "COMPLETE", "schema": TRACE_SCHEMA,
            "traces": {"s1": {"records": 7, "status": "COMPLETE",
                "data_sha256": sha256(self.trace), "manifest_sha256": sha256(mpath)}}}
        write_json(ppath, protocol)

    def run(self):
        return analyze_run(self.observed, "b", self.gt, self.output,
                           reference_run_root=self.reference)


class DiagnosticAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def read_csv(self, path):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    def test_trace_accepts_native_boxmot_removed_state(self):
        fixture = AuditFixture(self.root)
        fixture.records[-1]["after_update"][0]["state"] = 4
        fixture.rewrite_trace()
        records = list(iter_trace(fixture.trace, "s1", len(fixture.records)))
        self.assertEqual(records[-1]["after_update"][0]["state"], 4)

    def test_assignment_rejects_forced_pairs_and_keeps_unique_threshold_edge(self):
        truth = [{"gt_id": 1, "xyxy": [0, 0, 10, 10]},
                 {"gt_id": 2, "xyxy": [30, 0, 40, 10]}]
        with patch("botsort_plus.diagnostics.common.box_iou_matrix",
                   return_value=np.array([[.5, .2], [.49, .49]])):
            labels, status, _, _ = diagnostic_assignment([[0]*4]*2, truth)
        self.assertEqual(labels, {0: 1})
        self.assertEqual(status, ["matched", "unmatched"])
        with patch("botsort_plus.diagnostics.common.box_iou_matrix",
                   return_value=np.array([[.9, .8], [.89, .1]])):
            _, status, _, assigned = diagnostic_assignment([[0]*4]*2, truth)
        self.assertEqual(assigned, {0: 1, 1: 0})  # maximum cardinality before total IoU
        self.assertEqual(status, ["ambiguous", "ambiguous"])

    def test_duplicates_are_ambiguous_not_two_identity_witnesses(self):
        labels, status, _, _ = diagnostic_assignment([[0, 0, 10, 10]] * 2,
                                                    [{"gt_id": 1, "xyxy": [0, 0, 10, 10]}])
        self.assertEqual(labels, {})
        self.assertEqual(status, ["ambiguous", "ambiguous"])

    def test_weak_duplicate_does_not_hide_a_reliable_usable_detection(self):
        truth = [{"gt_id": 1, "xyxy": [0, 0, 10, 10]}]
        det = [{"xyxy": [x, 0, x + 10, 10], "score": .8} for x in (0, 1)]
        labels, status, _, _ = diagnostic_assignment([d["xyxy"] for d in det], truth)
        self.assertEqual(labels, {0: 1})
        self.assertEqual(status, ["matched", "ambiguous"])
        state = _frame_state(2, truth, det, labels, status, [], {}, [], [],
                             {"track_high_thresh": .6, "track_low_thresh": .1}, .5)
        self.assertFalse(state["availability"][1]["ambiguous_det_available"])
        self.assertEqual(_output_status(state, 9, 1), "missing_with_usable_detection")

    def test_gt_pools_class_seven_and_preserves_empty_frames(self):
        path = self.root / "gt.txt"
        path.write_text("1,2,0,0,10,10,1,7,.2\n2,3,0,0,10,10,0,1,.8\n")
        truth, ignored = load_gt(path, 3)
        self.assertEqual(truth[1][0]["gt_class"], 7)
        self.assertEqual(truth[1][0]["gt_visibility"], .2)
        self.assertEqual(truth[2], [])
        self.assertEqual(truth[3], [])
        self.assertEqual(ignored, 1)
        path.write_text("1,2,0,0,10,10,1\n1,2,0,0,10,10,1\n")
        with self.assertRaisesRegex(ValueError, "duplicate GT"):
            load_gt(path, 3)

    def test_full_analysis_retains_failed_tracks_and_uses_only_past_identity(self):
        fixture = AuditFixture(self.root)
        result = fixture.run()
        self.assertEqual(result["status"], "COMPLETE")
        self.assertTrue(result["identical_original_event_bytes"])
        self.assertEqual(len(result["analysis_source_sha256"]), 3)
        self.assertEqual(result["D1"]["pred_center_error_px"]["n"], 4)
        track = {int(row["frame"]): row for row in self.read_csv(fixture.output / "track_error.csv")}
        self.assertEqual(track[2]["output_status"], "missing_with_usable_detection")
        self.assertEqual(track[2]["accepted_det_index"], "")
        self.assertAlmostEqual(float(track[2]["gt_paired_nis4"]), 1.)  # S=4I; no second R
        self.assertEqual(track[5]["anchor_status"], "unknown")  # f4 unannotated
        self.assertEqual(track[5]["accepted_gt_id"], "1")  # f5 cannot fix its own prior anchor
        self.assertEqual(track[6]["output_status"], "missing_without_usable_detection")
        self.assertEqual(track[7]["det_available"], "True")
        self.assertEqual(track[7]["usable_det_available"], "False")
        recovery = self.read_csv(fixture.output / "recovery_events.csv")[0]
        self.assertEqual(recovery["identity_status"], "same_gt")
        self.assertEqual(float(recovery["geometry_cost"]), .25)
        windows = {int(row["horizon"]): row for row in self.read_csv(fixture.output / "recovery_windows.csv")}
        self.assertEqual(windows[3]["unannotated_followup_frames"], "1")
        self.assertEqual(windows[3]["complete_observable_window"], "False")
        self.assertEqual(windows[3]["first_failure_offset"], "3")
        self.assertEqual(windows[5]["right_censored"], "True")
        self.assertEqual(windows[5]["stable_all_frames"], "")
        frames = self.read_csv(fixture.output / "frame_error.csv")
        self.assertEqual(frames[0]["quality_source"], "deterministic_LK_replay_against_native_warp")
        detections = self.read_csv(fixture.output / "detection_audit.csv")
        self.assertEqual([int(row["frame"]) for row in detections], [1, 2, 3, 4, 5, 7])
        self.assertEqual(detections[3]["gt_match_status"], "no_gt")
        self.assertEqual(result["files_sha256"]["detection_audit.csv"], sha256(fixture.output / "detection_audit.csv"))
        with self.assertRaises(FileExistsError):
            fixture.run()

    def test_end_censor_and_unverified_recovery_do_not_become_failures(self):
        state = {"frame": 3, "gt": {1: {}}, "availability": {1: {"usable_det_available": True,
            "ambiguous_det_available": False}}, "output_labels": {9: 1}, "gt_output": {1: 9},
            "output_status": {9: "matched"}}
        rec = {"frame": 2, "track_id": 9, "previous_gt_id": 1, "identity_status": "same_gt"}
        w = recovery_windows("s", rec, {3: state}, (3,))[0]
        self.assertTrue(w["right_censored"])
        self.assertIsNone(w["stable_all_frames"])
        self.assertIsNone(w["first_failure_reason"])
        rec["identity_status"] = "unknown"
        w = recovery_windows("s", rec, {3: state}, (1,))[0]
        self.assertFalse(w["complete_observable_window"])
        self.assertIsNone(w["stable_annotated_fraction"])

    def test_fully_observed_window_distinguishes_changed_id_from_detection_miss(self):
        rec = {"frame": 1, "track_id": 9, "previous_gt_id": 1, "identity_status": "same_gt"}
        states = {}
        for frame in (2, 3, 4):
            tid = 10 if frame == 4 else 9
            states[frame] = {"frame": frame, "gt": {1: {}}, "availability": {},
                "output_labels": {tid: 1}, "gt_output": {1: tid}, "output_status": {tid: "matched"}}
        result = recovery_windows("s", rec, states, (3,))[0]
        self.assertTrue(result["complete_observable_window"])
        self.assertFalse(result["stable_all_frames"])
        self.assertEqual(result["first_failure_offset"], 3)
        self.assertEqual(result["first_failure_reason"], "different_id")

    def test_no_reference_and_changed_images_are_rejected(self):
        fixture = AuditFixture(self.root)
        with self.assertRaisesRegex(ValueError, "independent"):
            analyze_run(fixture.observed, "b", fixture.gt, fixture.output)
        ppath = fixture.reference / "protocols" / "b.json"
        protocol = read_json(ppath)
        protocol["decoded_image_hashes"]["s1"] = "9" * 64
        write_json(ppath, protocol)
        with self.assertRaisesRegex(ValueError, "decoded_image_hashes"):
            fixture.run()
        self.assertFalse(fixture.output.exists())

    def test_equal_mot_with_changed_internal_events_is_not_observational_parity(self):
        fixture = AuditFixture(self.root)
        events = fixture.reference / "events" / "b" / "s1.jsonl"
        events.write_text(events.read_text() + '{"event":"unexpected"}\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "original tracking events"):
            fixture.run()
        self.assertFalse(fixture.output.exists())

    def test_changed_mot_is_rejected_even_with_refreshed_prediction_hash(self):
        fixture = AuditFixture(self.root)
        path = fixture.reference / "trackers" / "b" / "data" / "s1.txt"
        path.write_text(path.read_text().replace("0.8", "0.7", 1), encoding="utf-8")
        ppath = fixture.reference / "protocols" / "b.json"
        protocol = read_json(ppath)
        protocol["output_prediction_hashes"]["s1"] = sha256(path)
        write_json(ppath, protocol)
        with self.assertRaisesRegex(ValueError, "changed MOT"):
            fixture.run()
        self.assertFalse(fixture.output.exists())

    def test_trace_truncation_and_hash_tampering_fail_closed(self):
        fixture = AuditFixture(self.root)
        fixture.records.pop()
        fixture.rewrite_trace()
        with self.assertRaisesRegex(ValueError, "truncated diagnostic frames"):
            fixture.run()
        self.assertEqual(read_json(fixture.output / "summary.json")["status"], "FAILED")
        with fixture.trace.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            list(iter_trace(fixture.trace, "s1", 7))

    def test_empty_gt_has_null_denominators_and_nonfinite_gt_is_rejected(self):
        fixture = AuditFixture(self.root)
        (fixture.gt / "s1" / "gt" / "gt.txt").write_text("")
        result = fixture.run()
        self.assertEqual(result["D1"]["pred_center_error_px"]["n"], 0)
        self.assertIsNone(result["D1"]["pred_center_error_px"]["mean"])
        self.assertIsNone(result["D3"]["windows"]["3"]["stable_all_frames_fraction"])
        path = self.root / "bad_gt.txt"
        path.write_text("1,1,nan,0,10,10,1\n")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            load_gt(path, 1)
        self.assertIsNone(squared_mahalanobis([1, 1], [[0, 0], [0, 1]]))
        path.write_text('{"value":1e309}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Nonfinite JSON number"):
            read_json(path)


if __name__ == "__main__":
    unittest.main()
