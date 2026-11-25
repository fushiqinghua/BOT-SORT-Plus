"""Read-only replay audits use small explicit MOT/event/protocol fixtures."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


CLI = Path(__file__).resolve().parents[2] / "tools/audit_botsort_plus_a2.py"
spec = importlib.util.spec_from_file_location("audit_a2", CLI)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def sha(data):
    return hashlib.sha256(data).hexdigest()


class A2AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.seq = "DJI_fixture"
        self.mot = b"5,1,10,20,4,4,0.2,-1,-1,-1\n6,1,10,20,4,4,0.2,-1,-1,-1\n"

    def write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")

    def events(self):
        result = []
        for frame in range(1, 7):
            result.append({"event": "frame", "frame": frame, "sequence": self.seq})
            if frame == 3:
                result.extend([
                    {"event": "birth_evidence", "frame": frame, "sequence": self.seq,
                     "candidate_id": 2, "reason": "verification_timeout"},
                    {"event": "birth_identity", "frame": frame, "sequence": self.seq,
                     "candidate_id": 2, "status": "DEFER"},
                    {"event": "birth_rejected", "frame": frame, "sequence": self.seq,
                     "candidate_id": 2, "reason": "identity_wait_exhausted"},
                ])
            if frame == 5:
                result.append({"event": "birth_proposed", "frame": frame, "sequence": self.seq,
                               "candidate_id": 1, "det_index": 0})
                commit = {"event": "birth_committed", "frame": frame, "sequence": self.seq,
                          "track_id": 1, "candidate_id": 1, "det_index": 0,
                          "candidate_first_frame": 1, "birth_frame": 5,
                          "confirmation_delay_frames": 4, "track_origin": "weak_birth"}
                result.append(commit)
                result.append({**commit, "event": "weak_birth", "alias_of": "birth_committed",
                               "evidence": 1.5, "evidence_role": "legacy_pool_priority_only"})
                result.append({"event": "first_emitted", "frame": frame, "sequence": self.seq,
                               "track_id": 1, "track_origin": "weak_birth", "birth_frame": 5})
            if frame == 6:
                result.append({"event": "recovered", "frame": frame, "sequence": self.seq,
                               "track_id": 1, "track_origin": "weak_birth"})
        result.append({"event": "track_lifetime", "frame": 6, "sequence": self.seq,
                       "track_id": 1, "origin": "weak_birth", "birth_frame": 5,
                       "candidate_id": 1, "candidate_first_frame": 1,
                       "first_emitted": 5, "last_emitted": 6, "output_frames": 2,
                       "recovery_count": 1, "native_high_match_count": 0})
        return result

    def write_run(self, name="A2", *, events=None, mot=None, instrumented=True):
        mot = self.mot if mot is None else mot
        prediction = self.root / "trackers" / name / "data" / (self.seq + ".txt")
        prediction.parent.mkdir(parents=True, exist_ok=True)
        prediction.write_bytes(mot)
        protocol = {
            "schema": "botsort_plus_run_v1", "name": name, "status": "COMPLETE",
            "sequences": {self.seq: 6}, "output_prediction_hashes": {self.seq: sha(mot)},
            "detector_sha256": "a" * 64, "rgb_encoder_sha256": "b" * 64,
            "source_protocol_sha256": "c" * 64,
            "feature_hashes": {self.seq: {"features": "d" * 64, "metadata": "e" * 64}},
            "detection_fingerprints": {self.seq: "f" * 64},
            "native_config": {"track_high_thresh": .5, "track_buffer": 60},
            "config": {"adapter_mode": "v2", "birth": {"enabled": instrumented}},
        }
        self.write_json(self.root / "protocols" / (name + ".json"), protocol)
        trace = self.root / "events" / name / (self.seq + ".jsonl")
        trace.parent.mkdir(parents=True, exist_ok=True)
        trace.write_text("".join(json.dumps(e) + "\n" for e in
                                 (self.events() if events is None else events)), encoding="utf-8")
        return protocol, prediction, trace

    def result(self, *names, pairs=()):
        return audit.audit_run(self.root, names or ("A2",), pairs)

    def test_valid_run_counts_canonical_birth_once(self):
        self.write_run()
        result = self.result()
        self.assertEqual(result["status"], "PASS", result)
        totals = result["trackers"]["A2"]["totals"]
        self.assertEqual(totals["births_committed"], 1)
        self.assertEqual(totals["compatibility_alias_events"], 1)
        self.assertEqual(totals["raw_event_counts"]["weak_birth"], 1)
        self.assertEqual(totals["verification_timeout_events"], 1)
        self.assertEqual(totals["identity_decisions_by_status"], {"DEFER": 1})
        self.assertEqual(totals["births_rejected_by_reason"], {"identity_wait_exhausted": 1})
        self.assertEqual(totals["weak_origin_recovery_events"], 1)
        weak = result["trackers"]["A2"]["sequences"][self.seq]["weak_tracks"][0]
        self.assertEqual(weak["output_frames"], 2)
        self.assertEqual(weak["lifespan_frames"], 2)
        self.assertEqual(weak["confirmation_delay_frames"], 4)

    def test_all_disabled_baseline_allows_empty_trace(self):
        self.write_run("baseline_ref", events=[], instrumented=False)
        result = self.result("baseline_ref")
        self.assertEqual(result["status"], "PASS", result)
        self.assertEqual(result["trackers"]["baseline_ref"]["totals"]["output_rows"], 2)

    def test_motion_output_does_not_count_as_a_measured_lifetime_frame(self):
        events = self.events()
        recovery = next(event for event in events if event["event"] == "recovered")
        recovery.update(event="motion_output", age=1, det_index=None)
        events[-1].update(last_emitted=5, output_frames=1, recovery_count=0)
        protocol, _, _ = self.write_run(events=events)
        protocol["config"]["recovery"] = {"enabled": True, "output_prediction_frames": 2}
        self.write_json(self.root / "protocols/A2.json", protocol)
        result = self.result()
        self.assertEqual(result["status"], "PASS", result)
        self.assertEqual(result["trackers"]["A2"]["totals"]["output_rows"], 2)

    def test_prediction_hash_mismatch_fails(self):
        _, path, _ = self.write_run()
        path.write_bytes(self.mot.replace(b"10,20", b"11,20"))
        result = self.result()
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("SHA-256", " ".join(result["trackers"]["A2"]["errors"]))

    def test_incomplete_protocol_fails(self):
        protocol, _, _ = self.write_run()
        protocol["status"] = "RUNNING"
        self.write_json(self.root / "protocols/A2.json", protocol)
        result = self.result()
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("COMPLETE", " ".join(result["trackers"]["A2"]["errors"]))

    def test_missing_trace_is_not_silently_accepted(self):
        _, _, trace = self.write_run()
        trace.unlink()
        result = self.result()
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("--trace", " ".join(result["trackers"]["A2"]["errors"]))

    def test_committed_track_must_appear_in_the_same_frame(self):
        self.write_run(mot=b"6,1,10,20,4,4,0.2,-1,-1,-1\n")
        result = self.result()
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("absent from same-frame", " ".join(result["trackers"]["A2"]["errors"]))

    def test_first_emitted_must_match_actual_first_row(self):
        events = self.events()
        first = next(x for x in events if x["event"] == "first_emitted")
        first["frame"] = 6
        events.sort(key=lambda x: x["frame"])
        self.write_run(events=events)
        result = self.result()
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("first_emitted differs", " ".join(result["trackers"]["A2"]["errors"]))

    def test_lifetime_rows_and_recovery_count_are_checked(self):
        for field, bad_value, diagnostic in (
                ("output_frames", 3, "output_frames differs"),
                ("recovery_count", 0, "recovery_count differs"),
                ("last_emitted", 5, "last_emitted differs")):
            with self.subTest(field=field):
                events = self.events()
                events[-1][field] = bad_value
                self.write_run(events=events)
                result = self.result()
                self.assertEqual(result["status"], "FAIL")
                self.assertIn(diagnostic, " ".join(result["trackers"]["A2"]["errors"]))

    def test_missing_final_lifetime_is_detected(self):
        self.write_run(events=[x for x in self.events() if x["event"] != "track_lifetime"])
        result = self.result()
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("track_lifetime", " ".join(result["trackers"]["A2"]["errors"]))

    def test_alias_mismatch_duplicate_and_missing_canonical_are_errors(self):
        for change in ("mismatch", "duplicate", "missing_canonical"):
            with self.subTest(change=change):
                events = self.events()
                alias = next(x for x in events if x["event"] == "weak_birth")
                if change == "mismatch":
                    alias["candidate_id"] = 99
                elif change == "duplicate":
                    events.insert(events.index(alias), dict(alias))
                else:
                    events = [x for x in events if x["event"] != "birth_committed"]
                self.write_run(events=events)
                result = self.result()
                self.assertEqual(result["status"], "FAIL")
                self.assertIn("alias", " ".join(result["trackers"]["A2"]["errors"]))

    def test_duplicate_mot_identity_same_frame_is_rejected(self):
        self.write_run(mot=self.mot + b"6,1,10,20,4,4,0.2,-1,-1,-1\n")
        result = self.result()
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("duplicate frame/track ID", " ".join(result["trackers"]["A2"]["errors"]))

    def test_truncated_per_frame_trace_is_detected(self):
        self.write_run(events=[x for x in self.events() if not (x["event"] == "frame" and x["frame"] == 2)])
        result = self.result()
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("frame trace records", " ".join(result["trackers"]["A2"]["errors"]))

    def test_shadow_comparison_checks_input_contract_then_exact_bytes(self):
        self.write_run("B_ref")
        protocol, prediction, _ = self.write_run("A2B_shadow")
        result = self.result("B_ref", "A2B_shadow", pairs=[("B_ref", "A2B_shadow")])
        self.assertEqual(result["status"], "PASS", result)
        self.assertEqual(result["comparisons"][0]["identical_sequences"], [self.seq])
        prediction.write_bytes(self.mot.replace(b"10,20", b"11,20"))
        protocol["output_prediction_hashes"][self.seq] = sha(prediction.read_bytes())
        self.write_json(self.root / "protocols/A2B_shadow.json", protocol)
        result = self.result("B_ref", "A2B_shadow", pairs=[("B_ref", "A2B_shadow")])
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["comparisons"][0]["different_sequences"], [self.seq])

    def test_identical_predictions_do_not_override_different_inputs(self):
        self.write_run("B_ref")
        protocol, _, _ = self.write_run("A2B_shadow")
        protocol["rgb_encoder_sha256"] = "0" * 64
        self.write_json(self.root / "protocols/A2B_shadow.json", protocol)
        result = self.result("B_ref", "A2B_shadow", pairs=[("B_ref", "A2B_shadow")])
        self.assertEqual(result["status"], "FAIL")
        comparison = result["comparisons"][0]
        self.assertEqual(comparison["input_mismatches"], ["rgb_encoder_sha256"])
        self.assertEqual(comparison["compared_sequences"], 0)

    def test_missing_shared_input_field_cannot_pass_as_equal_none(self):
        for name in ("B_ref", "A2B_shadow"):
            protocol, _, _ = self.write_run(name)
            del protocol["source_protocol_sha256"]
            self.write_json(self.root / "protocols" / (name + ".json"), protocol)
        result = self.result("B_ref", "A2B_shadow", pairs=[("B_ref", "A2B_shadow")])
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("source_protocol_sha256", result["comparisons"][0]["input_mismatches"])

    def test_invalid_sequence_contract_returns_failed_report_instead_of_crashing(self):
        for name in ("B_ref", "A2B_shadow"):
            protocol, _, _ = self.write_run(name)
            protocol["sequences"] = None
            self.write_json(self.root / "protocols" / (name + ".json"), protocol)
        result = self.result("B_ref", "A2B_shadow", pairs=[("B_ref", "A2B_shadow")])
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["comparisons"][0]["status"], "FAIL")

    def test_cli_writes_failed_report_exits_two_and_never_overwrites(self):
        self.write_run(events=[])
        output = self.root / "audit.json"
        cmd = [sys.executable, str(CLI), "--run-dir", str(self.root),
               "--trackers", "A2", "--output", str(output)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 2)
        report = json.loads(output.read_text())
        self.assertEqual(report["status"], "FAIL")
        before = output.read_bytes()
        second = subprocess.run(cmd, capture_output=True, text=True, check=False)
        self.assertEqual(second.returncode, 2)
        self.assertIn("already exists", second.stderr)
        self.assertEqual(output.read_bytes(), before)

    def test_run_audit_does_not_modify_or_access_ground_truth(self):
        self.write_run()
        gt = self.root / "gt.txt"
        gt.write_text("this is deliberately not a valid MOT file")
        before = {str(p.relative_to(self.root)): p.read_bytes()
                  for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(self.result()["status"], "PASS")
        after = {str(p.relative_to(self.root)): p.read_bytes()
                 for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_compare_names_and_path_traversal_are_rejected(self):
        self.write_run()
        with self.assertRaises(ValueError):
            self.result("../A2")
        with self.assertRaisesRegex(ValueError, "also appear"):
            self.result("A2", pairs=[("A2", "missing")])


if __name__ == "__main__":
    unittest.main()
