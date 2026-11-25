"""C2 audit fixtures check honest accounting, not tracking accuracy."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from botsort_plus.config import PlusConfig
from tools import audit_botsort_plus_identity as audit


CLI = Path(__file__).resolve().parents[2] / "tools/audit_botsort_plus_identity.py"


class IdentityAuditTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.seq = "DJI_fixture"

    def config(self, name, enabled=True, shadow=False):
        return PlusConfig.from_dict({
            "name": name, "adapter_mode": "v2", "recovery": {"enabled": True},
            "identity": {"enabled": enabled, "observe_only": shadow},
        }).to_dict()

    def events(self, enabled=True, shadow=False):
        result = []
        for frame in range(1, 6):
            def emit(kind, **values):
                result.append({"event": kind, "frame": frame, "sequence": self.seq, **values})
            emit("frame")
            if enabled:
                for stage in ("high", "unconfirmed"):
                    pairs = int(stage == "high" and frame in (2, 3))
                    emit("identity_association", stage=stage, pair_count=pairs,
                         applied=not shadow, proposed_changed_pairs=0,
                         proposed_max_cost_increase=0., cold_reference_pairs=pairs)
            if frame == 1:
                emit("memory_update", track_id=1, det_index=0, modalities=["rgb"])
                if enabled:
                    emit("identity_seed", track_id=1, det_index=0, provisional=True,
                         private_anchor_written=True, would_allow_write=True, observe_only=shadow,
                         memory_gate_applied=not shadow, native_feature_present=True,
                         custom_modalities_written=["rgb"])
                emit("native_birth", track_id=1, det_index=0)
                emit("first_emitted", track_id=1, track_origin="native", birth_frame=1)
            else:
                stage = "high" if frame < 4 else "low" if frame == 4 else "recovery"
                allow = frame < 4
                custom = ["rgb"] if allow or shadow or not enabled else []
                native = stage != "low" and (allow or shadow or not enabled)
                if custom:
                    emit("memory_update", track_id=1, det_index=0, modalities=custom)
                if enabled:
                    emit("identity_decision", stage=stage, track_id=1, det_index=0,
                         reliability=None if frame < 4 else 0. if frame == 4 else 1.,
                         allow_write=allow,
                         reason="warmup_geometry_verified" if allow else
                                "history_conflict" if frame == 4 else "recovered_quarantine",
                         distance=1. if frame == 4 else 0., appearance_margin=None,
                         geometry_margin=None, competitor_count=0, competition_known=False,
                         history_updates=min(frame-1, 3), quality=1., geometry_cost=.1,
                         observe_only=shadow, association_applied=not shadow and stage == "high",
                         memory_gate_applied=not shadow, native_feature_written=native,
                         custom_modalities_written=custom, private_anchor_written=allow,
                         reference_cost=.1, applied_cost=.1,
                         base_cost=None if stage == "recovery" else .1)
                emit("recovered" if stage == "recovery" else "match", track_id=1, det_index=0,
                     **({} if stage == "recovery" else {"stage": stage}), track_origin="native")
        result.append({"event": "track_lifetime", "frame": 5, "sequence": self.seq,
                       "track_id": 1, "origin": "native", "birth_frame": 1,
                       "first_emitted": 1, "last_emitted": 5, "output_frames": 5,
                       "recovery_count": 1, "native_high_match_count": 2})
        return result

    def write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def write_run(self, name="B_C2", enabled=True, shadow=False, events=None):
        prediction = self.root / "trackers" / name / "data" / (self.seq + ".txt")
        prediction.parent.mkdir(parents=True, exist_ok=True)
        mot = "".join(f"{i},1,10,20,5,6,0.9,-1,-1,-1\n" for i in range(1, 6)).encode()
        prediction.write_bytes(mot)
        protocol = {
            "schema": "botsort_plus_run_v1", "status": "COMPLETE", "name": name,
            "config": self.config(name, enabled, shadow), "sequences": {self.seq: 5},
            "output_prediction_hashes": {self.seq: hashlib.sha256(mot).hexdigest()},
            "detector_sha256": "a" * 64, "rgb_encoder_sha256": "b" * 64,
            "source_protocol_sha256": "c" * 64,
            "feature_hashes": {self.seq: {"features": "d" * 64, "metadata": "e" * 64}},
            "detection_fingerprints": {self.seq: "f" * 64},
            "native_config": {"track_high_thresh": .5, "track_buffer": 60},
        }
        protocol_path = self.root / "protocols" / (name + ".json")
        self.write_json(protocol_path, protocol)
        trace = self.root / "events" / name / (self.seq + ".jsonl")
        trace.parent.mkdir(parents=True, exist_ok=True)
        rows = self.events(enabled, shadow) if events is None else events
        trace.write_text("".join(json.dumps(e) + "\n" for e in rows), encoding="utf-8")
        return protocol, protocol_path, trace

    def result(self, *names, pairs=()):
        return audit.audit_run(self.root, names or ("B_C2",), pairs)

    def test_complete_trace_reports_stage_counts_and_numeric_population(self):
        self.write_run()
        report = self.result()
        self.assertEqual(report["status"], "PASS", report)
        stages = report["trackers"]["B_C2"]["stages"]
        self.assertEqual(stages["high"]["decisions"], 2)
        self.assertEqual(stages["high"]["reliability"]["reference_fallback_fraction"], 1.)
        self.assertIsNone(stages["high"]["reliability"]["mean"])
        self.assertEqual(stages["low"]["reliability"]["histogram"]["[0,0.1)"], 1)
        self.assertEqual(stages["recovery"]["reasons"], {"recovered_quarantine": 1})
        self.assertEqual(stages["unconfirmed"]["decisions"], 0)
        self.assertIsNone(stages["unconfirmed"]["allow_write_fraction"])
        json.dumps(report, allow_nan=False)

    def test_missing_or_empty_trace_is_not_zero_evidence(self):
        _, _, trace = self.write_run()
        for content in (None, ""):
            if content is None:
                trace.unlink()
            else:
                trace.write_text(content, encoding="utf-8")
            report = self.result()
            self.assertEqual(report["status"], "FAIL")
            entry = report["trackers"]["B_C2"]
            self.assertFalse(entry["trace_complete"])
            self.assertIsNone(entry["stages"])

    def test_disabled_reference_has_no_identity_statistics(self):
        self.write_run("B_ref", enabled=False)
        report = self.result("B_ref")
        self.assertEqual(report["status"], "PASS", report)
        self.assertEqual(report["trackers"]["B_ref"]["status"], "NOT_ENABLED")
        self.assertIsNone(report["trackers"]["B_ref"]["stages"])

    def test_low_native_writer_and_recovery_private_write_are_rejected(self):
        for stage, field in (("low", "native_feature_written"), ("recovery", "private_anchor_written")):
            events = self.events()
            event = next(e for e in events if e["event"] == "identity_decision" and e["stage"] == stage)
            event[field] = True
            self.write_run(events=events)
            report = self.result()
            self.assertEqual(report["status"], "FAIL", report)

    def test_shadow_must_preserve_old_writers_even_when_private_gate_rejects(self):
        self.write_run("B_ref", enabled=False)
        self.write_run("B_C2_shadow", shadow=True)
        report = self.result("B_ref", "B_C2_shadow", pairs=(("B_ref", "B_C2_shadow"),))
        self.assertEqual(report["status"], "PASS", report)
        stages = report["trackers"]["B_C2_shadow"]["stages"]
        self.assertEqual(stages["recovery"]["native_feature_writes"], 1)
        self.assertEqual(stages["recovery"]["custom_memory_writes"], 1)
        self.assertEqual(stages["recovery"]["private_anchor_writes"], 0)
        self.assertEqual(stages["low"]["custom_memory_writes"], 1)
        self.assertEqual(stages["low"]["native_feature_writes"], 0)

    def test_wrong_runtime_flags_and_missing_numeric_fields_fail(self):
        for change in (lambda e: e.update(observe_only=True),
                       lambda e: e.update(memory_gate_applied=False),
                       lambda e: e.update(association_applied=False),
                       lambda e: e.pop("reliability"),
                       lambda e: e.update(reason="made_up"),
                       lambda e: e.update(allow_write="true")):
            events = self.events()
            change(next(e for e in events if e["event"] == "identity_decision"))
            self.write_run(events=events)
            self.assertEqual(self.result()["status"], "FAIL")

    def test_nonfinite_json_is_rejected(self):
        events = self.events()
        next(e for e in events if e["event"] == "identity_decision")["reliability"] = float("nan")
        _, _, trace = self.write_run(events=events)
        self.assertEqual(self.result()["status"], "FAIL")
        trace.write_text(trace.read_text().replace("NaN", "1e999"), encoding="utf-8")
        self.assertEqual(self.result()["status"], "FAIL")

    def test_configuration_identity_and_name_mismatches_fail(self):
        for path_keys, value in ((("identity", "enabled"), False),
                                 (("identity", "enabled"), "true"),
                                 (("name",), "wrong"),
                                 (("identity", "unexpected"), 1)):
            protocol, path, _ = self.write_run()
            current = protocol["config"]
            for key in path_keys[:-1]:
                current = current[key]
            current[path_keys[-1]] = value
            self.write_json(path, protocol)
            self.assertEqual(self.result()["status"], "FAIL")

    def test_comparison_requires_same_inputs_and_foundation_config(self):
        self.write_run("B_ref", enabled=False)
        original, path, _ = self.write_run("B_C2_shadow", shadow=True)
        for case in ("inputs", "foundation", "not_shadow"):
            protocol = copy.deepcopy(original)
            if case == "inputs":
                protocol["detection_fingerprints"][self.seq] = "0" * 64
            elif case == "foundation":
                protocol["config"]["recovery"]["geometry_weight"] = .7
            else:
                protocol["config"]["identity"]["observe_only"] = False
            self.write_json(path, protocol)
            report = self.result("B_ref", "B_C2_shadow", pairs=(("B_ref", "B_C2_shadow"),))
            self.assertEqual(report["status"], "FAIL", case)

    def test_missing_decisions_associations_or_write_logs_fail(self):
        for kind in ("identity_decision", "identity_association", "memory_update", "identity_seed"):
            events = self.events()
            events.remove(next(e for e in events if e["event"] == kind))
            self.write_run(events=events)
            self.assertEqual(self.result()["status"], "FAIL", kind)

    def test_custom_memory_claim_must_have_matching_memory_update(self):
        events = self.events()
        next(e for e in events if e["event"] == "identity_decision")["custom_modalities_written"] = []
        self.write_run(events=events)
        self.assertEqual(self.result()["status"], "FAIL")

    def test_unconfirmed_bounds_use_fused_base_not_raw_iou(self):
        events = self.events()
        decision = next(e for e in events if e["event"] == "identity_decision" and e["frame"] == 3)
        decision.update(stage="unconfirmed", history_updates=3, reliability=.5,
                        reference_cost=.05, applied_cost=.275, base_cost=.5,
                        geometry_cost=.1, reason="identity_verified")
        next(e for e in events if e["event"] == "match" and e["frame"] == 3)["stage"] = "unconfirmed"
        for event in events:
            if event["event"] == "identity_association" and event["frame"] == 3:
                unc = event["stage"] == "unconfirmed"
                event.update(pair_count=int(unc), proposed_changed_pairs=int(unc),
                             cold_reference_pairs=0, proposed_max_cost_increase=.225 if unc else 0.)
        self.write_run(events=events)
        self.assertEqual(self.result()["status"], "PASS")
        decision["applied_cost"] = .6
        self.write_run(events=events)
        self.assertEqual(self.result()["status"], "FAIL")

    def test_cost_roundoff_does_not_create_false_audit_failure(self):
        events = self.events()
        event = next(e for e in events if e["event"] == "identity_association")
        event["proposed_max_cost_increase"] = -1e-17
        self.write_run(events=events)
        self.assertEqual(self.result()["status"], "PASS")

    def test_cli_writes_new_report_and_refuses_overwrite_or_evidence_path(self):
        self.write_run()
        output = self.root / "audit.json"
        args = [sys.executable, str(CLI), "--run-dir", str(self.root),
                "--trackers", "B_C2", "--output", str(output)]
        first = subprocess.run(args, capture_output=True, text=True)
        self.assertEqual(first.returncode, 0, first.stderr + first.stdout)
        previous = output.read_bytes()
        second = subprocess.run(args, capture_output=True, text=True)
        self.assertEqual(second.returncode, 2)
        self.assertEqual(output.read_bytes(), previous)
        args[-1] = str(self.root / "events" / "forbidden.json")
        third = subprocess.run(args, capture_output=True, text=True)
        self.assertEqual(third.returncode, 2)
        self.assertFalse(Path(args[-1]).exists())


if __name__ == "__main__":
    unittest.main()
