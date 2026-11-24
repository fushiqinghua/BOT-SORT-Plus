#!/usr/bin/env python3
"""Read-only validation of completed A2 replay outputs and diagnostic events.

This tool never reads GT and never computes HOTA, IDF1, precision, or recall.
It checks output integrity, birth/lifecycle accounting, and optional byte-exact
shadow comparisons. A failed audit still writes the requested NEW report and
returns exit code 2. The report path must not already exist.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys


INPUT_FIELDS = (
    "sequences", "detector_sha256", "rgb_encoder_sha256", "source_protocol_sha256",
    "feature_hashes", "detection_fingerprints", "native_config",
)
_NAME = re.compile(r"^[A-Za-z0-9_-]+$")
_SHA = re.compile(r"^[0-9a-f]{64}$")


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _name(value):
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise ValueError(f"Expected a plain tracker/sequence name: {value!r}")
    return value


def _integer(value, label, minimum=0):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str) and re.fullmatch(r"[+-]?[0-9]+", value.strip()):
        result = int(value)
    else:
        raise ValueError(f"{label} must be an integer")
    if result < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return result


def _json_loads(value):
    def reject_constant(token):
        raise ValueError(f"Nonfinite JSON number: {token}")
    def finite_float(token):
        number = float(token)
        if not math.isfinite(number):
            raise ValueError(f"Nonfinite JSON number: {token}")
        return number
    return json.loads(value, parse_constant=reject_constant, parse_float=finite_float)


def _read_json(path):
    return _json_loads(Path(path).read_text(encoding="utf-8-sig"))


def _read_mot(path, frame_count):
    by_id, pairs, rows = defaultdict(list), set(), 0
    previous_frame = 0
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for line_number, row in enumerate(csv.reader(stream), 1):
            if not row:
                continue
            if len(row) != 10:
                raise ValueError(f"MOT line {line_number}: expected exactly 10 columns")
            frame = _integer(row[0], f"MOT line {line_number} frame", 1)
            tid = _integer(row[1], f"MOT line {line_number} track ID", 1)
            numbers = [float(x) for x in row[2:]]
            if not all(math.isfinite(x) for x in numbers):
                raise ValueError(f"MOT line {line_number}: nonfinite value")
            if numbers[2] <= 0 or numbers[3] <= 0:
                raise ValueError(f"MOT line {line_number}: box width/height must be positive")
            if not 0 <= numbers[4] <= 1:
                raise ValueError(f"MOT line {line_number}: invalid confidence")
            if frame > frame_count or frame < previous_frame:
                raise ValueError(f"MOT line {line_number}: frame outside protocol/order")
            previous_frame = frame
            if (frame, tid) in pairs:
                raise ValueError(f"MOT line {line_number}: duplicate frame/track ID")
            pairs.add((frame, tid))
            by_id[tid].append(frame)
            rows += 1
    return dict(by_id), pairs, rows


def _instrumented(protocol):
    cfg = protocol.get("config", {})
    return isinstance(cfg, dict) and cfg.get("adapter_mode") == "v2" and any(
        isinstance(cfg.get(key), dict) and cfg[key].get("enabled") is True
        for key in ("birth", "recovery", "appearance"))


def _audit_events(path, sequence, frame_count, by_id, pairs, instrumented):
    errors, commits, firsts, lifetimes = [], {}, {}, {}
    aliases, legacy_births, recovery_events = [], [], []
    counters, rejected, identities, frame_records = Counter(), Counter(), Counter(), Counter()
    high_matches, recovery_counts = Counter(), Counter()
    proposals = set()
    timeout_count, last_event_frame = 0, 0
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_number, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            try:
                event = _json_loads(raw)
                if not isinstance(event, dict) or not isinstance(event.get("event"), str):
                    raise ValueError("event must be a JSON object with an event name")
                if event.get("sequence") != sequence:
                    raise ValueError("event sequence differs from its file")
                frame = _integer(event.get("frame"), "event frame", 1)
                if frame > frame_count or frame < last_event_frame:
                    raise ValueError("event frame outside protocol/order")
                last_event_frame = frame
                event["frame"] = frame
                if "track_id" in event:
                    event["track_id"] = _integer(event["track_id"], "event track ID", 1)
                kind = event["event"]
                counters[kind] += 1
                if kind == "frame":
                    frame_records[frame] += 1
                elif kind == "birth_proposed":
                    cid = _integer(event.get("candidate_id"), "candidate ID", 1)
                    proposals.add((frame, cid, _integer(event.get("det_index"), "detection index")))
                elif kind == "birth_committed":
                    tid = _integer(event.get("track_id"), "committed track ID", 1)
                    if tid in commits:
                        errors.append(f"track {tid}: duplicate canonical birth_committed")
                    else:
                        commits[tid] = event
                elif kind == "weak_birth":
                    if event.get("alias_of") == "birth_committed":
                        aliases.append(event)
                    else:
                        legacy_births.append(event)
                elif kind == "birth_rejected":
                    reason = event.get("reason")
                    if not isinstance(reason, str) or not reason:
                        raise ValueError("birth_rejected lacks a reason")
                    rejected[reason] += 1
                elif kind == "birth_evidence":
                    if event.get("reason") == "verification_timeout":
                        timeout_count += 1
                elif kind == "birth_identity":
                    status = event.get("status")
                    if status not in ("CLEAR_NEW", "DEFER", "REJECT", "RECOVERED"):
                        raise ValueError("unknown birth_identity status")
                    identities[status] += 1
                elif kind == "first_emitted":
                    tid = _integer(event.get("track_id"), "first-emitted track ID", 1)
                    if tid in firsts:
                        errors.append(f"track {tid}: duplicate first_emitted")
                    else:
                        firsts[tid] = event
                elif kind == "track_lifetime":
                    tid = _integer(event.get("track_id"), "lifetime track ID", 1)
                    if tid in lifetimes:
                        errors.append(f"track {tid}: duplicate track_lifetime")
                    else:
                        lifetimes[tid] = event
                    if frame != frame_count:
                        errors.append(f"track {tid}: lifetime summary is not at sequence end")
                elif kind == "recovered":
                    tid = _integer(event.get("track_id"), "recovered track ID", 1)
                    recovery_counts[tid] += 1
                    recovery_events.append(event)
                elif kind == "match" and event.get("stage") in ("high", "unconfirmed"):
                    tid = _integer(event.get("track_id"), "matched track ID", 1)
                    high_matches[tid] += 1
            except (ValueError, TypeError, KeyError, OverflowError) as error:
                errors.append(f"event line {line_number}: {error}")

    if instrumented:
        missing = [i for i in range(1, frame_count + 1) if frame_records[i] != 1]
        if missing:
            errors.append(f"missing/duplicate frame trace records: {missing[:20]}"
                          f" ({len(missing)} affected frames)")
        if legacy_births:
            errors.append("v2 trace contains weak_birth events without canonical alias_of")

    alias_counts = Counter()
    for alias in aliases:
        try:
            tid = _integer(alias.get("track_id"), "alias track ID", 1)
            canonical = commits.get(tid)
            if canonical is None:
                raise ValueError(f"track {tid}: alias has no canonical birth_committed")
            for key, value in canonical.items():
                if key != "event" and alias.get(key) != value:
                    raise ValueError(f"track {tid}: compatibility alias differs in {key}")
            alias_counts[tid] += 1
            if alias_counts[tid] > 1:
                raise ValueError(f"track {tid}: duplicate compatibility alias")
        except (ValueError, TypeError) as error:
            errors.append(str(error))

    for tid, event in commits.items():
        try:
            frame = event["frame"]
            cid = _integer(event.get("candidate_id"), "committed candidate ID", 1)
            index = _integer(event.get("det_index"), "committed detection index")
            if (frame, cid, index) not in proposals:
                raise ValueError(f"track {tid}: committed without a same-frame proposal")
            if (frame, tid) not in pairs:
                raise ValueError(f"track {tid}: committed but absent from same-frame MOT output")
            if by_id[tid][0] != frame:
                raise ValueError(f"track {tid}: output precedes canonical birth commit")
            if event.get("birth_frame") != frame:
                raise ValueError(f"track {tid}: committed birth_frame mismatch")
            start = _integer(event.get("candidate_first_frame"), "candidate first frame", 1)
            delay = _integer(event.get("confirmation_delay_frames"), "confirmation delay")
            if start > frame or delay != frame - start:
                raise ValueError(f"track {tid}: inconsistent confirmation delay")
            if tid not in firsts:
                raise ValueError(f"track {tid}: committed without first_emitted")
            if tid not in lifetimes:
                raise ValueError(f"track {tid}: committed without final track_lifetime")
        except (ValueError, TypeError, KeyError) as error:
            errors.append(str(error))

    for tid, event in firsts.items():
        frames = by_id.get(tid, [])
        if not frames or event["frame"] != frames[0]:
            errors.append(f"track {tid}: first_emitted differs from MOT first frame")
        if tid not in lifetimes:
            errors.append(f"track {tid}: first_emitted lacks final track_lifetime")
        if event.get("track_origin") == "weak_birth" and tid not in commits:
            errors.append(f"track {tid}: weak first_emitted lacks canonical birth_committed")

    for tid, event in lifetimes.items():
        frames = by_id.get(tid, [])
        try:
            count = _integer(event.get("output_frames"), "lifetime output_frames")
            if count != len(frames):
                raise ValueError(f"track {tid}: lifetime output_frames differs from MOT rows")
            if event.get("first_emitted") != (frames[0] if frames else None):
                raise ValueError(f"track {tid}: lifetime first_emitted differs from MOT")
            if event.get("last_emitted") != (frames[-1] if frames else None):
                raise ValueError(f"track {tid}: lifetime last_emitted differs from MOT")
            if _integer(event.get("recovery_count"), "lifetime recovery_count") != recovery_counts[tid]:
                raise ValueError(f"track {tid}: lifetime recovery_count differs from events")
            if "native_high_match_count" in event and _integer(
                    event["native_high_match_count"], "native high match count") != high_matches[tid]:
                raise ValueError(f"track {tid}: native_high_match_count differs from events")
            if frames and tid not in firsts:
                raise ValueError(f"track {tid}: emitted lifetime lacks first_emitted event")
            if event.get("origin") == "weak_birth" and tid not in commits:
                raise ValueError(f"track {tid}: weak lifetime lacks canonical birth_committed")
            if tid in commits and event.get("origin") != "weak_birth":
                raise ValueError(f"track {tid}: committed weak track has inconsistent origin")
            if tid in firsts and event.get("origin") != firsts[tid].get("track_origin"):
                raise ValueError(f"track {tid}: lifetime and first-emitted origins differ")
        except (ValueError, TypeError) as error:
            errors.append(str(error))
    if instrumented:
        for tid in sorted(set(by_id) - set(lifetimes)):
            errors.append(f"track {tid}: MOT output lacks final track_lifetime")

    weak_recoveries = 0
    for event in recovery_events:
        tid = event["track_id"]
        origin = lifetimes.get(tid, {}).get("origin")
        if origin is None and tid in commits:
            origin = "weak_birth"
        if event.get("track_origin") is not None and origin is not None and event["track_origin"] != origin:
            errors.append(f"track {tid}: recovered event origin differs from lifetime")
        if origin == "weak_birth":
            weak_recoveries += 1

    weak_tracks = []
    for tid, commit in sorted(commits.items()):
        frames = by_id.get(tid, [])
        weak_tracks.append({
            "track_id": tid, "candidate_id": commit.get("candidate_id"),
            "candidate_first_frame": commit.get("candidate_first_frame"),
            "birth_frame": commit.get("birth_frame"),
            "first_emitted": frames[0] if frames else None,
            "last_emitted": frames[-1] if frames else None,
            "output_frames": len(frames),
            "lifespan_frames": frames[-1] - frames[0] + 1 if frames else 0,
            "recovery_count": recovery_counts[tid],
            "confirmation_delay_frames": commit.get("confirmation_delay_frames"),
        })
    return {
        "lifecycle_instrumented": instrumented, "raw_event_counts": dict(sorted(counters.items())),
        "births_proposed": counters["birth_proposed"],
        "births_committed": len(commits),
        "compatibility_alias_events": len(aliases),
        "legacy_weak_birth_events": len(legacy_births),
        "births_rejected_by_reason": dict(sorted(rejected.items())),
        "first_emitted_events": len(firsts), "track_lifetime_events": len(lifetimes),
        "verification_timeout_events": timeout_count,
        "identity_decisions_by_status": dict(sorted(identities.items())),
        "weak_origin_recovery_events": weak_recoveries,
        "weak_tracks": weak_tracks, "errors": errors,
    }


def audit_tracker(root, name):
    result = {"status": "FAIL", "errors": [], "sequences": {}}
    protocol = None
    try:
        protocol = _read_json(root / "protocols" / (name + ".json"))
        if not isinstance(protocol, dict) or protocol.get("status") != "COMPLETE":
            raise ValueError("Protocol is missing or not COMPLETE")
        if protocol.get("schema") != "botsort_plus_run_v1" or protocol.get("name") != name:
            raise ValueError("Protocol schema/name mismatch")
        sequences = protocol.get("sequences")
        hashes = protocol.get("output_prediction_hashes")
        if not isinstance(sequences, dict) or not sequences:
            raise ValueError("Protocol sequences must be a nonempty frame-count object")
        if not isinstance(hashes, dict) or set(hashes) != set(sequences):
            raise ValueError("Prediction hash sequence set differs from protocol")
        for seq, count in sequences.items():
            _name(seq)
            _integer(count, f"{seq} frame count", 1)
            if not isinstance(hashes[seq], str) or not _SHA.fullmatch(hashes[seq]):
                raise ValueError(f"Invalid prediction SHA-256 for {seq}")
        instrumented = _instrumented(protocol)
        for seq, frames in sorted(sequences.items()):
            entry = {"status": "FAIL", "frames": frames, "errors": []}
            result["sequences"][seq] = entry
            try:
                prediction = root / "trackers" / name / "data" / (seq + ".txt")
                actual_hash = sha256(prediction)
                entry["prediction_sha256"] = actual_hash
                if actual_hash != hashes[seq]:
                    entry["errors"].append("Prediction bytes differ from COMPLETE protocol SHA-256")
                by_id, pairs, rows = _read_mot(prediction, frames)
                entry.update(output_rows=rows, output_track_ids=len(by_id))
                trace = root / "events" / name / (seq + ".jsonl")
                if not trace.is_file():
                    raise ValueError("Missing event JSONL; replay this run with --trace")
                events = _audit_events(trace, seq, frames, by_id, pairs, instrumented)
                entry["errors"].extend(events.pop("errors"))
                entry.update(events)
                entry["status"] = "PASS" if not entry["errors"] else "FAIL"
            except (OSError, ValueError, TypeError, KeyError, OverflowError) as error:
                entry["errors"].append(str(error))
            result["errors"].extend(f"{seq}: {error}" for error in entry["errors"])
    except (OSError, ValueError, TypeError, KeyError) as error:
        result["errors"].append(str(error))
    totals = Counter()
    aggregate_counters = {k: Counter() for k in
                          ("raw_event_counts", "births_rejected_by_reason", "identity_decisions_by_status")}
    for entry in result["sequences"].values():
        for key in ("output_rows", "output_track_ids", "births_proposed", "births_committed",
                    "compatibility_alias_events", "legacy_weak_birth_events", "first_emitted_events",
                    "track_lifetime_events", "verification_timeout_events", "weak_origin_recovery_events"):
            totals[key] += entry.get(key, 0)
        for key, counter in aggregate_counters.items():
            counter.update(entry.get(key, {}))
    result["totals"] = dict(totals)
    result["totals"].update({k: dict(sorted(v.items())) for k, v in aggregate_counters.items()})
    result["status"] = "PASS" if not result["errors"] else "FAIL"
    return result, protocol


def compare_pair(root, left, right, protocols):
    result = {"left": left, "right": right, "status": "FAIL", "errors": [],
              "input_fields_checked": list(INPUT_FIELDS), "input_mismatches": [],
              "compared_sequences": 0, "identical_sequences": [], "different_sequences": []}
    a, b = protocols.get(left), protocols.get(right)
    if not isinstance(a, dict) or not isinstance(b, dict):
        result["errors"].append("Cannot compare without both readable protocols")
        return result
    for name, proto in ((left, a), (right, b)):
        if proto.get("status") != "COMPLETE":
            result["errors"].append(f"{name}: protocol is not COMPLETE")
        if not isinstance(proto.get("sequences"), dict) or not proto["sequences"]:
            result["errors"].append(f"{name}: invalid sequence contract")
    if result["errors"]:
        return result
    for field in INPUT_FIELDS:
        if field not in a or field not in b or a[field] != b[field]:
            result["input_mismatches"].append(field)
    for field in ("feature_hashes", "detection_fingerprints"):
        for name, proto in ((left, a), (right, b)):
            values = proto.get(field)
            if not isinstance(values, dict) or set(values) != set(proto.get("sequences", {})):
                result["errors"].append(f"{name}: incomplete {field}")
    if result["input_mismatches"] or result["errors"]:
        result["errors"].append("Input contracts differ or are incomplete; no shadow equality claim")
        return result
    sequences = a["sequences"]
    if not isinstance(sequences, dict):
        result["errors"].append("Invalid sequence contract")
        return result
    for seq in sorted(sequences):
        try:
            _name(seq)
            pa = root / "trackers" / left / "data" / (seq + ".txt")
            pb = root / "trackers" / right / "data" / (seq + ".txt")
            equal = pa.read_bytes() == pb.read_bytes()
            result["compared_sequences"] += 1
            result["identical_sequences" if equal else "different_sequences"].append(seq)
        except (OSError, ValueError) as error:
            result["errors"].append(f"{seq}: {error}")
    if result["different_sequences"]:
        result["errors"].append("Prediction files are not byte-identical")
    result["status"] = "PASS" if not result["errors"] else "FAIL"
    return result


def audit_run(run_dir, trackers, compare_pairs=()):
    root = Path(run_dir).resolve()
    if not root.is_dir():
        raise ValueError(f"Run directory does not exist: {root}")
    trackers = [_name(name) for name in trackers]
    if len(set(trackers)) != len(trackers) or not trackers:
        raise ValueError("Tracker names must be unique and nonempty")
    for pair in compare_pairs:
        if len(pair) != 2 or not set(pair).issubset(trackers):
            raise ValueError("Every --compare-pair name must also appear in --trackers")
    result = {"schema": "botsort_plus_a2_audit_v1", "status": "FAIL", "run_directory": str(root),
              "scope": "Output/log integrity only; no GT is read and no tracking accuracy is computed.",
              "birth_counting": "birth_committed is canonical; weak_birth aliases are not added to it.",
              "trackers": {}, "comparisons": []}
    protocols = {}
    for name in trackers:
        result["trackers"][name], protocols[name] = audit_tracker(root, name)
    for left, right in compare_pairs:
        result["comparisons"].append(compare_pair(root, left, right, protocols))
    result["error_count"] = (sum(len(r["errors"]) for r in result["trackers"].values())
                             + sum(len(r["errors"]) for r in result["comparisons"]))
    result["status"] = "PASS" if not result["error_count"] else "FAIL"
    return result


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--run-dir", type=Path, required=True)
    value.add_argument("--trackers", nargs="+", required=True)
    value.add_argument("--compare-pair", nargs=2, action="append", default=[], metavar=("LEFT", "RIGHT"))
    value.add_argument("--output", type=Path, required=True, help="New JSON report; existing files are refused")
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError(f"Report already exists; choose a new --output: {args.output}")
        output = args.output.resolve()
        root = args.run_dir.resolve()
        if any(output.is_relative_to(root / section) for section in ("protocols", "trackers", "events", "timing")):
            raise ValueError("Report must be outside the run's protocol/prediction/event/timing directories")
        report = audit_run(args.run_dir, args.trackers, args.compare_pair)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"status": report["status"], "error_count": report["error_count"],
                          "report": str(output)}, ensure_ascii=False))
        return 0 if report["status"] == "PASS" else 2
    except (OSError, ValueError, TypeError) as error:
        print(f"Audit failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
