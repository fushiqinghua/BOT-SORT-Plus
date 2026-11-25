#!/usr/bin/env python3
"""Audit C2 replay logs and summarize intervention rates, without reading GT.

PASS means output/protocol integrity and log consistency, not correct identities
or measured tracking gains. Missing/incomplete traces never become zero counts.
All reliability summaries describe accepted matches, not all candidate pairs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from botsort_plus.config import PlusConfig
from tools.audit_botsort_plus_a2 import audit_run as audit_integrity, _json_loads, _name, _read_json


STAGES = ("high", "low", "unconfirmed", "recovery")
REASONS = {
    "outside_geometry", "missing_feature", "low_quality", "warmup_requires_high",
    "history_conflict", "warmup_geometry_ambiguous", "warmup_geometry_verified",
    "write_geometry_weak", "competition_unknown", "appearance_ambiguous",
    "singleton_geometry_weak", "identity_verified", "recovered_quarantine",
}
HISTOGRAM = ((.1, "[0,0.1)"), (.25, "[0.1,0.25)"), (.5, "[0.25,0.5)"),
             (.75, "[0.5,0.75)"), (1.01, "[0.75,1]"))


def _int(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value, label, lower, upper, *, nullable=False):
    if nullable and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    if not lower <= value <= upper:
        raise ValueError(f"{label} must lie in [{lower},{upper}]")
    return float(value)


def _flag(event, key, expected=None):
    value = event.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be boolean")
    if expected is not None and value != expected:
        raise ValueError(f"{key} disagrees with the protocol/configuration")
    return value


def _modalities(value):
    if not isinstance(value, list) or any(x != "rgb" for x in value) or len(value) > 1:
        raise ValueError("C2 custom writes must be [] or ['rgb']")
    return value


def _key(event):
    return (_int(event.get("frame"), "frame", 1),
            _int(event.get("track_id"), "track_id", 1),
            _int(event.get("det_index"), "det_index"))


def _empty_stats():
    return {stage: {"counts": Counter(), "reasons": Counter(), "r_sum": 0.,
                    "histogram": Counter()} for stage in STAGES}


def _add_decision(stats, event):
    entry = stats[event["stage"]]
    counts = entry["counts"]
    counts["decisions"] += 1
    for key in ("allow_write", "private_anchor_written", "native_feature_written"):
        counts[key] += int(event[key])
    counts["custom_memory_written"] += int(bool(event["custom_modalities_written"]))
    counts["memory_gate_applied"] += int(event["memory_gate_applied"])
    entry["reasons"][event["reason"]] += 1
    r = event["reliability"]
    if r is None:
        counts["reference_fallback"] += 1
    else:
        counts["numeric_reliability"] += 1
        counts["r_below_0_1"] += int(r < .1)
        entry["r_sum"] += r
        for upper, label in HISTOGRAM:
            if r < upper:
                entry["histogram"][label] += 1
                break


def _summary(stats):
    output = {}
    for stage, entry in stats.items():
        c = entry["counts"]
        total, numeric = c["decisions"], c["numeric_reliability"]
        output[stage] = {
            "decisions": total, "allow_write": c["allow_write"],
            "allow_write_fraction": c["allow_write"] / total if total else None,
            "private_anchor_writes": c["private_anchor_written"],
            "native_feature_writes": c["native_feature_written"],
            "custom_memory_writes": c["custom_memory_written"],
            "memory_gate_applied": c["memory_gate_applied"],
            "reasons": dict(sorted(entry["reasons"].items())),
            "reliability": {
                "numeric_count": numeric, "reference_fallback_count": c["reference_fallback"],
                "reference_fallback_fraction": c["reference_fallback"] / total if total else None,
                "mean": entry["r_sum"] / numeric if numeric else None,
                "below_0_1_fraction": c["r_below_0_1"] / numeric if numeric else None,
                "histogram": {label: entry["histogram"][label] for _, label in HISTOGRAM},
            },
        }
    return output


def _validate_decision(event, cfg):
    required = {"reliability", "distance", "appearance_margin", "geometry_margin", "base_cost"}
    if required - set(event):
        raise ValueError(f"Identity decision missing fields: {sorted(required - set(event))}")
    _key(event)
    stage = event.get("stage")
    if stage not in STAGES or event.get("reason") not in REASONS:
        raise ValueError("Unknown identity decision stage/reason")
    allow = _flag(event, "allow_write")
    private = _flag(event, "private_anchor_written")
    native = _flag(event, "native_feature_written")
    custom = _modalities(event.get("custom_modalities_written"))
    gate = cfg.memory_enabled and not cfg.observe_only
    _flag(event, "observe_only", cfg.observe_only)
    _flag(event, "memory_gate_applied", gate)
    applied = cfg.association_enabled and not cfg.observe_only and stage in ("high", "unconfirmed")
    _flag(event, "association_applied", applied)
    r = _number(event.get("reliability"), "reliability", 0., 1., nullable=True)
    quality = _number(event.get("quality"), "quality", 0., 1.)
    _number(event.get("distance"), "distance", 0., 2., nullable=True)
    _number(event.get("appearance_margin"), "appearance_margin", -2., 2., nullable=True)
    _number(event.get("geometry_margin"), "geometry_margin", -1., 1., nullable=True)
    geometry = _number(event.get("geometry_cost"), "geometry_cost", 0., 1.)
    count = _int(event.get("history_updates"), "history_updates")
    rivals = _int(event.get("competitor_count"), "competitor_count")
    known = _flag(event, "competition_known")
    if (known and not rivals) or (not known and event.get("appearance_margin") is not None):
        raise ValueError("Competition availability/margin is inconsistent")
    if r is None and count >= cfg.min_history:
        raise ValueError("Mature history cannot claim cold reference fallback")
    if private and not allow:
        raise ValueError("Private anchor was written despite a rejected decision")
    if allow and (quality < cfg.min_quality or quality <= 0):
        raise ValueError("Write eligibility contradicts configured quality gate")
    if gate and not allow and (native or custom):
        raise ValueError("Rejected identity write reached native/custom memory")
    if stage == "low" and native:
        raise ValueError("Low-score stage must never write native features")
    if stage == "recovery" and (allow or private or event["reason"] != "recovered_quarantine"):
        raise ValueError("B recovery violates private-memory quarantine")
    reference = _number(event.get("reference_cost"), "reference_cost", 0., 1.)
    actual = _number(event.get("applied_cost"), "applied_cost", 0., 1.)
    base = _number(event.get("base_cost"), "base_cost", 0., 1., nullable=stage == "recovery")
    if stage == "recovery" and base is not None:
        raise ValueError("B recovery does not use the native association base cost")
    if base is not None and (reference > base + 1e-12 or actual > base + 1e-12):
        raise ValueError("Reference/applied cost exceeds the native base cost")
    if stage == "low" and abs(base - geometry) > 1e-12:
        raise ValueError("Low-score base cost differs from raw geometry")
    if (not applied or r is None) and actual != reference:
        raise ValueError("Reference/shadow cost was changed")
    if applied and actual < reference - 1e-12:
        raise ValueError("C2 increased the reference appearance bonus")
    if applied and r is not None and abs(actual - (base - r * (base - reference))) > 1e-12:
        raise ValueError("Applied cost disagrees with logged C2 calibration")


def _validate_seed(event, cfg):
    _key(event)
    _flag(event, "provisional", True)
    _flag(event, "observe_only", cfg.observe_only)
    gate = cfg.memory_enabled and not cfg.observe_only
    _flag(event, "memory_gate_applied", gate)
    allow = _flag(event, "would_allow_write")
    _flag(event, "private_anchor_written", allow)
    native = _flag(event, "native_feature_present")
    custom = _modalities(event.get("custom_modalities_written"))
    if gate and not allow and (native or custom):
        raise ValueError("Rejected birth seed remains in native/custom memory")


def _sequence(path, sequence, frames, cfg, aggregate):
    result = {"status": "FAIL", "trace_complete": False, "errors": [], "stages": None,
              "association": None, "seeds": None}
    if not path.is_file():
        result["errors"].append("Missing event JSONL; replay with --trace")
        return result
    local, associations, seeds = _empty_stats(), Counter(), Counter()
    frame_records, association_records = Counter(), Counter()
    association_pairs = {}
    decisions, matches, seed_keys, birth_keys = Counter(), Counter(), Counter(), Counter()
    writers, claimed_writes = {}, {}
    previous_frame = 0
    with path.open(encoding="utf-8-sig") as stream:
        for line, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            try:
                event = _json_loads(raw)
                if not isinstance(event, dict) or not isinstance(event.get("event"), str):
                    raise ValueError("Event must be an object with an event name")
                kind = event["event"]
                frame = _int(event.get("frame"), "frame", 1)
                if event.get("sequence") != sequence or not previous_frame <= frame <= frames:
                    raise ValueError("Event sequence/frame differs from protocol or order")
                previous_frame = frame
                if kind == "frame":
                    frame_records[frame] += 1
                if kind.startswith("identity_"):
                    if not cfg.enabled:
                        raise ValueError("Identity events exist but protocol identity is disabled")
                if kind == "identity_decision":
                    _validate_decision(event, cfg)
                    key = _key(event)
                    decisions[(event["stage"], *key)] += 1
                    if key in claimed_writes:
                        raise ValueError("Duplicate identity write decision")
                    claimed_writes[key] = event["custom_modalities_written"]
                    _add_decision(local, event)
                elif kind == "identity_seed":
                    _validate_seed(event, cfg)
                    key = _key(event)
                    seed_keys[key] += 1
                    if key in claimed_writes:
                        raise ValueError("Duplicate identity seed/write decision")
                    claimed_writes[key] = event["custom_modalities_written"]
                    seeds["count"] += 1
                    seeds["provisional_anchor_written"] += event["private_anchor_written"]
                    seeds["custom_memory_written"] += bool(event["custom_modalities_written"])
                    seeds["native_feature_present"] += event["native_feature_present"]
                elif kind == "identity_association":
                    stage = event.get("stage")
                    if stage not in ("high", "unconfirmed"):
                        raise ValueError("Identity association has an invalid stage")
                    _flag(event, "applied", cfg.association_enabled and not cfg.observe_only)
                    pairs = _int(event.get("pair_count"), "pair_count")
                    changed = _int(event.get("proposed_changed_pairs"), "proposed_changed_pairs")
                    cold = _int(event.get("cold_reference_pairs"), "cold_reference_pairs")
                    maximum = _number(event.get("proposed_max_cost_increase"), "proposed_max_cost_increase", -1e-12, 1.)
                    if max(changed, cold) > pairs or (changed == 0 and abs(maximum) > 1e-12):
                        raise ValueError("Association pair counts/cost increase are inconsistent")
                    association_records[(frame, stage)] += 1
                    association_pairs[(frame, stage)] = pairs
                    associations["records"] += 1
                    associations["pairs"] += pairs
                    associations["proposed_changed_pairs"] += changed
                    associations["cold_reference_pairs"] += cold
                    associations["proposed_max_cost_increase"] = max(associations["proposed_max_cost_increase"], maximum)
                elif kind.startswith("identity_"):
                    raise ValueError("Unknown identity event")
                elif cfg.enabled and kind in ("match", "recovered"):
                    stage = "recovery" if kind == "recovered" else event.get("stage")
                    if stage not in STAGES:
                        raise ValueError("Native match has an invalid stage")
                    matches[(stage, *_key(event))] += 1
                elif cfg.enabled and kind == "native_birth":
                    birth_keys[_key(event)] += 1
                elif cfg.enabled and kind == "memory_update":
                    key = _key(event)
                    if key in writers:
                        raise ValueError("Duplicate custom memory update")
                    writers[key] = _modalities(event.get("modalities"))
            except (ValueError, TypeError, KeyError, OverflowError) as error:
                result["errors"].append(f"event line {line}: {error}")
    if cfg.enabled:
        if any(frame_records[f] != 1 for f in range(1, frames + 1)):
            result["errors"].append("Missing/duplicate frame trace records")
        if any(association_records[f, stage] != 1 for f in range(1, frames + 1)
               for stage in ("high", "unconfirmed")):
            result["errors"].append("Missing/duplicate identity association trace records")
        if decisions != matches:
            result["errors"].append("Identity decisions do not match native match/recovery events")
        matched_pairs = Counter()
        for (stage, frame, _, _), count in decisions.items():
            if stage in ("high", "unconfirmed"):
                matched_pairs[frame, stage] += count
        if any(count > association_pairs.get(key, 0) for key, count in matched_pairs.items()):
            result["errors"].append("Accepted identity decisions exceed association pair counts")
        if seed_keys != birth_keys:
            result["errors"].append("Identity seeds do not match native birth events")
        expected_writes = {key: values for key, values in claimed_writes.items() if values}
        if writers != expected_writes:
            result["errors"].append("Custom memory_update events differ from claimed identity writes")
    result["trace_complete"] = not result["errors"]
    if not result["errors"]:
        result["status"] = "PASS" if cfg.enabled else "NOT_ENABLED"
        if cfg.enabled:
            result.update(stages=_summary(local), association=dict(associations), seeds=dict(seeds))
            for stage in STAGES:
                for field in ("counts", "reasons", "histogram"):
                    aggregate[stage][field].update(local[stage][field])
                aggregate[stage]["r_sum"] += local[stage]["r_sum"]
    return result


def audit_run(run_dir, trackers, compare_pairs=()):
    integrity = audit_integrity(run_dir, trackers, compare_pairs)
    root = Path(run_dir).resolve()
    result = {"schema": "botsort_plus_identity_audit_v1", "status": "FAIL", "errors": [],
              "scope": "Protocol/output integrity and log claims only; no GT, identity accuracy or runtime-memory proof.",
              "summary_population": "Accepted match decisions; association pair totals are separate.",
              "integrity": integrity, "trackers": {}, "comparisons": []}
    configs = {}
    for name in trackers:
        entry = {"status": "FAIL", "errors": [], "trace_complete": False,
                 "stages": None, "sequences": {}}
        result["trackers"][name] = entry
        try:
            protocol = _read_json(root / "protocols" / (name + ".json"))
            raw = protocol.get("config")
            if not isinstance(raw, dict) or raw.get("name") != name:
                raise ValueError("Protocol config name differs from tracker name")
            cfg = PlusConfig.from_dict(raw)
            configs[name] = cfg
            entry["identity_config"] = cfg.to_dict()["identity"]
            if protocol.get("status") != "COMPLETE":
                raise ValueError("Protocol is not COMPLETE")
            sequences = protocol.get("sequences")
            if not isinstance(sequences, dict) or not sequences:
                raise ValueError("Protocol has no sequence contract")
            aggregate = _empty_stats()
            for sequence, frames in sorted(sequences.items()):
                _name(sequence)
                frames = _int(frames, "sequence frames", 1)
                item = _sequence(root / "events" / name / (sequence + ".jsonl"),
                                 sequence, frames, cfg.identity, aggregate)
                entry["sequences"][sequence] = item
                entry["errors"].extend(f"{sequence}: {error}" for error in item["errors"])
            entry["trace_complete"] = not entry["errors"]
            if not entry["errors"]:
                entry["status"] = "PASS" if cfg.identity.enabled else "NOT_ENABLED"
                entry["stages"] = _summary(aggregate) if cfg.identity.enabled else None
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            entry["errors"].append(str(error))
    for left, right in compare_pairs:
        check = {"left": left, "right": right, "status": "FAIL", "errors": []}
        a, b = configs.get(left), configs.get(right)
        if a is None or b is None:
            check["errors"].append("Both validated configs are required")
        elif a.identity.enabled or not b.identity.enabled or not b.identity.observe_only:
            check["errors"].append("C2 shadow comparison requires identity-disabled reference then identity-enabled observe_only run")
        else:
            def foundation(cfg):
                return {k: v for k, v in cfg.to_dict().items() if k not in ("name", "identity")}
            if foundation(a) != foundation(b):
                check["errors"].append("Reference/shadow configs differ outside name and identity options")
        check["status"] = "PASS" if not check["errors"] else "FAIL"
        result["comparisons"].append(check)
    result["error_count"] = (integrity["error_count"]
                             + sum(len(x["errors"]) for x in result["trackers"].values())
                             + sum(len(x["errors"]) for x in result["comparisons"]))
    result["status"] = "PASS" if not result["error_count"] else "FAIL"
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--trackers", nargs="+", required=True)
    parser.add_argument("--compare-pair", nargs=2, action="append", default=[], metavar=("REFERENCE", "SHADOW"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError("Report exists; choose a new --output")
        output, root = args.output.resolve(), args.run_dir.resolve()
        if any(output.is_relative_to(root / part) for part in ("protocols", "trackers", "events", "timing")):
            raise ValueError("Report must not be written inside run evidence directories")
        report = audit_run(root, args.trackers, args.compare_pair)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"status": report["status"], "error_count": report["error_count"], "report": str(output)}))
        return 0 if report["status"] == "PASS" else 2
    except (OSError, ValueError, TypeError) as error:
        print(f"Identity audit failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
