"""Join read-only W replay snapshots to official VT-Tiny-MOT CLEAR truth."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from label_rules import (GT_IOU_MARGIN, GT_IOU_MIN, HISTORY_FRAMES, HISTORY_MIN, IGNORE_FRACTION,
                         classify_identity, identity_anchor, make_truth, observe_ground_truth,
                         overlap, xywh_to_xyxy)


BOX_DUPLICATE_REVIEW_IOU = 0.90
METHODS = {"B": "B_early_moderate", "BG": "evidence_both"}
EDGE_COUNTS = (
    "history_valid_track_frames", "history_valid_and_any_s0_track_frames", "any_s0_missing_track_frames",
    "same_identity_track_frames", "legal_unowned_same_identity_track_frames",
    "already_covered_legal_same_identity_track_frames", "candidate_edges",
)


def read_json(path):
    with opener(path)(path, "rt", encoding="utf-8-sig") as stream:
        return json.load(stream)


def opener(path):
    return gzip.open if path.suffix == ".gz" else open


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def emit(stream, row):
    stream.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        yield from csv.reader(stream)


def ignored_boxes(path):
    boxes = defaultdict(list)
    for row in read_csv(path):
        if not row or float(row[6]) != 0:
            continue
        x, y, w, h = map(float, row[2:6])
        if w > 0 and h > 0:
            boxes[int(row[0])].append([x, y, x+w, y+h])
    return boxes


def indexed(rows, field):
    index = {row[field]: row for row in rows}
    if len(index) != len(rows):
        raise ValueError(f"Repeated snapshot {field}")
    return index


def frozen_class(row, low, high):
    score = float(row[4])
    if score > high:
        return "high"
    if score == high:
        return "high_boundary"
    if score > low:
        return "low"
    return "s0"


def validate_frame(capture, cache, truth, reference, sequence):
    if capture["reference"] != reference or capture["sequence"] != sequence or not (
            capture["frame"] == cache["frame"] == truth["frame"]):
        raise ValueError("Reference, sequence, or frame alignment is invalid")
    raw = np.asarray(cache["detections"], dtype=np.float32).astype(float).reshape(-1, 6)
    observations = capture["detections"]
    if list(range(len(raw))) != [row["det_index"] for row in observations]:
        raise ValueError("Snapshot does not preserve frozen raw row order")
    low, high = (capture["thresholds"][key] for key in ("low", "high"))
    if (low, high) != (0.1, 0.5):
        raise ValueError("Expected frozen reference thresholds .1/.5")
    s0 = set()
    for item, values in zip(observations, raw, strict=True):
        if item["raw_xyxy"] != values[:4].tolist() or item["raw_score"] != values[4]:
            raise ValueError("Snapshot detection row changed")
        kind = frozen_class(values, low, high)
        actual = str(item["band"]).lower()
        if actual != kind and not (kind == "high_boundary" and actual == "high_equal_excluded"):
            raise ValueError(f"Snapshot score split changed: expected {kind}, got {actual}")
        if item["detector_class"] != values[5]:
            raise ValueError("Snapshot detector class changed")
        if kind == "s0":
            s0.add(item["det_index"])
    outputs = indexed(capture["outputs"], "track_id")
    if outputs.keys() != truth["outputs"].keys():
        raise ValueError("Snapshot output IDs do not match official reference MOT")
    accepted = indexed(capture["accepted"], "track_id")
    for tid, item in outputs.items():
        if tid not in accepted or item["det_index"] != accepted[tid]["det_index"]:
            raise ValueError("Output is not linked to a same-frame accepted raw row")
        box = xywh_to_xyxy(truth["outputs"][tid]["box_xywh"])
        if not np.allclose(item["xyxy"], box, atol=1.e-4, rtol=0):
            raise ValueError("Snapshot output box disagrees with original MOT serialization")
    consumed = set(capture["consumed"])
    if consumed != {item["det_index"] for item in accepted.values()}:
        raise ValueError("Snapshot consumed set and accepted observation ledger disagree")
    reserved = {item["det_index"] for item in capture["candidates_current"] if item["reserved"]}
    if any(i < 0 or i >= len(raw) for i in consumed | reserved):
        raise ValueError("Ownership ledger contains a non-current row")
    for obs in capture["accepted"]:
        i = obs["det_index"]
        if obs["raw_xyxy"] != raw[i, :4].tolist() or obs["raw_score"] != raw[i, 4]:
            raise ValueError("Accepted observation differs from the frozen row")
    return raw, s0, accepted, outputs, consumed, reserved


def ownership(det, consumed, reserved, raw):
    owned = sorted((consumed | reserved) - {det})
    sims, _ = overlap(raw[det, :4], raw[owned, :4])
    best_iou = float(sims.max(initial=0))
    neighbor = owned[int(np.argmax(sims))] if best_iou > 0 else None
    exact = det in consumed or det in reserved
    if det in consumed:
        status = "raw_row_consumed"
    elif det in reserved:
        status = "raw_row_reserved_by_B"
    elif best_iou >= BOX_DUPLICATE_REVIEW_IOU:
        status = "near_identical_owned_box_review_required"
    else:
        status = "no_observed_exact_ownership_conflict"
    return dict(ownership_status=status, hard_ownership_conflict=exact, nearest_owned_row=neighbor,
                nearest_owned_iou=best_iou, near_duplicate_ownership_risk=best_iou >= BOX_DUPLICATE_REVIEW_IOU)


class SequenceLabels:
    def __init__(self, role, sequence, group, reviews):
        self.role, self.sequence, self.group = role, sequence, group
        self.reference, self.reviews = METHODS[role], reviews
        self.counts = Counter(dict.fromkeys(EDGE_COUNTS, 0))
        self.past = {}
        self.last_covering_id = {}
        self.last_uncovered = {}
        self.opportunities = {}
        self.fn_partitions = Counter()
        self.anchor_states = Counter()
        self.labels = Counter()

    def annotate(self, capture, frozen, truth, ignored, edge_stream, track_stream, fn_stream):
        raw, s0, accepted, outputs, consumed, reserved = validate_frame(
            capture, frozen, truth, self.reference, self.sequence)
        frame = capture["frame"]
        current = make_truth(truth, ignored)
        gt_ids, coverage = set(current["gt_ids"]), current["coverage"]
        episode_start = {}
        for gid in gt_ids - coverage.keys():
            previous, start = self.last_uncovered.get(gid, (None, frame))
            episode_start[gid] = start if previous == frame - 1 else frame
            self.last_uncovered[gid] = frame, episode_start[gid]
        info = dict(role=self.role, reference=self.reference, sequence=self.sequence, source_video=self.group, frame=frame)
        tracks_by_collection = {}
        for collection in ("removed_stracks", "lost_stracks", "active_tracks"):
            rows = [x for x in capture["start_tracks"] if x["collection"] == collection]
            if collection == "removed_stracks":
                # Native bookkeeping can append the same terminal ID more than once.
                tracks_by_collection[collection] = {x["track_id"]: x for x in rows}
            else:
                tracks_by_collection[collection] = indexed(rows, "track_id")
        # A recent Removed track can still be present in both archival lists.
        # The active collection alone decides frame-start W eligibility.
        starts = {tid: track for index in tracks_by_collection.values() for tid, track in index.items()}
        eligibility = {tid for tid, track in tracks_by_collection["active_tracks"].items()
                       if track["state"] == 1 and track["confirmed"]}
        missing = eligibility - accepted.keys()
        if missing != set(capture["at_risk_track_ids"]):
            raise ValueError("Snapshot frame-start eligibility diverged")
        pairs = {(r["track_id"], r["det_index"]) for r in capture["s0_edges"]}
        if len(pairs) != len(capture["s0_edges"]) or pairs != {(tid, det) for tid in missing for det in s0}:
            raise ValueError("Snapshot S0 edges do not match the complete active missing × current S0 product")
        self.counts.update(frames=1, gt_frames=len(gt_ids), reference_gt_missed=len(episode_start),
                           reference_covered=len(coverage), s0_rows=len(s0),
                           repeated_start_identity_collections=len(capture["start_tracks"])-len(starts),
                           s0_frames=int(bool(s0)), frame_start_active_track_frames=len(eligibility),
                           missing_after_native_b_track_frames=len(missing))
        candidate_truth = {det: observe_ground_truth(raw[det, :4], current) for det in s0} if missing else {}
        candidate_ownership = {det: ownership(det, consumed, reserved, raw) for det in candidate_truth}
        grouped_edges = defaultdict(list)
        for edge in capture["s0_edges"]:
            grouped_edges[edge["track_id"]].append(edge)
        good_by_gt = defaultdict(list)
        for tid in sorted(missing):
            anchor, anchor_status, evidence = identity_anchor(starts[tid]["raw_history"], frame, tid, self.past)
            self.anchor_states[anchor_status] += 1
            base = dict(info, track_id=tid, gt_id=anchor, gt_box_xyxy=current["boxes"].get(anchor),
                        frame_start_active=True, missing_after_native_b=True, history_valid=anchor is not None,
                        anchor_status=anchor_status, causal_history_evidence=evidence, anchor_present=anchor in gt_ids,
                        reference_gt_covered=None if anchor not in gt_ids else anchor in coverage,
                        reference_gt_covering_track_id=coverage.get(anchor), fn_episode_start_frame=episode_start.get(anchor))
            bins = Counter()
            for edge in grouped_edges[tid]:
                det = edge["det_index"]
                candidate = candidate_truth[det]
                own = candidate_ownership[det]
                review = self.reviews.get((self.sequence, frame, det))
                if review is not None and review["raw_box_xyxy"] != raw[det, :4].tolist():
                    raise ValueError("Independent review is bound to a different raw box")
                label, reason = classify_identity(anchor, candidate, review)
                self.labels[label] += 1
                bins[label] += 1
                geometry, gmc = edge["native_low_feasible"], edge["gmc_guard_feasible"]
                no_conflict = not own["hard_ownership_conflict"] and not own["near_duplicate_ownership_risk"]
                legitimate = label == "same_identity" and geometry and gmc and no_conflict
                bins.update(same_and_legal=int(legitimate), same_blocked_geometry=int(label == "same_identity" and not geometry),
                            same_blocked_gmc=int(label == "same_identity" and geometry and not gmc),
                            same_blocked_ownership=int(label == "same_identity" and geometry and gmc and not no_conflict))
                opportunity = bool(legitimate and anchor in episode_start)
                if opportunity:
                    key = frame, anchor
                    entry = self.opportunities.setdefault(key, dict(info, gt_id=anchor,
                                      fn_episode_start_frame=episode_start[anchor], edge_count=0,
                                      track_ids=set(), det_indices=set()))
                    entry["edge_count"] += 1
                    entry["track_ids"].add(tid)
                    entry["det_indices"].add(det)
                other_iou = max((value for gid, value in candidate["ious"].items() if gid != anchor), default=0.0)
                emit(edge_stream, dict(base, det_index=det, raw_box_xyxy=raw[det, :4].tolist(), raw_score=float(raw[det, 4]),
                                       source="native_cache", original_row=det, identity_label=label, identity_rule=reason,
                                       best_candidate_gt_id=candidate["gt_id"], best_gt_iou=candidate["best_iou"],
                                       best_second_iou=candidate["second_iou"], best_overlap_margin=candidate["overlap_gap"],
                                       ignored_coverage=candidate["ignored_fraction"], same_gt_iou=candidate["ious"].get(anchor),
                                       best_other_iou=other_iou, formal_geometry_allowed=geometry, gmc_guard_allowed=gmc,
                                       selected_prediction_iou=edge["selected_iou"], backup_prediction_iou=edge["backup_iou"],
                                       independent_anchor_feasible=edge["independent_anchor_feasible"], **own,
                                       opportunity=opportunity, model_assisted_status=None if review is None else review.get("model_assisted_status"),
                                       human_review_status="not_reviewed" if review is None else review.get("human_review_status", "not_reviewed")))
            if anchor in episode_start:
                good_by_gt[anchor].append(bins)
            self.counts.update(history_valid_track_frames=int(anchor is not None),
                               history_valid_and_any_s0_track_frames=int(anchor is not None and bool(s0)),
                               any_s0_missing_track_frames=int(bool(s0)),
                               same_identity_track_frames=int(bool(bins["same_identity"])),
                               legal_unowned_same_identity_track_frames=int(bool(bins["same_and_legal"])),
                               already_covered_legal_same_identity_track_frames=int(bool(bins["same_and_legal"]) and anchor in coverage),
                               candidate_edges=len(grouped_edges[tid]))
            emit(track_stream, dict(base, s0_rows=len(s0), candidate_labels=dict(bins)))
        for gid, first in sorted(episode_start.items()):
            previous = self.last_covering_id.get(gid)
            bins = sum(good_by_gt[gid], Counter())
            stage = self.missed_partition(frame, gid, bins, bool(s0), bool(good_by_gt[gid]), starts, missing, previous)
            self.fn_partitions[stage] += 1
            emit(fn_stream, dict(info, gt_id=gid, fn_episode_start_frame=first, partition=stage,
                                 previous_clear_track_id=previous, matched_causal_anchor_tracks=len(good_by_gt[gid])))
        self.last_covering_id.update(coverage)
        self.past[frame] = dict(raw=raw, accepted=accepted,
                                output_indices={tid: row["det_index"] for tid, row in outputs.items()}, truth=current)
        self.past.pop(frame-HISTORY_FRAMES, None)

    def missed_partition(self, frame, gt_id, bins, any_s0, any_anchor, starts, missing, last_owner):
        if (frame, gt_id) in self.opportunities:
            return "conditional_uncovered_opportunity"
        if any_anchor:
            if not any_s0:
                return "strict_anchor_without_s0"
            if not bins["same_identity"]:
                return "s0_without_strict_same_identity"
            if bins["same_blocked_ownership"]:
                return "same_identity_blocked_by_ownership"
            if bins["same_blocked_gmc"]:
                return "same_identity_blocked_by_gmc_guard"
            return "same_identity_blocked_by_native_geometry"
        if last_owner in missing:
            return "active_missing_without_strict_history"
        if last_owner is None:
            return "no_earlier_clear_owner"
        if last_owner not in starts:
            return "earlier_id_absent_at_start"
        if starts[last_owner]["state"] == 2:
            return "earlier_id_lost_at_start"
        return "earlier_id_updated_or_otherwise_ineligible"

    def summary(self):
        episodes = {(r["gt_id"], r["fn_episode_start_frame"]) for r in self.opportunities.values()}
        tracks = {track for r in self.opportunities.values() for track in r["track_ids"]}
        targets = {r["gt_id"] for r in self.opportunities.values()}
        if sum(self.fn_partitions.values()) != self.counts["reference_gt_missed"]:
            raise ValueError("FN partitions do not account for the official denominator")
        return dict(role=self.role, reference=self.reference, sequence=self.sequence, source_video=self.group,
                    **self.counts, candidate_identity_labels=dict(self.labels), history_anchor_status=dict(self.anchor_states),
                    official_fn_partition=dict(self.fn_partitions),
                    uncovered_opportunity_gt_frames=len(self.opportunities), uncovered_opportunity_fn_episodes=len(episodes),
                    uncovered_opportunity_tracks=len(tracks), uncovered_opportunity_gt_ids=len(targets))


def digestable_snapshots(path):
    with opener(path)(path, "rt", encoding="utf-8-sig") as stream:
        yield from map(json.loads, stream)


def frozen_path(root, sequence):
    options = [root / (sequence + suffix) for suffix in (".json.gz", ".json")]
    paths = [path for path in options if path.is_file()]
    if len(paths) != 1:
        raise ValueError(f"Expected one frozen detection cache for {sequence}")
    return paths[0]


def read_reviews(path):
    if path is None:
        return {}
    rows = list(digestable_snapshots(path))
    index = {(r["sequence"], r["frame"], r["det_index"]): r for r in rows}
    if len(index) != len(rows):
        raise ValueError("Independent review repeats a frozen observation key")
    return index


def flatten(summary):
    row = {}
    for key, value in summary.items():
        if isinstance(value, dict):
            row.update({f"{key}.{name}": count for name, count in value.items()})
        else:
            row[key] = value
    return row


def group_summary(summaries):
    total = Counter()
    for summary in summaries:
        total.update({key: value for key, value in flatten(summary).items() if isinstance(value, int)})
    total["sequences"] = len(summaries)
    positive_groups = {r["source_video"] for r in summaries if r["uncovered_opportunity_gt_frames"] and r["source_video"]}
    total["candidate_opportunity_proxy_source_groups"] = len(positive_groups)
    total["candidate_opportunity_sequences"] = sum(bool(r["uncovered_opportunity_gt_frames"]) for r in summaries)
    return dict(total)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_table(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    sys.path.insert(0, str(args.repo.resolve(strict=True)))
    from experiments.vttiny_gmc.round3_joint import analysis

    trackeval, audit = analysis.load_trackeval(args.trackeval)
    inventory = read_json(args.inventory)
    official_inputs = read_json(args.tco_input)
    official_references = read_json(args.tco_reference)["references"]
    parity = read_json(args.snapshot_proof)
    if parity["status"] != "PASS" or parity["benchmark_gt_read"] or parity["online_w_updates"]:
        raise ValueError("R0 parity has not passed for the requested scope")
    sequences = sorted(args.sequences or parity["sequences"])
    if len(sequences) != len(set(sequences)) or not set(sequences) <= inventory["files"].keys():
        raise ValueError("Requested scope is not in the prior R1a frozen inventory")
    if not set(sequences) <= set(parity["sequences"]):
        raise ValueError("Requested scope is not in the verified R0 proof")
    groups = read_json(args.proxy_groups)["development_sequence_to_source_video"] if args.proxy_groups else {}
    if groups and not set(sequences) <= groups.keys():
        raise ValueError("Proxy source-group manifest omits requested sequences")
    reviews = read_reviews(args.background_review)
    refs = {role: (Path(trackers), Path(snapshots)) for role, trackers, snapshots in args.reference}
    if len(refs) != len(args.reference) or not refs.keys() <= METHODS.keys():
        raise ValueError("Expected each of the frozen B and BG roles at most once")
    reserved_inputs = {}
    for role, (trackers, snapshots) in refs.items():
        method = METHODS[role]
        if method != official_references[role]["tracker"]:
            raise ValueError("Reference method differs from frozen manifest")
        for seq in sequences:
            source = frozen_path(args.detections, seq)
            gt = args.gt / seq / "gt" / "gt.txt"
            mot = trackers / method / "data" / (seq + ".txt")
            trace = snapshots / (seq + ".jsonl.gz")
            observed = parity["references"][role]["files"][seq]
            if trace.resolve(strict=True) != Path(observed["snapshot_file"]).resolve(strict=True):
                raise ValueError("Snapshot path differs from the verified R0 output")
            proof = {"capture_sha256": sha256(trace), "tracker_mot_sha256": sha256(mot),
                     "gt_sha256": sha256(gt), "detections_sha256": sha256(source)}
            if proof != {"capture_sha256": observed["observation_proof"]["snapshots_sha256"],
                         "tracker_mot_sha256": official_references[role]["trajectory_hashes"][seq],
                         "gt_sha256": official_inputs["ground_truth"]["sha256"][seq],
                         "detections_sha256": inventory["files"][seq]["detector_cache_sha256"]} or (
                             proof["tracker_mot_sha256"] != observed["mot_sha256"]):
                raise ValueError(f"Required R0/TCO/R1a input fingerprint mismatch at {role}/{seq}")
            reserved_inputs[role, seq] = (source, gt, trace, proof)
    args.output.mkdir(parents=True, exist_ok=False)
    reports, opportunity_rows, provenance = [], [], {}
    with (gzip.open(args.output / "offline_labels.jsonl.gz", "wt", encoding="utf-8") as labels,
          gzip.open(args.output / "eligible_missing_tracks.jsonl.gz", "wt", encoding="utf-8") as misses,
          gzip.open(args.output / "official_fn_partitions.jsonl.gz", "wt", encoding="utf-8") as official_fn):
        for role, (trackers, snapshots) in refs.items():
            method = METHODS[role]
            for seq in sequences:
                expected = inventory["files"][seq]
                source, gt, trace, proof = reserved_inputs[role, seq]
                official = analysis.evaluate_sequence(trackeval, args.gt, trackers, method, seq,
                                                      expected["frames"], args.output)
                ignored = ignored_boxes(gt)
                cache = read_json(source)
                case = SequenceLabels(role, seq, groups.get(seq), reviews)
                for capture, frozen, truth in zip(digestable_snapshots(trace), cache["frames"], official["frames"], strict=True):
                    case.annotate(capture, frozen, truth, ignored.get(truth["frame"], []), labels, misses, official_fn)
                report = case.summary()
                if report["frames"] != expected["frames"] or report["reference_gt_missed"] != official["counts"]["FN"]:
                    raise ValueError("Scope or missed-GT count differs from exact TrackEval CLEAR")
                report["reference_clear"] = official["counts"]
                reports.append(report)
                for key, row in sorted(case.opportunities.items()):
                    opportunity_rows.append(dict(row, track_ids=" ".join(map(str, sorted(row["track_ids"]))),
                                                 det_indices=" ".join(map(str, sorted(row["det_indices"])))))
                provenance[role + "/" + seq] = proof
                print(role, seq, "CLEAR_FN", report["reference_gt_missed"], "edges", report["candidate_edges"],
                      "conditional_gt_frames", report["uncovered_opportunity_gt_frames"], flush=True)
    payload = {
        "schema": "wem_r1b_official_labels_v1", "status": "OFFLINE_DIAGNOSTIC_ONLY_NOT_W_METRICS",
        "sequences": sequences,
        "opportunity_definition": "Preconditions visible in R1b only: strict past raw/CLEAR anchor, current unique same GT raw IoU, formal low and GMC guard, no exact or near-identical observed ownership risk, and official target frame not covered.",
        "unvalidated_eligibility": ["independent online trusted anchor and weak episode budget", "RGB ReID identity and background separability", "bidirectional online ambiguity and finalization", "effects of applying a weak update"],
        "background_review_supplied": bool(args.background_review),
        "proxy_source_groups_used": bool(args.proxy_groups),
        "label_policy": {"iou_min": GT_IOU_MIN, "best_gt_margin": GT_IOU_MARGIN,
                         "ignore_box_fraction": IGNORE_FRACTION, "strict_past_observations_min": HISTORY_MIN,
                         "history_frames": HISTORY_FRAMES, "near_identical_owned_review_iou": BOX_DUPLICATE_REVIEW_IOU},
        "source": {"inventory_sha256": sha256(args.inventory), "labeler_sha256": sha256(Path(__file__)),
                   "tco_input_manifest_sha256": sha256(args.tco_input),
                   "tco_references_manifest_sha256": sha256(args.tco_reference), "r0_proof_sha256": sha256(args.snapshot_proof),
                   "label_rules_sha256": sha256(Path(__file__).with_name("label_rules.py")),
                   "sequence_proofs": provenance, "pinned_trackeval": audit},
        "by_reference": {role: group_summary([r for r in reports if r["role"] == role]) for role in refs},
        "per_sequence": reports,
    }
    flattened = [flatten(row) for row in reports]
    write_table(args.output / "per_sequence.csv", flattened, sorted({k for row in flattened for k in row}))
    write_table(args.output / "opportunities.csv", opportunity_rows,
                ["role", "reference", "sequence", "source_video", "frame", "gt_id", "fn_episode_start_frame",
                 "edge_count", "track_ids", "det_indices"])
    payload["output_sha256"] = {p.name: sha256(p) for p in sorted(args.output.iterdir()) if p.is_file()}
    write_json(args.output / "opportunity_summary.json", payload)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--repo", required=True, type=Path, help="Pinned repo checkout containing the verified Round 3 evaluator")
    cli.add_argument("--trackeval", required=True, type=Path)
    cli.add_argument("--gt", required=True, type=Path, help="Official class-agnostic RGB MOT test GT directory")
    cli.add_argument("--detections", required=True, type=Path)
    cli.add_argument("--inventory", required=True, type=Path, help="R1a input_verification.json")
    cli.add_argument("--tco-input", required=True, type=Path, help="Frozen TCO input manifest")
    cli.add_argument("--tco-reference", required=True, type=Path, help="Frozen TCO reference MOT manifest")
    cli.add_argument("--snapshot-proof", required=True, type=Path, help="Verified R0 snapshot checksums")
    cli.add_argument("--reference", required=True, action="append", nargs=3,
                     metavar=("ROLE", "TRACKERS_ROOT", "METHOD_SNAPSHOTS"))
    cli.add_argument("--sequences", nargs="+")
    cli.add_argument("--proxy-groups", type=Path, help="Existing offline development sequence-to-source group manifest")
    cli.add_argument("--background-review", type=Path,
                     help="Independent JSONL review keyed by sequence/frame/det_index/frozen raw_box_xyxy")
    cli.add_argument("--output", required=True, type=Path)
    run(cli.parse_args())


if __name__ == "__main__":
    main()
