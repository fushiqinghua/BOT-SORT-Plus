"""Offline labels are joined only after the label-free assignment is finished."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import gzip
import json

from .engine import METHODS, evaluate_frame
from .inputs import edge_key, group_key
from .protocol import ENGINE, LABELS, PARTITIONS


def evaluate(bundle):
    frames = defaultdict(list)
    for row in bundle.rows:
        frames[(row["sequence"], row["frame"])].append(row)
    decisions = []
    for frame in sorted(frames):
        inference_rows = frames[frame]
        for method in METHODS:
            decisions.extend(evaluate_frame(inference_rows, method, ENGINE))
    # No label dictionary is passed to any score or assignment operation.
    for decision in decisions:
        key = edge_key(decision)
        label = bundle.labels[key]
        decision.update(partition=bundle.partitions[key[0]], source_video=bundle.sources[key[0]],
                        identity_label=label["identity_label"], opportunity=label["opportunity"],
                        gt_id=label.get("gt_id"), fn_episode_start_frame=label.get("fn_episode_start_frame"))
    return decisions


def metrics(rows, decisions, labels):
    budget = [row for row in rows if row["geometry_group_rank"] <= ENGINE["budget"]]
    group_rows = {group_key(row): row for row in budget}
    available = Counter()
    group_sources = defaultdict(set)
    possible = set()
    for row in group_rows.values():
        label = labels[edge_key(row)]
        category = label["identity_label"]
        available[category] += 1
        group_sources[category].add(label["source_video"])
        if label["opportunity"]:
            possible.add((row["sequence"], row["frame"], label["gt_id"]))
    adopted = [row for row in decisions if row["selected"]]
    accepted = Counter(row["identity_label"] for row in adopted)
    useful = [row for row in adopted if row["opportunity"]]
    targets = {(row["sequence"], row["frame"], row["gt_id"]) for row in useful}
    episodes = {(row["sequence"], row["gt_id"], row["fn_episode_start_frame"]) for row in useful}
    eligible = [row for row in decisions if row["eligible"]]
    track_groups, group_tracks = defaultdict(set), defaultdict(set)
    for row in eligible:
        track = row["sequence"], row["frame"], row["track_id"]
        box = row["sequence"], row["frame"], tuple(row["raw_box_xyxy"])
        track_groups[track].add(box)
        group_tracks[box].add(track)
    reason_counts = Counter(reason for row in decisions for reason in row["reasons"])
    result = dict(input_raw_edges=len(rows), budget_raw_edges=len(budget), available_groups=len(group_rows),
                  eligible_raw_edges=len(eligible), eligible_groups=len({group_key(row) for row in eligible}),
                  accepted=len(adopted), useful_target_frames=len(targets), useful_fn_episodes=len(episodes),
                  useful_proxy_groups=len({row["source_video"] for row in useful}),
                  possible_target_frames=len(possible),
                  correct_frame_retention=len(targets) / len(possible) if possible else None,
                  assignment_choice_track_frames=sum(len(v) > 1 for v in track_groups.values()),
                  shared_candidate_groups=sum(len(v) > 1 for v in group_tracks.values()),
                  isolated_singleton_track_frames=sum(len(v) == 1 and len(group_tracks[next(iter(v))]) == 1
                                                      for v in track_groups.values()),
                  external_identity_competition_raw_edges=sum(bool(row["challengers"]) for row in budget),
                  appearance_missing_fallback_raw_edges=sum(row["appearance_status"] == "missing_fallback"
                                                            for row in decisions),
                  w3_no_competition_raw_edges=sum(row["method"] == "W3" and row["row_competition_status"] ==
                                                  row["column_competition_status"] == "not_applicable"
                                                  for row in decisions),
                  w3_unavailable_raw_edges=sum(any(reason.endswith("appearance_unavailable")
                                                  for reason in row["reasons"]) for row in decisions),
                  w3_margin_rejected_raw_edges=sum(any(reason.endswith("relative_margin")
                                                      for reason in row["reasons"]) for row in decisions),
                  rejection_reason_counts=dict(sorted(reason_counts.items())))
    for name in LABELS:
        result["available_" + name] = available[name]
        result["available_proxy_groups_" + name] = len(group_sources[name])
        result["accepted_" + name] = accepted[name]
    result["other_identity_acceptance_rate"] = (accepted["other_identity"] / available["other_identity"]
                                                 if available["other_identity"] else None)
    result["confirmed_background_acceptance_rate"] = (accepted["confirmed_background"] / available["confirmed_background"]
                                                       if available["confirmed_background"] else None)
    result["background_budget_state"] = "NOT_AVAILABLE" if not available["confirmed_background"] else "COUNTED_ONLY"
    return result


def build_tables(bundle, decisions):
    comparison, per_sequence, per_source = [], [], []
    for part in PARTITIONS:
        names = [seq for seq, group in bundle.partitions.items() if group == part]
        part_rows = [row for row in bundle.rows if row["sequence"] in names]
        for method in METHODS:
            chosen = [row for row in decisions if row["partition"] == part and row["method"] == method]
            comparison.append(dict(partition=part, method=method, **metrics(part_rows, chosen, bundle.labels)))
        for seq in names:
            rows = [row for row in part_rows if row["sequence"] == seq]
            for method in METHODS:
                chosen = [row for row in decisions if row["sequence"] == seq and row["method"] == method]
                per_sequence.append(dict(partition=part, sequence=seq, source_video=bundle.sources[seq],
                                         method=method, **metrics(rows, chosen, bundle.labels)))
        for source in sorted({bundle.sources[seq] for seq in names}):
            seqs = {seq for seq in names if bundle.sources[seq] == source}
            rows = [row for row in part_rows if row["sequence"] in seqs]
            for method in METHODS:
                chosen = [row for row in decisions if row["source_video"] == source and row["method"] == method]
                per_source.append(dict(partition=part, source_video=source, method=method,
                                       **metrics(rows, chosen, bundle.labels)))
    return comparison, per_sequence, per_source


def paired_deltas(decisions):
    pairs = (("W0", "W1"), ("W1", "W2"), ("W2", "W3"), ("W_old", "W2"))
    result = []
    for part in PARTITIONS:
        for left, right in pairs:
            before = {edge_key(row): row for row in decisions if row["partition"] == part and row["method"] == left}
            after = {edge_key(row): row for row in decisions if row["partition"] == part and row["method"] == right}
            left_eligible = {k for k, row in before.items() if row["eligible"]}
            right_eligible = {k for k, row in after.items() if row["eligible"]}
            if left == "W1" and right == "W2" and left_eligible != right_eligible:
                raise AssertionError("W1/W2 feasible edge sets diverged; soft appearance became a hidden gate")
            if left == "W2" and right == "W3" and not right_eligible <= left_eligible:
                raise AssertionError("W3 added an edge absent from W2")
            left_selected = {k for k, row in before.items() if row["selected"]}
            right_selected = {k for k, row in after.items() if row["selected"]}
            added, removed = right_selected - left_selected, left_selected - right_selected
            a = Counter(after[k]["identity_label"] for k in added)
            r = Counter(before[k]["identity_label"] for k in removed)
            def targets(mapping, selected):
                return {(mapping[k]["sequence"], mapping[k]["frame"], mapping[k]["gt_id"])
                        for k in selected if mapping[k]["opportunity"]}
            lt, rt = targets(before, left_selected), targets(after, right_selected)
            changed_tracks = {(k[0], k[1], k[2]) for k in added | removed}
            row = dict(partition=part, comparison=f"{left}_to_{right}",
                       interpretation="historical_compound_reference" if left == "W_old" else "component_comparison",
                       same_eligible_edges=left_eligible == right_eligible,
                       added_eligible_edges=len(right_eligible - left_eligible),
                       removed_eligible_edges=len(left_eligible - right_eligible),
                       unchanged_selected_edges=len(left_selected & right_selected),
                       added_selected_edges=len(added), removed_selected_edges=len(removed),
                       selected_edge_delta=len(right_selected) - len(left_selected),
                       changed_track_frames=len(changed_tracks),
                       added_useful_target_frames=len(rt - lt), lost_useful_target_frames=len(lt - rt),
                       useful_target_frame_delta=len(rt) - len(lt))
            for label in LABELS:
                row["added_" + label] = a[label]
                row["removed_" + label] = r[label]
            result.append(row)
    return result


def review_queue(bundle, decisions):
    pending = defaultdict(list)
    for row in decisions:
        if row["selected"] and row["identity_label"] in ("unknown", "ignored"):
            pending[group_key(row)].append(row)
    queue = []
    for n, key in enumerate(sorted(pending), 1):
        rows = pending[key]
        example = min(rows, key=lambda row: (row["det_index"], row["method"]))
        label = bundle.labels[edge_key(example)]
        queue.append(dict(case_id=f"WEM_R31_{n:05d}", partition=example["partition"], source_video=example["source_video"],
                          sequence=example["sequence"], frame=example["frame"], track_id=example["track_id"],
                          det_indices=" ".join(map(str, sorted({row["det_index"] for row in rows}))),
                          raw_box_xyxy=example["raw_box_xyxy"], raw_scores=sorted({row["raw_score"] for row in rows}),
                          adopted_by=" ".join(method for method in METHODS if any(row["method"] == method for row in rows)),
                          offline_identity_label=example["identity_label"], identity_rule=label.get("identity_rule"),
                          anchor_status=label.get("anchor_status"), best_gt_iou=label.get("best_gt_iou"),
                          model_assisted_status=label.get("model_assisted_status"), human_review_status="pending",
                          human_decision="", reviewer="", evidence=""))
    return queue


def check_historical_controls(comparison, archive_path):
    with archive_path.open(encoding="utf-8-sig", newline="") as stream:
        historical = list(csv.DictReader(stream))
    fields = ("accepted", "available_groups", "possible_target_frames", "useful_target_frames",
              "useful_fn_episodes", "useful_proxy_groups", *["accepted_" + label for label in LABELS],
              *["available_" + label for label in LABELS])
    checks = []
    for part in PARTITIONS:
        for method, old_method in (("W0", "geometry"), ("W_old", "history_competition")):
            prior = [row for row in historical if row["partition"] == part and row["method"] == old_method and row["budget"] == "5"]
            current = [row for row in comparison if row["partition"] == part and row["method"] == method]
            if len(prior) != 1 or len(current) != 1:
                raise ValueError("Historical fixed-control rows are not uniquely identified")
            mismatches = {name: {"expected": int(prior[0][name]), "observed": current[0][name]}
                          for name in fields if int(prior[0][name]) != current[0][name]}
            if mismatches:
                raise ValueError(f"Historical control parity failed: {part}/{method}: {mismatches}")
            checks.append(dict(partition=part, method=method, fields_checked=list(fields), status="PASS"))
    return dict(status="PASS", checks=checks)


def json_text(data):
    return json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def write_csv(path, rows, empty_fields=()):
    fields = list(dict.fromkeys(key for row in rows for key in row)) or list(empty_fields)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
                             if isinstance(value, (list, dict)) else value for key, value in row.items()})


def write_decisions(path, decisions):
    # Stable compression header; repeated deterministic comparisons are diffable.
    with path.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as stream:
        for row in decisions:
            stream.write((json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
