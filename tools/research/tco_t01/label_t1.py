"""Join immutable TCO candidate records to official CLEAR labels and assess box reachability."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from botsort_plus.io import detection_files, detection_fingerprint, read_detection_cache, sequence_name
from experiments.vttiny_gmc.round3_joint import analysis

THRESHOLD = 0.50
AMBIGUITY_MARGIN = 0.05
ROLES = {"B": "B_early_moderate", "BG": "evidence_both"}
MIN_HISTORY = 2


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def encode(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")


def to_xyxy(box):
    value = np.asarray(box, dtype=float).copy()
    value[2:] += value[:2]
    return value


def matrix_iou(left, right):
    left = np.asarray(left, dtype=float).reshape(-1, 4)
    right = np.asarray(right, dtype=float).reshape(-1, 4)
    inter = np.maximum(0, np.minimum(left[:, None, 2:], right[None, :, 2:])
                       - np.maximum(left[:, None, :2], right[None, :, :2])).prod(axis=2)
    union = (left[:, 2:] - left[:, :2]).prod(axis=1)[:, None] + (
        right[:, 2:] - right[:, :2]).prod(axis=1)[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def center_and_size(box):
    return (box[:2] + box[2:]) / 2, box[2:] - box[:2]


def geometry_reachability(raw, target, valid_roi, residual):
    """Finite legal size search and a closed-domain exact GT check, without clipping the GT."""
    raw = np.asarray(raw, dtype=float)
    target = np.asarray(target, dtype=float)
    valid_roi = np.asarray(valid_roi, dtype=float)
    center, wh = center_and_size(raw)
    goal, goal_wh = center_and_size(target)
    delta = np.minimum(residual["center_cap_pixels"], wh * residual["center_fraction"])
    s_min, s_max = residual["scale_min"], residual["scale_max"]
    lo = center - delta
    hi = center + delta
    exact = bool(np.all(goal >= lo) and np.all(goal <= hi)
                 and np.all(goal_wh >= s_min * wh) and np.all(goal_wh <= s_max * wh)
                 and np.all(target[:2] >= valid_roi[:2]) and np.all(target[2:] <= valid_roi[2:]))
    # For each searched size, the center closest to the target center maximizes
    # overlap with the target among allowed centers. Include the GT size clamp.
    options = [
        np.unique(np.r_[np.linspace(s_min, s_max, 7), 1, np.clip(g / w, s_min, s_max)]) * w
        for w, g in zip(wh, goal_wh, strict=True)
    ]
    sizes = np.stack(np.meshgrid(*options, indexing="ij"), axis=-1).reshape(-1, 2)
    half = sizes / 2
    allowed_lo = np.maximum(lo, valid_roi[:2] + half)
    allowed_hi = np.minimum(hi, valid_roi[2:] - half)
    viable = np.all(allowed_lo <= allowed_hi, axis=1)
    if not np.any(viable):
        return exact, None, None
    choice = np.clip(goal, allowed_lo[viable], allowed_hi[viable])
    boxes = np.c_[choice - half[viable], choice + half[viable]]
    scores = matrix_iou(boxes, [target]).ravel()
    best = int(np.argmax(scores))
    return exact, float(scores[best]), boxes[best].tolist()


def read_ignored(path):
    """Official mark=0 boxes have separate diagnostic treatment and never become negatives."""
    ignored = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for line in csv.reader(stream):
            if len(line) < 7 or float(line[6]) != 0:
                continue
            rect = to_xyxy([float(v) for v in line[2:6]])
            if np.all(rect[2:] > rect[:2]):
                ignored[int(float(line[0]))].append(rect)
    return ignored


def old_anchor(track, t, prior):
    observations = track["history"]
    if len(observations) > 3:
        raise ValueError("Collector exceeded the frozen history limit")
    values = []
    recorded = []
    for obs in observations:
        frame = int(obs["frame"])
        if not 1 <= t - frame <= 12 or obs["track_id"] != track["track_id"]:
            raise ValueError("Future, expired, or foreign identity in observation history")
        recorded.append(frame)
        got = prior[frame].get(obs["track_id"])
        if got is not None and got[0] == obs["det_index"] and got[1] is not None:
            values.append(int(got[1]))
    if recorded != sorted(set(recorded)):
        raise ValueError("Unordered or duplicated history keys")
    if len(set(values)) > 1:
        return None, "conflicting_known_history", len(values)
    if len(values) < MIN_HISTORY:
        return None, "insufficient_clear_history", len(values)
    return values[0], "verified_clear_history", len(values)


def gt_bin(target):
    if target is None:
        return "unknown"
    short = float(np.min(target[2:] - target[:2]))
    if short < 8:
        return "<8"
    if short < 16:
        return "8–<16"
    if short < 32:
        return "16–<32"
    return ">=32"


def detection_status(iou, other, ignored):
    if iou is None or ignored >= THRESHOLD:
        return "unknown"
    if iou >= THRESHOLD and (other <= 0 or iou - other >= AMBIGUITY_MARGIN):
        return "known_same_identity_from_raw_iou"
    if other >= THRESHOLD and other - iou >= AMBIGUITY_MARGIN:
        # A separate top-vs-second check below qualifies this as a known negative.
        return "other_identity_candidate"
    return "unknown"


def decide_episode(row):
    """Order matters. Exact triggers and compound reasons stay in the row."""
    if not row["history_eligible"]:
        return "history_unavailable"
    if row["gt_id"] is None or not row["anchor_present"]:
        return "identity_or_visibility_unknown"
    if row["assigned_support_label"] == "positive" and row["assigned_online_eligible"]:
        return "eligible_low_pair_correct_control"
    if row["assigned_raw_bias_supported"] and row["assigned_online_eligible"]:
        return "eligible_low_pair_localization"
    if row["unchosen_identity_supported_edge"]:
        return "eligible_low_edge_assignment"
    if row["ineligible_low_gt_aligned"]:
        return "low_outside_original_edge"
    if row["out_of_band_gt_aligned"] and not row["anchor_covered"]:
        return "score_out_of_band"
    if row["assigned_det_index"] is not None and (
            row["assigned_support_label"] == "ignore" or not row["assigned_online_eligible"]):
        return "identity_or_visibility_unknown"
    return "no_usable_proposal"


def same_key(detection, frozen):
    return np.array_equal(np.asarray(detection["raw_xyxy"], dtype=np.float32), frozen[:4]) and (
        np.float32(detection["raw_score"]) == frozen[4]
        and np.float32(detection["raw_class"]) == frozen[5])


def stage_frames(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def resume_counts(counters, episode):
    bucket = episode["primary_bucket"]
    counters.update({
        "all_low_stage_track_frames": 1,
        bucket: 1,
        "assigned_pairs": int(episode["assigned_det_index"] is not None),
        "history_eligible": int(episode["history_eligible"]),
        "assigned_history_roi_and_guard_eligible": int(episode["assigned_online_eligible"]),
        "known_anchor_present": int(episode["anchor_present"]),
        "assigned_eligible_and_known_anchor": int(episode["assigned_online_eligible"]
                                                  and episode["anchor_present"]),
        "assigned_eligible_explicit_positive_raw": int(episode["assigned_online_eligible"]
                                                       and episode["assigned_support_label"] == "positive"),
        "assigned_eligible_anchored_raw_below_050": int(episode["raw_below_050_known_anchor"]),
        "assigned_eligible_anchored_raw_below_050_other_gt_overlap": int(
            episode["raw_below_050_known_anchor"] and episode["nearest_other_iou"] > 0),
        "assigned_eligible_anchored_raw_below_050_other_gt_at_least_050": int(
            episode["raw_below_050_known_anchor"] and episode["nearest_other_iou"] >= THRESHOLD),
        "geometry_only_raw_bias": int(episode["geometry_only_raw_bias"]),
        "geometry_only_gt_exact_reachable": int(episode["geometry_only_raw_bias"] and episode["gt_exact_reachable"]),
        "geometry_only_bounded_cross_050": int(episode["geometry_only_bounded_cross_050"]),
        "geometry_only_bounded_cross_050_anchor_uncovered": int(episode["geometry_only_bounded_cross_050"]
                                                                and not episode["anchor_covered"]),
        "geometry_only_bounded_cross_050_anchor_covered": int(episode["geometry_only_bounded_cross_050"]
                                                              and episode["anchor_covered"]),
        "geometry_only_bounded_cross_050_other_gt_overlap": int(
            episode["geometry_only_bounded_cross_050"] and episode["nearest_other_iou"] > 0),
        "linked_raw_bias": int(episode["assigned_raw_bias_supported"]),
        "linked_gt_exact_reachable": int(episode["assigned_raw_bias_supported"] and episode["gt_exact_reachable"]),
        "linked_legal_candidate_crosses_050": int(episode["bounded_cross_050"]),
        "linked_legal_candidate_crosses_050_other_gt_overlap": int(
            episode["bounded_cross_050"] and episode["nearest_other_iou"] > 0),
        "linked_same_gt_output_before_refine": int(episode["assigned_raw_bias_supported"]
                                                   and episode["assigned_same_gt_clear_link"]),
        "ideal_supported_edge_any_track": int(episode["ideal_supported_edge"]),
        "ideal_supported_edge_anchor_uncovered": int(episode["ideal_supported_edge"]
                                                     and not episode["anchor_covered"]),
    })


def process_sequence(context, role, method, seq, protocol, cache_path, output_streams, counts):
    trackers, gt_root, lengths, output, trackeval, residual, source_groups, unique = context
    runtime_stream, labels_stream, episodes_stream = output_streams
    group = source_groups[seq]
    run = trackers / role
    cache = read_detection_cache(cache_path)
    expected = protocol["detection_fingerprints"][seq]
    if detection_fingerprint(cache) != expected:
        raise ValueError(f"Frozen detector discrepancy for {method}/{seq}")
    result = analysis.evaluate_sequence(trackeval, gt_root, run/"trackers", method, seq, lengths[seq], output)
    ignored = read_ignored(gt_root / seq / "gt/gt.txt")
    trace = run / "tco_capture" / method / (seq + ".jsonl.gz")
    trace_digest = analysis.digest(trace)
    capture_meta = read_json(trace.with_name(seq + ".json"))
    if capture_meta["capture_sha256"] != trace_digest or capture_meta["detection_fingerprint"] != expected:
        raise ValueError(f"Capture provenance mismatch at {method}/{seq}")
    past = {}
    fingerprint = capture_meta["run_fingerprint"]
    records = zip(stage_frames(trace), cache["frames"], result["frames"], strict=True)
    for capture, detections, truth in records:
        t = int(capture["frame"])
        if t != truth["frame"] or t != detections["frame"] or capture["sequence"] != seq:
            raise ValueError(f"Frame order mismatch at {method}/{seq}/{t}")
        if capture["run_fingerprint"] != fingerprint or capture.get("source_video", group) != group:
            raise ValueError(f"Mixed runtime fingerprints at {method}/{seq}/{t}")
        frozen = np.asarray(detections["detections"], dtype=np.float32).reshape(-1, 6)
        dets = capture["low_dets"]
        det_indices = np.asarray([int(d["det_index"]) for d in dets], dtype=int)
        low, high = capture["score_thresholds"]
        # BoxMOT appends the integer index by hstack before computing its score
        # masks. This promotes the original float32 score to float64.
        score = frozen[:, 4].astype(np.float64)
        strict_indices = np.flatnonzero((score > low) & (score < high))
        if not np.array_equal(det_indices, strict_indices) or any(
                not same_key(d, frozen[i]) for d, i in zip(dets, det_indices, strict=True)):
            raise ValueError(f"Original low-score rows differ at {method}/{seq}/{t}")
        members = capture["low_tracks"]
        original = np.asarray(capture["original_cost"], dtype=float).reshape(len(members), len(dets))
        guarded = np.asarray(capture["guarded_cost"], dtype=float).reshape(len(members), len(dets))
        original_allowed = original <= capture["cost_threshold"]
        guard_allowed = guarded <= capture["cost_threshold"]
        assigned = {int(k): int(i) for k, i in capture["assignments"]}
        assignment_indices = set(assigned.values())
        if len(assignment_indices) != len(assigned) or len(assigned) != len(capture["assignments"]):
            raise ValueError(f"Assignment ownership mismatch at {method}/{seq}/{t}")
        outputs = {int(k): int(i) for k, i in capture["outputs"]}
        if len(outputs) != len(capture["outputs"]) or len(set(outputs.values())) != len(outputs) or (
                set(outputs) != set(truth["outputs"])) or any(not 0 <= i < len(frozen) for i in outputs.values()):
            raise ValueError(f"Tracker output index mismatch at {method}/{seq}/{t}")
        gts = sorted(truth["gt_ids"])
        gt_cols = {g: k for k, g in enumerate(gts)}
        boxes = np.asarray([to_xyxy(truth["gt_boxes"][g]) for g in gts]).reshape(-1, 4)
        ious = matrix_iou(frozen[:, :4], boxes)
        ignored_ious = matrix_iou(frozen[:, :4], ignored[t]).max(axis=1, initial=0)
        past[t] = {tid: (det, truth["outputs"][tid]["matched_gt"]) for tid, det in outputs.items()}
        past.pop(t-13, None)
        frame_meta = {k: capture[k] for k in ("sequence", "frame", "image_shape", "image_sha256")}
        frame_meta.update(run_fingerprint=fingerprint, method=method, source_video=group,
                          detection_fingerprint=expected, feature_hashes=protocol["feature_hashes"][seq],
                          frame_image=capture.get("frame_image", detections["image"]))
        for p, track in enumerate(members):
            tid = int(track["track_id"])
            anchor, anchor_status, known_history = old_anchor(track, t, past)
            gi = gt_cols.get(anchor)
            target = boxes[gi] if gi is not None else None
            competing = np.delete(ious, gi, axis=1) if gi is not None else ious
            others = competing.max(axis=1, initial=0)
            score_iou = ious[:, gi] if gi is not None else None
            history_eligible = bool(track["history_eligible"])
            cur = assigned.get(tid)
            row = dict(**frame_meta, track_id=tid, gt_id=anchor,
                       gt_xyxy=target.tolist() if target is not None else None, gt_short_side_bin=gt_bin(target),
                       anchor_present=target is not None, anchor_covered=anchor in truth["coverage"],
                       history_eligible=history_eligible, anchor_status=anchor_status, known_history=known_history,
                       assigned_det_index=cur, assigned_online_eligible=False, assigned_support_label="ignore",
                       assigned_raw_bias_supported=False, assigned_same_gt_clear_link=False,
                       geometry_only_raw_bias=False, geometry_only_bounded_cross_050=False,
                       unchosen_identity_supported_edge=False, ineligible_low_gt_aligned=False,
                       out_of_band_gt_aligned=False, ideal_supported_edge=False,
                       raw_iou=None, raw_xyxy=None, gt_exact_reachable=False, feasible_iou=None,
                       bounded_cross_050=False, identity_status="unknown", nearest_other_iou=None,
                       raw_below_050_known_anchor=False, reference_low_output_status="unassigned")
            if score_iou is not None:
                masked = np.ones(len(frozen), dtype=bool)
                masked[det_indices] = False
                row["out_of_band_gt_aligned"] = bool(np.any((score_iou >= THRESHOLD) & masked))
            for q, d in enumerate(dets):
                idx = int(d["det_index"])
                selected = cur == idx
                roi = d["roi"]
                allowed = bool(original_allowed[p, q] and guard_allowed[p, q])
                online = history_eligible and allowed and bool(roi["eligible"])
                raw_score_iou = float(score_iou[idx]) if score_iou is not None else None
                if raw_score_iou is not None and raw_score_iou >= THRESHOLD and not allowed:
                    row["ineligible_low_gt_aligned"] = True
                if not original_allowed[p, q] and not selected:
                    continue
                if selected and not allowed:
                    raise ValueError(f"Selected edge was not admissible at {method}/{seq}/{t}/{tid}/{idx}")
                other = float(others[idx])
                relation = detection_status(raw_score_iou, other, float(ignored_ious[idx]))
                if relation == "other_identity_candidate":
                    sorted_other = np.sort(competing[idx])
                    if len(sorted_other) <= 1 or sorted_other[-1] - sorted_other[-2] >= AMBIGUITY_MARGIN:
                        relation = "known_other_identity_from_raw_iou"
                emitted = outputs.get(tid) == idx
                linked = bool(emitted and target is not None and
                              truth["outputs"][tid]["matched_gt"] == anchor)
                raw_bias = raw_score_iou is not None and 0 < raw_score_iou < THRESHOLD and (
                    ignored_ious[idx] < THRESHOLD and (other <= 0 or raw_score_iou - other >= AMBIGUITY_MARGIN))
                # A correct emitted track supplies a separate diagnostic link. It
                # is not independent evidence for a training label on the raw box.
                if relation == "unknown" and raw_bias and linked:
                    relation = "same_gt_clear_output_for_this_exact_low_row"
                support = {
                    "known_same_identity_from_raw_iou": "positive",
                    "known_other_identity_from_raw_iou": "negative",
                }.get(relation, "ignore")
                truth_geometry = online and (support == "positive" or raw_bias)
                exact, feasible, legal_box = (geometry_reachability(d["raw_xyxy"], target, roi["valid_xyxy"], residual)
                                               if truth_geometry else (False, None, None))
                crosses = bool(raw_bias and feasible is not None and feasible >= THRESHOLD)
                if not selected and online and support == "positive":
                    row["unchosen_identity_supported_edge"] = True
                if online and (support == "positive" or linked and raw_bias):
                    row["ideal_supported_edge"] = True
                key = {k: frame_meta[k] for k in ("run_fingerprint", "method", "sequence", "frame", "source_video")}
                key.update(track_id=tid, det_index=idx)
                runtime = dict(frame_meta, track_id=tid, pred_xyxy=track["pred_xyxy"], history=track["history"],
                               history_eligible=history_eligible, **d, original_cost=float(original[p, q]),
                               guarded_cost=float(guarded[p, q]), original_feasible=bool(original_allowed[p, q]),
                               guard_allowed=bool(guard_allowed[p, q]), assigned=selected, online_eligible=online)
                encode(runtime_stream, runtime)
                encode(labels_stream, dict(**key, gt_id=anchor, gt_xyxy=row["gt_xyxy"], anchor_status=anchor_status,
                       known_history=known_history, identity_status=relation, support_label=support,
                       identity_conditioned_raw_iou=raw_score_iou if support=="positive" else 0 if support=="negative" else None,
                       same_gt_clear_output=linked, official_coverage_of_anchor_before_refine=row["anchor_covered"],
                       raw_iou=raw_score_iou, nearest_other_iou=other, ignored_iou=float(ignored_ious[idx]),
                       gt_exact_reachable=exact, feasible_iou=feasible, feasible_xyxy=legal_box,
                       finite_candidate_crosses_050=crosses, low_iou_identity_unverified=bool(raw_bias and not linked),
                       human_visibility_status="UNREVIEWED"))
                if selected:
                    reference_status = "not_emitted"
                    if emitted:
                        matched_gt = truth["outputs"][tid]["matched_gt"]
                        reference_status = ("FP" if matched_gt is None else
                                            "same_anchored_GT_TP" if linked else "other_or_unanchored_GT_TP")
                    row.update(assigned_online_eligible=online, assigned_support_label=support,
                               assigned_same_gt_clear_link=linked, assigned_raw_bias_supported=bool(online and raw_bias and linked),
                               geometry_only_raw_bias=bool(online and raw_bias and not linked),
                               geometry_only_bounded_cross_050=bool(crosses and not linked),
                               raw_iou=raw_score_iou, raw_xyxy=d["raw_xyxy"],
                               gt_exact_reachable=exact, feasible_iou=feasible,
                               bounded_cross_050=bool(crosses and linked), identity_status=relation,
                               nearest_other_iou=other, reference_low_output_status=reference_status,
                               raw_below_050_known_anchor=bool(online and raw_score_iou is not None
                                                               and raw_score_iou < THRESHOLD))
                    if crosses:
                        category = ("linked_already_TP" if linked else "geometry_only_anchor_uncovered"
                                    if not row["anchor_covered"] else "geometry_only_anchor_covered")
                        unique[method, category].add((seq, t, anchor))
                        counts[method, "all", "ALL"][f"crossing_by_reference_output_status_{reference_status}"]+=1
                counts[method, "edge", group].update({
                    "geometrically_feasible_edges": 1,
                    "guard_allowed_edges": int(allowed),
                    "online_eligible_edges": int(online),
                    "known_positive_edges": int(online and support == "positive"),
                    "known_negative_edges": int(online and support == "negative"),
                    "unknown_edges": int(online and support == "ignore"),
                })
            row["primary_bucket"] = decide_episode(row)
            for scope, k in (("sequence", seq), ("source_video", group), ("size_bin", row["gt_short_side_bin"]),
                             ("all", "ALL")):
                resume_counts(counts[method, scope, k], row)
            encode(episodes_stream, row)
    return {"tracker": method, "sequence": seq, **result["counts"],
            "capture_sha256": trace_digest}


def make_report(path, counts, unique, groups):
    compact = {m: dict(counts[m, "all", "ALL"]) for m in ROLES.values()}
    for method, row in compact.items():
        portions = {}
        for (name, category), keys in unique.items():
            if name == method:
                portions[category] = {
                    "unique_gt_frames": len(keys),
                    "sequences": len({seq for seq, _, _ in keys}),
                    "source_videos": len({groups[seq] for seq, _, _ in keys}),
                    "source_video_names": sorted({groups[seq] for seq, _, _ in keys}),
                }
        row["candidate_crossings_deduplicated_by_gt_frame"] = portions
    lines = ["# T1 · 原生低分观测的几何可达性", "",
             "以逐帧官方 CLEAR 匹配获取历史身份。至少两个实际检测的原始行与对应已输出轨迹精确吻合，"
             "且全部已知历史同属一个真值，才认作身份已锚定。当前检测的可靠自动身份监督来自原框唯一支持"
             "的真值，使用 IoU 0.50 及 0.05 歧义间隔。", "",
             "原框 IoU 低于 0.50 时，若同一检测行已由该轨迹输出且经 CLEAR 匹配到锚定真值，"
             "另记为有身份迹象的偏框，保留训练标签为未知。这些帧的基线轨迹已命中，"
             "可达性统计描述原量测定位机会。若缺少同一输出的身份链接，仅记录几何可能；"
             "不能把其直接计作已证实正确身份或能补回的漏检。", "",
             "| 方法 | 原生低分轨迹帧 | 已配对观测 | 历史/ROI/保护均合规 |合规且已知身份|"
             " 有同 ID 输出证据的偏框 | GT 完全在可行域 | 有合法修正可过 0.50 |"
             " 身份未验证但有合法修正可过 0.50 | 其中当前 GT 漏检 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method,c in compact.items():
        values = [c.get(k, 0) for k in ("all_low_stage_track_frames", "assigned_pairs",
                  "assigned_history_roi_and_guard_eligible", "assigned_eligible_and_known_anchor",
                  "linked_raw_bias", "linked_gt_exact_reachable", "linked_legal_candidate_crosses_050",
                  "geometry_only_bounded_cross_050", "geometry_only_bounded_cross_050_anchor_uncovered")]
        lines.append(f"| {method} | " + " | ".join(map(str, values)) + " |")
    lines.extend(["", "有限搜索使用载入的冻结中心/尺度残差约束、有效图像 ROI 和未裁剪真值。"
                  "合法候选越过阈值可证实几何机会，有限搜索未越过时仍保留不确定性。"
                  "理论上的理想定位/选择只计原合法且有身份迹象的边，允许忽略学习误差、图像能否辨别，"
                  "也没有在多个轨迹间求同时可实现的一对一分配。统计单位是每组各自的轨迹帧。", "",
                  "当前状态为待独立图像可辨别性复核。T1 几何核对使用已反复使用的官方开发 30 段，"
                  "未用于 TCO 训练或工作点确定。"])
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")
    return compact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True, help="Parent of B and BG replay directories")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="Prepared tco_v0.1.local.json")
    parser.add_argument("--split-manifest", type=Path, required=True)
    args = parser.parse_args()
    settings = read_json(args.config)
    groups = read_json(args.split_manifest)["development_sequence_to_source_video"]
    residual = read_json(analysis.REPO/settings["protocol_file"])["model"]["box_residual"]
    inputs = settings["paths"]
    gt_root = Path(inputs["gt_root"])
    trackeval, eval_audit = analysis.load_trackeval(inputs["trackeval_root"])
    caches = {sequence_name(p): p for p in detection_files(Path(inputs["baseline_root"])/"detections")}
    counts = defaultdict(Counter)
    unique = defaultdict(set)
    results = []
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        output_streams = [stack.enter_context(gzip.open(output / name, "xt", encoding="utf-8", compresslevel=2))
                          for name in ("pairs.runtime.jsonl.gz", "pairs.labels.jsonl.gz", "episodes.jsonl.gz")]
        for role, method in ROLES.items():
            protocol = read_json(args.run_root/role/"protocols"/(method+".json"))
            lengths = protocol["sequences"]
            if set(lengths) != set(groups) or len(lengths) != 30 or sum(lengths.values()) != 12590:
                raise ValueError(f"{method} is not the full registered 30-clip development run")
            context = args.run_root, gt_root, lengths, output, trackeval, residual, groups, unique
            for seq in sorted(lengths):
                result = process_sequence(context, role, method, seq, protocol, caches[seq], output_streams, counts)
                results.append(result)
                values = counts[method, "sequence", seq]
                print(method, seq, values["assigned_pairs"], values["linked_legal_candidate_crosses_050"], flush=True)
    count_columns = sorted(set().union(*counts.values()))
    with (output/"reachability.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["method", "scope", "group", *count_columns])
        writer.writeheader()
        for (method, scope, group), metric in sorted(counts.items()):
            writer.writerow(dict(method=method, scope=scope, group=group, **{k: metric[k] for k in count_columns}))
    compact = make_report(output/"T1_DECISION.md", counts, unique, groups)
    readout = dict(status="GEOMETRY_EVALUATED_HUMAN_VISIBILITY_PENDING", methods=compact,
                   sequence_scores=results, trackeval_commit=eval_audit["commit"],
                   positive_iou=THRESHOLD, ambiguity_margin=AMBIGUITY_MARGIN,
                   gt_sha256={s: analysis.digest(gt_root/s/"gt/gt.txt") for s in sorted(groups)})
    analysis.dump(output/"routing_summary.json", readout)
    print(json.dumps(compact, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
