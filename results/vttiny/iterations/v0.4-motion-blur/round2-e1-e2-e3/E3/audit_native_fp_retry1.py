"""Offline E3-A attribution with verified native rows and pinned CLEAR semantics."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from contextlib import ExitStack
import csv
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

BASE = "B_early_moderate"
PIN = "12c8791b303e0a0b50f753af204249e622d0281a"
EXPECTED = {"FP": 15943, "FN": 117264, "IDSW": 374}
MEASURED = {"match", "recovered", "native_birth"}
THRESHOLDS = (.05, .10, .20)


def fingerprint(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_bytes())


def csv_output(folder, filename, rows):
    if not rows:
        raise ValueError(f"No rows for {filename}")
    path = folder / filename
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def original_video(sequence):
    match = re.fullmatch(r"(DJI_\d+)_\d+", sequence)
    if not match:
        raise ValueError(f"Unverified source-video name structure {sequence}")
    return match[1]


def read_mot(path):
    result = {}
    with path.open(newline="", encoding="ascii") as stream:
        for line in csv.reader(stream):
            f, tid = map(int, line[:2])
            x, y, w, h, score = map(float, line[2:7])
            if (f, tid) in result or not (w > 0 and h > 0 and all(map(math.isfinite, (x, y, w, h, score)))):
                raise ValueError(f"Invalid native MOT row {path}:{f}:{tid}")
            result[f, tid] = ([x, y, x + w, y + h], score)
    return result


def clear_rows(dataset, sequence, clear, assign):
    """Emit exact assignment and validate against the installed CLEAR evaluation."""
    raw = dataset.get_raw_seq_data(BASE, sequence)
    prepared = dataset.get_preprocessed_seq_data(raw, "object")
    last_match = np.full(prepared["num_gt_ids"], np.nan)
    previous = np.full_like(last_match, np.nan)
    count = Counter(TP=0, FP=0, FN=0, IDSW=0)
    labels = {}
    gt_counts = {}
    for frame, (gt, tracks, raw_gt, raw_tracks, similarities) in enumerate(zip(
        prepared["gt_ids"], prepared["tracker_ids"], raw["gt_ids"], raw["tracker_ids"],
        prepared["similarity_scores"],
    ), 1):
        if len(raw_gt) != len(gt) or len(raw_tracks) != len(tracks):
            raise ValueError(f"Raw/CLEAR rows differ at {sequence}:{frame}")
        gt_counts[frame] = len(gt)
        assigned = {}
        if len(gt) and len(tracks):
            weights = 1000 * (tracks[np.newaxis, :] == previous[gt[:, np.newaxis]]) + similarities
            eps = np.finfo(float).eps
            weights[similarities < clear.threshold - eps] = 0
            rr, cc = assign(-weights)
            valid = weights[rr, cc] > eps
            rr, cc = rr[valid], cc[valid]
            matched_gt, matched_track = gt[rr], tracks[cc]
            switched = (~np.isnan(last_match[matched_gt])) & (last_match[matched_gt] != matched_track)
            previous[:] = np.nan
            previous[matched_gt] = matched_track
            last_match[matched_gt] = matched_track
            assigned = {int(c): (int(raw_gt[r]), bool(change), float(similarities[r, c]))
                        for r, c, change in zip(rr, cc, switched)}
            count["IDSW"] += int(switched.sum())
        # The pinned CLEAR code preserves the previous matching array across
        # empty frames. Resetting it here would break metric parity.
        count["TP"] += len(assigned)
        count["FP"] += len(tracks) - len(assigned)
        count["FN"] += len(gt) - len(assigned)
        for col, track in enumerate(raw_tracks):
            label, switched, iou = assigned.get(col, (None, False, None))
            values = similarities[:, col]
            best = int(np.argmax(values)) if len(values) else None
            labels[frame, int(track)] = {
                "matched_gt": label, "clear_idsw": switched, "clear_iou": iou,
                "output_max_iou": float(values[best]) if best is not None else None,
                "output_best_gt": int(raw_gt[best]) if best is not None and values[best] > 0 else None,
            }
    metric = clear.eval_sequence(prepared)
    for key, ref in (("TP", "CLR_TP"), ("FP", "CLR_FP"), ("FN", "CLR_FN"), ("IDSW", "IDSW")):
        if count[key] != metric[ref]:
            raise ValueError(f"{sequence} CLEAR row reconstruction differs for {key}")
    return labels, gt_counts, dict(count)


def center(box):
    return (np.asarray(box[:2]) + box[2:]) * .5


def gmc_transforms(path, expected_hash, frames, sequence):
    if fingerprint(path) != expected_hash:
        raise ValueError(f"{sequence} state-stream digest changed")
    result = [np.eye(3)]
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            frame = json.loads(line)
            if frame["sequence"] != sequence or frame["frame"] != len(result):
                raise ValueError(f"Trace order/identity changed: {sequence}")
            warp = np.eye(3)
            warp[:2] = frame["gmc"]["warp"]
            if not np.isfinite(warp).all():
                raise ValueError(f"Non-finite GMC: {sequence}:{frame['frame']}")
            result.append(warp @ result[-1])
    if len(result) != frames + 1:
        raise ValueError(f"Trace length changed: {sequence}")
    return result, [np.linalg.inv(t) if abs(np.linalg.det(t)) > 1e-10 else None for t in result]


def compact_box(box):
    return json.dumps(box, separators=(",", ":")) if box is not None else ""


def motion(previous, f, box, warp, inv):
    if previous is None:
        return None
    prev_f, prev_box = previous
    if inv[prev_f] is None:
        return None
    location = warp[f] @ inv[prev_f] @ np.r_[center(prev_box), 1.]
    delta = float(np.linalg.norm(center(box) - location[:2]))
    diagonal = float(np.linalg.norm(np.asarray(prev_box[2:]) - prev_box[:2]))
    return delta / diagonal / (f - prev_f)


def causal_row(seq, frame, tid, box, score, current, past, last_rgb, event, warps, inverses, feature_cache):
    history = [entry for entry in past if frame - entry[0] <= 15]
    all_scores = [entry[1] for entry in history]
    det_score, det_box, det_pos = current
    prior_points = [(entry[0], entry[2]) for entry in history]
    previous = prior_points[-1] if prior_points else None
    if det_box is None:
        speed = None
    else:
        speed = motion(previous, frame, det_box, warps, inverses)
    vectors = feature_cache.arrays
    valid_rgb = det_pos is not None and bool(vectors["rgb_valid"][det_pos])
    direction = None
    if valid_rgb and last_rgb is not None:
        direction = float(np.clip(1 - np.dot(vectors["rgb"][det_pos], vectors["rgb"][last_rgb]), 0, 2))
    return {
        "sequence": seq, "original_video": original_video(seq), "frame": frame, "track_id": tid,
        "output_short_side": float(min(np.asarray(box[2:]) - box[:2])), "output_score": score,
        "detection_short_side": float(min(np.asarray(det_box[2:]) - det_box[:2])) if det_box is not None else None,
        "detection_score": det_score,
        "past15_observations": len(history), "past15_score_median": float(np.median(all_scores)) if all_scores else None,
        "past15_high_observations": sum(s >= .6 for s in all_scores),
        "past15_recoveries": sum(entry[3] for entry in history),
        "frames_since_previous_measured": frame - previous[0] if previous else None,
        "gmc_speed_diag_per_frame": speed, "rgb_valid": valid_rgb,
        "rgb_quality": float(vectors["rgb_quality"][det_pos]) if valid_rgb else None,
        "rgb_cosine_to_previous_real_feature": direction,
        "observed_now": bool(event is not None),
    }


def explain(row, profile):
    output = row["output_max_iou"]
    matched = row["matched_gt"]
    if matched is not None:
        return "CLEAR_TP"
    if not row["gt_count"]:
        return "no_official_gt_at_frame"
    if output >= .5 - np.finfo(float).eps:
        return "competition_or_duplicate"
    if profile["lifetime_unannotated_candidate_010"]:
        return "persistent_no_gt_support"
    if row["prior_clear_gt"] is not None:
        if (.1 <= (row["output_iou_to_prior_gt"] or 0)
                and row["output_best_gt"] == row["prior_clear_gt"]):
            return "same_target_localization_below_threshold"
        if row["past_final_annotation_of_prior_gt"]:
            return "possible_tail_after_last_annotation"
        if ((row["output_best_gt"] is not None and row["output_best_gt"] != row["prior_clear_gt"] and output >= .1)
                or (row["det_best_gt"] is not None and row["det_best_gt"] != row["prior_clear_gt"]
                    and (row["det_max_iou"] or 0) >= .1)):
            return "different_gt_nearby_after_known_identity"
        return "displaced_or_unconfirmed_after_known_identity"
    return "unconfirmed_identity_or_annotation"


def row_scan(sequence, boxes, labels, gt, event_path, cache, fc, gmc, inverses, overlaps):
    commitments = {}
    with event_path.open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            if event["event"] in MEASURED:
                key = int(event["frame"]), int(event["track_id"])
                if key in commitments:
                    raise ValueError(f"Multiple native measurements {sequence}:{key}")
                commitments[key] = event
    last_presence = {}
    for f, targets in gt.items():
        for target in targets:
            last_presence[target["gt_id"]] = f
    lifetime = defaultdict(list)
    signal = []
    prior_anchor, last_rgb = {}, {}
    history = defaultdict(lambda: deque(maxlen=16))
    profile = defaultdict(lambda: dict(lifetime_annotated_real_n=0, lifetime_annotated_output_n=0,
                                       lifetime_best_iou=0., real_n=0, real_frame_min=None, real_frame_max=None))
    all_keys = sorted(set(boxes) | set(commitments))
    for f, keys in itertools.groupby(all_keys, key=lambda pair: pair[0]):
        targets = gt[f]
        target_ids = [t["gt_id"] for t in targets]
        target_boxes = [t["xyxy"] for t in targets]
        for _, tid in keys:
            key = f, tid
            event = commitments.get(key)
            native = boxes.get(key)
            stat = profile[tid]
            det_score = det_box = det_pos = det_index = det_max = det_best_gt = det_anchor = None
            output_anchor = None
            anchor = prior_anchor.get(tid)
            if event is not None:
                det_index = int(event["det_index"])
                detections = cache["frames"][f - 1]["detections"]
                if det_index < 0 or det_index >= len(detections):
                    raise ValueError(f"Native index missing at {sequence}:{key}")
                det_box = detections[det_index][:4]
                det_score = float(detections[det_index][4])
                det_pos = int(fc.arrays["offsets"][f - 1]) + det_index
                stat["real_n"] += 1
                if stat["real_frame_min"] is None:
                    stat["real_frame_min"] = f
                stat["real_frame_max"] = f
                if targets:
                    det_overlap = overlaps([det_box], target_boxes)[0]
                    best = int(det_overlap.argmax())
                    det_max = float(det_overlap[best])
                    det_best_gt = target_ids[best] if det_max > 0 else None
                    det_anchor = float(det_overlap[target_ids.index(anchor[1])]) if anchor and anchor[1] in target_ids else None
                    stat["lifetime_annotated_real_n"] += 1
                    stat["lifetime_best_iou"] = max(stat["lifetime_best_iou"], det_max)
            if native is not None:
                box, score = native
                known = labels[key]
                chosen = known["matched_gt"]
                if targets:
                    stat["lifetime_annotated_output_n"] += 1
                    stat["lifetime_best_iou"] = max(stat["lifetime_best_iou"], known["output_max_iou"])
                if anchor and anchor[1] in target_ids:
                    output_anchor = float(overlaps([box], [target_boxes[target_ids.index(anchor[1])]])[0][0])
                result = {
                    "sequence": sequence, "original_video": original_video(sequence), "frame": f,
                    "track_id": tid, "output_box_xyxy": compact_box(box), "output_score": score,
                    "measurement_stage": event.get("stage", event["event"]) if event is not None else None,
                    "det_index": det_index, "det_box_xyxy": compact_box(det_box),
                    "detector_score": det_score, "gt_count": len(targets), **known,
                    "det_best_gt": det_best_gt, "det_max_iou": det_max,
                    "prior_clear_frame": anchor[0] if anchor else None, "prior_clear_gt": anchor[1] if anchor else None,
                    "prior_gt_present_now": bool(anchor and anchor[1] in target_ids),
                    "past_final_annotation_of_prior_gt": bool(anchor and f > last_presence[anchor[1]]),
                    "det_iou_to_prior_gt": det_anchor, "output_iou_to_prior_gt": output_anchor,
                }
                lifetime[tid].append(result)
                signal.append(causal_row(sequence, f, tid, box, score, (det_score, det_box, det_pos),
                                         history[tid], last_rgb.get(tid), event, gmc, inverses, fc))
                if chosen is not None:
                    prior_anchor[tid] = f, chosen
            if event is not None:
                history[tid].append((f, det_score, det_box, event["event"] == "recovered"))
                if fc.arrays["rgb_valid"][det_pos]:
                    last_rgb[tid] = det_pos
    return lifetime, profile, signal


def flatten(seq, lifetime, profiles):
    rows, profile_rows = [], []
    for track_id, members in sorted(lifetime.items()):
        profile = profiles[track_id]
        for t in THRESHOLDS:
            profile[f"lifetime_unannotated_candidate_{int(t * 100):03}"] = (
                profile["lifetime_annotated_real_n"] >= 3 and profile["lifetime_best_iou"] < t)
        for row in members:
            row["offline_category"] = explain(row, profile)
            row["lifetime_candidate_010_Gt_analysis_only"] = profile["lifetime_unannotated_candidate_010"]
            rows.append(row)
        counter = Counter(row["offline_category"] for row in members)
        good = [row for row in members if row["matched_gt"] is not None]
        profile_rows.append({
            "sequence": seq, "original_video": original_video(seq), "track_id": track_id,
            "first_output_frame": members[0]["frame"], "last_output_frame": members[-1]["frame"],
            "output_count": len(members), "clear_tp": len(good), "clear_fp": len(members) - len(good),
            "max_iou_all": profile["lifetime_best_iou"], "matched_gt_count": len({r["matched_gt"] for r in good}),
            **profile, "offline_category_counts": json.dumps(dict(counter), sort_keys=True),
        })
    rows.sort(key=lambda r: (r["frame"], r["track_id"]))
    return rows, profile_rows


def source_checks(args):
    audit_path = args.historical_eval / "evaluation_audit.json"
    summary_path = args.historical_eval / "summary.json"
    audit = read_json(audit_path)
    if audit["status"] != "COMPLETE" or fingerprint(summary_path) != audit["summary_sha256"]:
        raise ValueError("Historical evaluation audit is incomplete or corrupted")
    sources = audit["evaluator"]["source_files"]
    if audit["evaluator"]["commit"] != PIN or not sources:
        raise ValueError("Historical TrackEval pin differs")
    for name, hashes in sources.items():
        if fingerprint(args.trackeval / name) != hashes["raw_sha256"]:
            raise ValueError(f"Historical TrackEval source differs: {name}")
    return audit, read_json(summary_path), sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "frozen_root", "trackeval", "historical_eval", "p0", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    if read_json(args.p0)["status"] != "PASS":
        raise ValueError("P0 must pass before E3-A")
    args.output.mkdir(parents=True)
    sys.path[:0] = [str(args.repo), str(args.trackeval)]
    for name, value in (("int", int), ("float", float), ("bool", bool)):
        if name not in np.__dict__:
            setattr(np, name, value)
    import trackeval
    from trackeval.metrics.clear import linear_sum_assignment
    from botsort_plus.diagnostics.common import box_iou_matrix, load_gt
    from botsort_plus.io import FeatureCache, detection_files, detection_fingerprint, read_detection_cache, sequence_name

    audit, official, sources = source_checks(args)
    prior = {r["sequence"]: r for r in official["results"] if r["tracker"] == BASE}
    frozen = args.frozen_root / "observed"
    protocol_path = frozen / "protocols" / f"{BASE}.json"
    protocol = read_json(protocol_path)
    sequences = protocol["sequences"]
    assert (protocol["status"], len(sequences), sum(sequences.values())) == ("COMPLETE", 30, 12590)
    assert sequences == audit["validation"]["sequences"]
    baseline = Path(protocol["input_roots"]["baseline"])
    features = Path(protocol["input_roots"]["features"])
    detector_paths = {sequence_name(p): p for p in detection_files(baseline / "detections")}
    truth_root = Path(audit["ground_truth"]["root"])
    analysis = read_json(args.frozen_root / "analysis/summary.json")
    assert analysis["run_protocol_sha256"] == fingerprint(protocol_path)
    historical_frozen = read_json(args.frozen_root / "round1_findings/historical_B_comparison.json")
    dataset = trackeval.datasets.MotChallenge2DBox({
        "GT_FOLDER": str(truth_root), "TRACKERS_FOLDER": str(frozen / "trackers"), "TRACKERS_TO_EVAL": [BASE],
        "BENCHMARK": "VT-Tiny-MOT", "SPLIT_TO_EVAL": "test", "SKIP_SPLIT_FOL": True, "SEQ_INFO": sequences,
        "DO_PREPROC": False, "PRINT_CONFIG": False, "CLASSES_TO_EVAL": ["pedestrian"],
    })
    dataset.valid_classes = dataset.class_list = ["object"]
    dataset.class_name_to_class_id["object"] = 1
    clear = trackeval.metrics.CLEAR({"THRESHOLD": .5, "PRINT_CONFIG": False})

    with ExitStack() as stack:
        all_writer = signal_writer = fp_writer = None
        profile_rows, sequence_rows = [], []
        all_categories, totals = Counter(), Counter()
        raw_data = {}
        for seq, length in sequences.items():
            gt_file = truth_root / seq / "gt" / "gt.txt"
            gt_hash = fingerprint(gt_file)
            assert gt_hash == audit["ground_truth"]["sha256"][seq], (seq, "official GT")
            mot_file = frozen / "trackers" / BASE / "data" / f"{seq}.txt"
            mot_hash = fingerprint(mot_file)
            assert mot_hash == protocol["output_prediction_hashes"][seq], (seq, "MOT contract")
            assert mot_hash == historical_frozen["prediction_sha256"][seq], (seq, "historic B bytes")
            event_path = frozen / "events" / BASE / f"{seq}.jsonl"
            event_hash = fingerprint(event_path)
            assert event_hash == analysis["original_events_sha256"][seq], (seq, "original event bytes")
            cache = read_detection_cache(detector_paths[seq])
            assert len(cache["frames"]) == length
            assert detection_fingerprint(cache) == protocol["detection_fingerprints"][seq]
            feature_path = features / f"{seq}.npz"
            feature_hashes = {"features": fingerprint(feature_path),
                              "metadata": fingerprint(feature_path.with_suffix(".json"))}
            assert feature_hashes == protocol["feature_hashes"][seq]
            feature_cache = FeatureCache(feature_path, cache, expected_rgb_sha=protocol["rgb_encoder_sha256"])
            trace = protocol["diagnostics"]["traces"][seq]
            stream_path = frozen / trace["path"]
            stream_manifest = read_json(frozen / trace["manifest_path"])
            warps, inverses = gmc_transforms(stream_path, stream_manifest["data_sha256"], length, seq)
            gt, ignored = load_gt(gt_file, length)
            labels, gt_counts, count = clear_rows(dataset, seq, clear, linear_sum_assignment)
            assert all(gt_counts[f] == len(g) for f, g in gt.items()), seq
            assert all(count[k] == prior[seq][k] for k in EXPECTED), (seq, count, prior[seq])
            mot = read_mot(mot_file)
            assert set(mot) == set(labels), seq
            lifetime, profiles, signals = row_scan(seq, mot, labels, gt, event_path, cache, feature_cache,
                                                    warps, inverses, box_iou_matrix)
            archive, seq_profile = flatten(seq, lifetime, profiles)
            fps = [r for r in archive if r["matched_gt"] is None]
            cats = Counter(r["offline_category"] for r in fps)
            assert len(fps) == count["FP"], seq
            assert len(archive) == len(signals) == count["FP"] + count["TP"], seq
            if all_writer is None:
                def new_writer(name, sample):
                    out = stack.enter_context((args.output / name).open("x", encoding="utf-8-sig", newline=""))
                    writer = csv.DictWriter(out, fieldnames=list(sample), lineterminator="\n")
                    writer.writeheader()
                    return writer
                all_writer = new_writer("E3_all_rows_CLEAR.csv", archive[0])
                fp_writer = new_writer("E3_native_fp_rows.csv", fps[0])
                signal_writer = new_writer("E3_causal_signal_rows.csv", signals[0])
            for writer, records in ((all_writer, archive), (fp_writer, fps), (signal_writer, signals)):
                writer.writerows(records)
            sequence_rows.append({
                "sequence": seq, "original_video": original_video(seq), "frames": length,
                "TP": count["TP"], "FP": count["FP"], "FN": count["FN"], "IDSW": count.get("IDSW",0),
                "zero_marked_gt": ignored, "fp_categories": json.dumps(dict(cats), sort_keys=True),
                "persistent_nogt_candidates": sum(p["lifetime_unannotated_candidate_010"] for p in seq_profile),
            })
            all_categories.update(cats)
            totals.update(count)
            profile_rows.extend(seq_profile)
            raw_data[seq] = {"gt": gt_hash, "MOT": mot_hash, "events": event_hash,
                             "state": stream_manifest["data_sha256"], "detection": protocol["detection_fingerprints"][seq],
                             "feature": protocol["feature_hashes"][seq]}
            print(seq, count, dict(cats), flush=True)
    assert all(totals[k] == prior["COMBINED_SEQ"][k] == EXPECTED[k] for k in EXPECTED), totals
    csv_output(args.output, "E3_trajectory_profiles.csv", profile_rows)
    csv_output(args.output, "E3_sequence_summary.csv", sequence_rows)
    sensitivity = {}
    for t in THRESHOLDS:
        selection = [p for p in profile_rows if p[f"lifetime_unannotated_candidate_{int(t * 100):03}"]]
        sensitivity[str(t)] = {"candidate_tracks": len(selection), "all_clear_fp_rows_on_candidate_tracks": sum(p["clear_fp"] for p in selection),
                               "sequences": len({p["sequence"] for p in selection}),
                               "original_videos": len({p["original_video"] for p in selection})}
    manifest = {
        "status": "PASS", "baseline": BASE, "expected_historical_CLEAR": EXPECTED, "actual_CLEAR": totals,
        "per_sequence_exact": True, "seq_count": len(sequences), "frame_count": sum(sequences.values()),
        "trackeval_commit": PIN, "trackeval_sources_verified": len(sources), "input_sha256": raw_data,
        "run_protocol_sha256": fingerprint(protocol_path), "p0_sha256": fingerprint(args.p0),
        "historical_clear_summary_sha256": fingerprint(args.historical_eval / "summary.json"),
        "audit_source_sha256": fingerprint(Path(__file__)), "fp_categories": all_categories,
        "persistent_nogt_sensitivity": sensitivity, "definition": {
            "persistent": "at least 3 actual measurements at frames with official GT and maximum IoU of all lifetime measured/produced rows with any marked GT below threshold",
            "empty_GT": "kept as annotation unknown, never automatically background",
            "signals": "strictly separate CSV with only current and earlier native observations, detections, feature-cache RGB, GMC and metadata",
        },
        "outputs_sha256": {f.name: fingerprint(f) for f in sorted(args.output.glob("*.csv"))},
    }
    with (args.output / "E3_A_evaluation_audit.json").open("x", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2, ensure_ascii=False)
        output.write("\n")
    print("E3_A_PASS", totals, all_categories, sensitivity, flush=True)


if __name__ == "__main__":
    main()
