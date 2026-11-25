"""Prove additive output parity and classify the added rows against official VT-Tiny-MOT GT."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import json
import math
import sys

import numpy as np


BASE = "B_early_moderate"
OLD = "B_postlow_v1"
NEW = "B_postlow_output_v1"
ROW_EVENT = "output_only_low"
ROW_STAGE = "post_B_weak_output"
PROBE_NAMES = {"output_only_low_check", "post_B_weak_output_check"}
MEASURED = {"match", "recovered", "native_birth"}
MAP_FIELDS = ("sequences", "detection_fingerprints", "feature_hashes", "decoded_image_hashes")
VALUE_FIELDS = ("native_config", "detector_sha256", "rgb_encoder_sha256", "source_protocol_sha256",
                "encoder_contracts", "boxmot_source_hashes")


def read_json(path):
    with path.open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def event_pairs(path):
    return [(record, json.loads(record)) for record in path.read_bytes().splitlines(keepends=True)]


def frame_track(event):
    return int(event["frame"]), int(event["track_id"])


def row_key(line):
    a, b, _ = line.split(b",", 2)
    return int(a), int(b)


def mot_box(line):
    fields = line.split(b",")
    x, y, w, h, score = map(float, fields[2:7])
    assert math.isfinite(score) and w > 0 and h > 0
    return [x, y, x + w, y + h], score


def read_mot(path):
    lines = path.read_bytes().splitlines(keepends=True)
    mapping = {row_key(line): line for line in lines}
    assert len(lines) == len(mapping), (path, "duplicate frame/ID")
    return lines, mapping


def config_paths(root, name, seq):
    return (root / "events" / name / f"{seq}.jsonl",
            root / "trackers" / name / "data" / f"{seq}.txt")


def protocol(root, name):
    return read_json(root / "protocols" / f"{name}.json")


def check_protocol(parents, scopes):
    frozen = scopes[BASE]
    new = scopes[NEW]
    assert all(p["status"] == "COMPLETE" and p["causal"]
               and not p["interpolation"] and not p["backfilled_frames"]
               for p in scopes.values())
    for name, p in scopes.items():
        for field in VALUE_FIELDS:
            assert p[field] == frozen[field], (name, field)
        assert {**p["config"], "name": BASE} == frozen["config"], name
        for field in MAP_FIELDS:
            for seq in new["sequences"]:
                assert p[field][seq] == frozen[field][seq], (name, field, seq)
        for seq in new["sequences"]:
            _, out = config_paths(parents[name], name, seq)
            assert digest(out) == p["output_prediction_hashes"][seq], (name, seq, "MOT SHA")


def check_sequence(paths, seq, length):
    ref_events, ref_text = config_paths(paths[BASE], BASE, seq)
    new_events, new_text = config_paths(paths[NEW], NEW, seq)
    reference = event_pairs(ref_events)
    active = event_pairs(new_events)
    additional = [e for _, e in active if e["event"] == ROW_EVENT]
    unchanged = [line for line, e in active if e["event"] not in PROBE_NAMES | {ROW_EVENT}]
    assert all(e.get("stage") == ROW_STAGE for e in additional)
    assert unchanged == [line for line, _ in reference], (seq, "native event bytes")
    assert sum(e["event"] == "frame" for _, e in reference) == length
    original_lines, original = read_mot(ref_text)
    replay_lines, replay = read_mot(new_text)
    extras = {frame_track(e) for e in additional}
    assert len(extras) == len(additional), (seq, "duplicate synthetic output")
    assert not extras.intersection(original), (seq, "overwrote native row")
    assert extras == replay.keys() - original.keys(), (seq, "unmarked missing or extra row")
    assert [line for line in replay_lines if row_key(line) not in extras] == original_lines, (
        seq, "native MOT bytes or ordering")
    # Keep every actual frame observation and B restore in frozen order. The synthetic
    # event itself never qualifies as a prior real observation or a B anchor.
    observation = defaultdict(list)
    restoration = defaultdict(list)
    used_dets = defaultdict(set)
    counts = Counter(e["event"] for _, e in reference)
    for _, e in reference:
        if e["event"] in MEASURED:
            frame, tid = frame_track(e)
            index = int(e["det_index"])
            observation[tid].append((frame, index))
            used_dets[frame].add(index)
            if e["event"] == "recovered":
                restoration[tid].append(frame)
    indices = {(int(e["frame"]), int(e["det_index"])) for e in additional}
    assert len(indices) == len(additional), (seq, "double assignment of a weak detector row")
    recovery_use = Counter()
    for e in additional:
        frame, tid = frame_track(e)
        past = [(f, d) for f, d in observation[tid] if f < frame]
        anchor = max((f for f in restoration[tid] if f < frame), default=0)
        assert past and past[-1][0] == frame - 1 == int(e["last_native_observation_frame"])
        assert not any(f == frame for f, _ in observation[tid]), (seq, frame, tid, "already matched")
        assert anchor == int(e["b_recovery_frame"]) and 1 <= frame - anchor <= 3
        assert int(e["det_index"]) not in used_dets[frame]
        assert .05 <= float(e["score"]) < .10 and 0 <= float(e["cost"]) <= .50
        recovery_use[tid, anchor] += 1
        assert recovery_use[tid, anchor] <= 2
        e["previous_native_det_index"] = past[-1][1]
    evidence = {
        "sequence": seq, "frames": length, "native_mot_rows": len(original_lines),
        "native_trace_rows": len(reference), "native_recovery_events": counts["recovered"],
        "native_appearance_updates": counts["memory_update"], "native_candidate_checks": counts["recovery_check"],
        "native_trace_sha256": digest(ref_events), "native_mot_sha256": digest(ref_text),
        "variant_trace_sha256": digest(new_events), "variant_mot_sha256": digest(new_text),
        "synthetic_rows": len(additional), "byte_exact_native_trace": True, "byte_exact_native_mot": True,
        "supplemental_probes": sum(e["event"] in PROBE_NAMES for _, e in active),
    }
    return additional, replay, evidence


def clear_row_assignment(dataset, tracker, seq, clear, assign):
    raw = dataset.get_raw_seq_data(tracker, seq)
    prepared = dataset.get_preprocessed_seq_data(raw, "object")
    last = np.full(prepared["num_gt_ids"], np.nan)
    epsilon = np.finfo(float).eps
    output = {}
    for frame, (truth, tracks, raw_gt, raw_tracks, similarity) in enumerate(zip(
        prepared["gt_ids"], prepared["tracker_ids"], raw["gt_ids"], raw["tracker_ids"],
        prepared["similarity_scores"],
    ), start=1):
        assert len(raw_gt) == len(truth) and len(raw_tracks) == len(tracks)
        matched = {}
        if len(truth) and len(tracks):
            priority = tracks[np.newaxis, :] == last[truth[:, np.newaxis]]
            score = 1000 * priority + similarity
            score[similarity < clear.threshold - epsilon] = 0
            rows, cols = assign(-score)
            valid = score[rows, cols] > epsilon
            rows, cols = rows[valid], cols[valid]
            last[:] = np.nan
            last[truth[rows]] = tracks[cols]
            matched = {int(c): int(raw_gt[r]) for r, c in zip(rows, cols)}
        for col, track in enumerate(raw_tracks):
            output[frame, int(track)] = {
                "gt_id": matched.get(col),
                "max_gt_iou": float(np.max(similarity[:, col])) if len(truth) else 0.0,
            }
    totals = {key: int(value) for key, value in clear.eval_sequence(prepared).items()
              if key in {"CLR_TP", "CLR_FP", "CLR_FN", "IDSW"}}
    assert sum(row["gt_id"] is None for row in output.values()) == totals["CLR_FP"]
    assert sum(row["gt_id"] is not None for row in output.values()) == totals["CLR_TP"]
    return output, totals


def classify_additions(seq, additional, motion, before, current, gt_path, cache_path, length,
                       load_gt, diagnostic_assignment, read_detection_cache):
    frames = read_detection_cache(cache_path)["frames"]
    gt, _ = load_gt(gt_path, length)
    relevant_frames = sorted({int(e["frame"]) - 1 for e in additional})
    observed = {f: diagnostic_assignment(np.asarray(frames[f - 1]["detections"])[:, :4], gt[f])[0]
                for f in relevant_frames}
    currently_matched = defaultdict(set)
    for (frame, _), row in before.items():
        if row["gt_id"] is not None:
            currently_matched[frame].add(row["gt_id"])
    rows = []
    for e in additional:
        frame, tid = frame_track(e)
        det = frames[frame - 1]["detections"][int(e["det_index"])]
        coords, conf = mot_box(motion[frame, tid])
        assert np.allclose(det[:4], e["xyxy"], atol=1e-5, rtol=1e-7), (seq, frame, "trace box")
        assert np.allclose(det[:4], coords, atol=1e-5, rtol=1e-7), (seq, frame, "written box")
        assert math.isclose(det[4], conf, abs_tol=1e-7)
        assert math.isclose(det[4], e["score"], abs_tol=1e-7)
        output = current[frame, tid]
        matched = output["gt_id"]
        anchor = observed[frame - 1].get(e["previous_native_det_index"])
        prior_prediction = before.get((frame - 1, tid), {}).get("gt_id")
        if matched is None:
            attribution = "CLEAR_false_positive"
        elif anchor == matched and prior_prediction == matched:
            attribution = "CLEAR_same_gt_and_prior_detection_agree"
        elif anchor is None or prior_prediction is None:
            attribution = "CLEAR_matched_prior_identity_unverified"
        else:
            attribution = "CLEAR_matched_prior_identity_disagrees"
        rows.append({
            "sequence": seq, "frame": frame, "track_id": tid, "det_index": int(e["det_index"]),
            "detector_score": conf, "last_native_observation_frame": frame - 1,
            "last_recovery_frame": int(e["b_recovery_frame"]), "prediction_iou": 1 - float(e["cost"]),
            "prior_det_gt": anchor, "prior_output_gt": prior_prediction, "clear_gt": matched,
            "best_gt_iou": output["max_gt_iou"], "attribution": attribution,
            "gt_absent_from_base_clear_matches": matched is not None and matched not in currently_matched[frame],
        })
    return rows


def run(args, target, roots, trackeval, load_gt, diagnostic_assignment, read_detection_cache):
    from trackeval.metrics.clear import linear_sum_assignment
    scopes = {name: protocol(path, name) for name, path in roots.items()}
    check_protocol(roots, scopes)
    sequences = scopes[NEW]["sequences"]
    assert len(sequences) in {2, 30}
    if len(sequences) == 2:
        assert set(sequences) == {"DJI_0339_2", "DJI_0327_3"}
    gt_root = args.project / "data/official_gt/GT/GT_00/test"
    gt_reference = read_json(args.prior_comparison / "official30/evaluation_audit.json")["ground_truth"]["sha256"]
    clear = trackeval.metrics.CLEAR()
    detail, sequences_out = [], []
    with TemporaryDirectory(prefix=".audit_clear_", dir=target) as temp:
        for name, root in roots.items():
            (Path(temp) / name).symlink_to(root / "trackers" / name, target_is_directory=True)
        dataset = trackeval.datasets.MotChallenge2DBox({
            "GT_FOLDER": str(gt_root), "TRACKERS_FOLDER": temp, "TRACKERS_TO_EVAL": list(roots),
            "BENCHMARK": "VT-Tiny-MOT", "SPLIT_TO_EVAL": "test", "SKIP_SPLIT_FOL": True,
            "SEQ_INFO": sequences, "DO_PREPROC": False, "PRINT_CONFIG": False,
            "CLASSES_TO_EVAL": ["pedestrian"],
        })
        dataset.valid_classes = dataset.class_list = ["object"]
        dataset.class_name_to_class_id["object"] = 1
        for seq, length in sequences.items():
            gt_path = gt_root / seq / "gt/gt.txt"
            assert digest(gt_path) == gt_reference[seq], (seq, "official GT SHA")
            additional, new_rows, evidence = check_sequence(roots, seq, length)
            labels, metrics = {}, {}
            for name in roots:
                labels[name], metrics[name] = clear_row_assignment(dataset, name, seq, clear, linear_sum_assignment)
            rows = classify_additions(
                seq, additional, new_rows, labels[BASE], labels[NEW], gt_path,
                args.repo / "results/vttiny/yolo11m_rgb_rgbt/rgbt/detections" / f"{seq}.json.gz",
                length, load_gt, diagnostic_assignment, read_detection_cache,
            )
            detail.extend(rows)
            stats = Counter(row["attribution"] for row in rows)
            native = labels[BASE]
            updated = labels[NEW]
            fp_delta = metrics[NEW]["CLR_FP"] - metrics[BASE]["CLR_FP"]
            assert fp_delta == (stats["CLEAR_false_positive"]
                                + sum(updated[k]["gt_id"] is None and v["gt_id"] is not None for k, v in native.items())
                                - sum(updated[k]["gt_id"] is not None and v["gt_id"] is None for k, v in native.items()))
            evidence["CLEAR"] = metrics
            evidence["synthetic_attribution"] = dict(stats)
            evidence["synthetic_missed_gt_recovered"] = sum(r["gt_absent_from_base_clear_matches"] for r in rows)
            evidence["native_CLEAR_assignment_changes"] = sum(updated[k]["gt_id"] != v["gt_id"]
                                                              for k, v in native.items())
            sequences_out.append(evidence)
            print(seq, "native exact", len(additional), dict(stats), "FP delta", fp_delta, flush=True)
    summary = {
        "status": "PASS", "rule": NEW, "sequence_count": len(sequences),
        "frames": sum(sequences.values()), "same_protocol_fields": [*MAP_FIELDS, *VALUE_FIELDS],
        "defined_same_gt": "synthetic row CLEAR assignment, prior real detection diagnostic GT, "
                           "and prior native MOT row CLEAR assignment all share same GT identity",
        "native_match_note": "CLEAR reassignment can occur through evaluation even with byte-exact native MOT rows",
        "synthetic_rows": len(detail), "synthetic_attribution": dict(Counter(r["attribution"] for r in detail)),
        "sequences": sequences_out,
    }
    if detail:
        with (target / "supplemental_rows.csv").open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(detail[0]))
            writer.writeheader()
            writer.writerows(detail)
    with (target / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print("OUTPUT_PARITY_PASS", {k: v for k, v in summary.items() if k != "sequences"}, flush=True)


def main():
    p = ArgumentParser(description=__doc__)
    for field in ("project", "repo", "base_root", "old_root", "new_root", "prior_comparison", "output"):
        p.add_argument(field, type=Path)
    args = p.parse_args()
    args.output.mkdir(exist_ok=False)
    sys.path[:0] = [str(args.project / "external/TrackEval"), str(args.repo)]
    for key, value in [("int", int), ("float", float), ("bool", bool)]:
        if key not in np.__dict__:
            setattr(np, key, value)
    import trackeval
    from botsort_plus.diagnostics.common import diagnostic_assignment, load_gt
    from botsort_plus.io import read_detection_cache
    run(args, args.output, {BASE: args.base_root, OLD: args.old_root, NEW: args.new_root},
        trackeval, load_gt, diagnostic_assignment, read_detection_cache)


if __name__ == "__main__":
    main()
