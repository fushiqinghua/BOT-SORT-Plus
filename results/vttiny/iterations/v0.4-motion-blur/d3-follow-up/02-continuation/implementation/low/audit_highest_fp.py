"""Trace the largest false-positive increase to emitted tracks with CLEAR's matching."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np


BASE, VARIANT = ("B_early_moderate", "B_postlow_v1")


def load_events(at, method, sequence):
    path = at / "events" / method / (sequence + ".jsonl")
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def classify_all_rows(dataset, method, sequence, assign):
    raw = dataset.get_raw_seq_data(method, sequence)
    prepared = dataset.get_preprocessed_seq_data(raw, "object")
    epsilon = np.finfo(float).eps
    previous = np.full(prepared["num_gt_ids"], np.nan)
    result = {}
    for ti, (truth, tracks, original, similarities) in enumerate(zip(
        prepared["gt_ids"], prepared["tracker_ids"], raw["tracker_ids"],
        prepared["similarity_scores"],
    )):
        assert len(original) == len(tracks)
        matched = set()
        if len(truth) and len(tracks):
            # Mirror the pinned CLEAR scorer's previous-association priority.
            priority = (tracks[np.newaxis, :] == previous[truth[:, np.newaxis]])
            scores = 1000 * priority + similarities
            scores[similarities < .5 - epsilon] = 0
            rows, cols = assign(-scores)
            keep = scores[rows, cols] > epsilon
            rows, cols = rows[keep], cols[keep]
            previous[:] = np.nan
            previous[truth[rows]] = tracks[cols]
            matched = set(cols)
        for ci, track_id in enumerate(original):
            result[(ti + 1, int(track_id))] = {
                "false_positive": ci not in matched,
                "best_gt_iou": float(np.max(similarities[:, ci])) if len(truth) else 0.,
            }
    return result


def main():
    p = ArgumentParser(description=__doc__)
    for arg in ("suite", "result"):
        p.add_argument(arg, type=Path)
    args = p.parse_args()
    project = args.suite.parents[1]
    comparison = args.suite / "round1/d3_next_comparison_v1"
    frozen = args.suite / "BOT-SORT-Plus/outputs/diagnostics_v0.4_full30_round1/observed"
    variant = Path(__file__).resolve().parent / "variant_full"
    with (comparison / "sequence_metrics.csv").open(encoding="utf-8-sig") as stream:
        table = list(csv.DictReader(stream))
    top = max((r for r in table if r["method"] == VARIANT), key=lambda r: int(r["delta_FP"]))
    sequence = top["sequence"]
    seqs = json.loads((variant / "protocols" / (VARIANT + ".json")).read_bytes())["sequences"]
    sys.path.insert(0, str(project / "external/TrackEval"))
    for key, cast in [("int", int), ("float", float), ("bool", bool)]:
        if key not in np.__dict__:
            setattr(np, key, cast)
    from trackeval.datasets import MotChallenge2DBox
    from trackeval.metrics.clear import linear_sum_assignment

    dataset = MotChallenge2DBox({
        "GT_FOLDER": str(project / "data/processed_v1/mot/rgb/test"),
        "TRACKERS_FOLDER": str(comparison / "comparison_input/trackers"),
        "TRACKERS_TO_EVAL": [BASE, VARIANT], "BENCHMARK": "VT-Tiny-MOT",
        "SPLIT_TO_EVAL": "test", "SKIP_SPLIT_FOL": True, "SEQ_INFO": {sequence: seqs[sequence]},
        "DO_PREPROC": False, "PRINT_CONFIG": False, "CLASSES_TO_EVAL": ["pedestrian"],
    })
    dataset.valid_classes = ["object"]
    dataset.class_list = ["object"]
    dataset.class_name_to_class_id["object"] = 1
    before, after = [classify_all_rows(dataset, method, sequence, linear_sum_assignment)
                     for method in (BASE, VARIANT)]
    previous_fp = {k for k, v in before.items() if v["false_positive"]}
    current_fp = {k for k, v in after.items() if v["false_positive"]}
    added, removed = current_fp - previous_fp, previous_fp - current_fp
    assert len(current_fp) == int(top["FP"])
    assert len(added) - len(removed) == int(top["delta_FP"])
    old_events = load_events(frozen, BASE, sequence)
    new_events = load_events(variant, VARIANT, sequence)
    direct = {(e["frame"], e["track_id"]) for e in new_events if e.get("stage") == "post_B_weak"}
    biggest_track = Counter(tid for _, tid in added).most_common(1)[0][0]
    indexed = defaultdict(list)
    for event in new_events:
        if event.get("track_id") == biggest_track and event["event"] in {"match", "recovered"}:
            indexed[event["frame"]].append({k: event[k] for k in (
                "event", "stage", "det_index", "age", "score", "cost", "b_recovery_frame"
            ) if k in event})
    after_frames = sorted(f for f, tid in after if tid == biggest_track)
    before_frames = sorted(f for f, tid in before if tid == biggest_track)
    rows = [{"frame": f, "direct_weak": (f, biggest_track) in direct,
             **after[(f, biggest_track)], "original_mot_emitted": (f, biggest_track) in before,
             "variant_match": indexed[f]} for f in after_frames]
    removed_counts = dict(Counter(tid for _, tid in removed))
    added_counts = dict(Counter(tid for _, tid in added))
    report = {
        "sequence": sequence, "official": top, "baseline_fp": len(previous_fp),
        "variant_fp": len(current_fp), "added_fp": len(added), "removed_fp": len(removed),
        "direct_trigger_emissions": len(direct), "fp_from_direct_trigger": len(direct & current_fp),
        "added_fp_from_direct_trigger": len(direct & added),
        "added_fp_after_other_output_paths": len(added - direct),
        "added_fp_by_track_id": added_counts, "removed_fp_by_track_id": removed_counts,
        "main_track_id": biggest_track, "main_track_before_frame_count": len(before_frames),
        "main_track_after_frame_count": len(after_frames),
        "main_track_after": rows, "main_track_before": [
            {"frame": f, **before[(f, biggest_track)]} for f in before_frames
        ],
        "baseline_recovery_frames": [e["frame"] for e in old_events
            if e["event"] == "recovered" and e["track_id"] == biggest_track],
    }
    with args.result.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print({k: v for k, v in report.items() if k not in {
        "main_track_before", "main_track_after", "baseline_recovery_frames",
    }})
    print([r for r in rows if r["frame"] in (47, 48, 49, 205, 206, 207, 236, 251, 599)])


if __name__ == "__main__":
    main()
