"""Recheck the geometry of output-only VT-Tiny-MOT supplements against official GT."""
from collections import Counter, defaultdict
from csv import DictReader
from hashlib import sha256
from io import TextIOWrapper
from pathlib import Path
import argparse
import json
import math
from zipfile import ZipFile


def rows(archive, name):
    with archive.open(name) as handle:
        return list(DictReader(TextIOWrapper(handle, encoding="utf-8-sig")))


def trajectory(archive, name, seq):
    with archive.open(f"full/trajectories/{name}/{seq}.txt") as handle:
        result = {}
        for line in handle:
            frame, tid, x, y, w, h, score, *_ = line.decode("ascii").split(",")
            x, y, w, h = map(float, (x, y, w, h))
            result[int(frame), int(tid)] = (x, y, x + w, y + h, float(score))
    return result


def truth(path, digest):
    content = path.read_bytes()
    assert sha256(content).hexdigest() == digest, path
    frames = defaultdict(dict)
    for line in content.decode("utf-8-sig").splitlines():
        if not line.strip():
            continue
        frame, tid, x, y, w, h, score, *_ = line.split(",")
        assert int(float(score)) == 1, (path, score)
        x, y, w, h = map(float, (x, y, w, h))
        frames[int(frame)][int(tid)] = (x, y, x + w, y + h)
    return frames


def iou(a, b):
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = w * h
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union


def center_gap_normalized(a, b):
    """Center distance divided by the truth box diagonal."""
    dx = ((a[0]+a[2]) - (b[0]+b[2])) / 2
    dy = ((a[1]+a[3]) - (b[1]+b[3])) / 2
    return math.hypot(dx, dy) / math.hypot(b[2]-b[0], b[3]-b[1])


def observed_geometry(box, refs):
    nearest = min(((center_gap_normalized(box, b), tid) for tid, b in refs.items()), default=None)
    best = max(((iou(box, b), tid) for tid, b in refs.items()), default=(0.0, None))
    return best, nearest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("ground_truth", type=Path)
    args = parser.parse_args()
    by_class, sequence_bins, cases = Counter(), defaultdict(Counter), {}
    historical_support = defaultdict(Counter)
    ids_events = {("DJI_0339_2", tag) for tag in (2, 6)}
    known_keys = {
        ("DJI_0339_2", 48, 2), ("DJI_0339_2", 207, 2),
        ("DJI_0327_3", 182, 27), ("DJI_0327_3", 268, 7),
        ("DJI_0327_3", 232, 41), ("DJI_0327_3", 289, 48),
        ("DJI_0061_2", 169, 177), ("DJI_0073_3", 92, 63), ("DJI_0327_3", 316, 62),
    }
    ids_hist = {}
    closest_fp = []
    with ZipFile(args.package) as archive:
        official = json.loads(archive.read("full/trackeval/evaluation_audit.json"))
        synthetic = rows(archive, "parity/full/supplemental_rows.csv")
        diag = {(r["sequence"], int(r["frame"]), int(r["track_id"])): r
                for r in rows(archive, "identity/full/output_only_truth.csv")}
        assert len(synthetic) == len(diag) == 191
        for seq in sorted({r["sequence"] for r in synthetic}):
            marks = truth(args.ground_truth / seq / "gt/gt.txt", official["ground_truth"]["sha256"][seq])
            native = trajectory(archive, "B_early_moderate", seq)
            new = trajectory(archive, "B_postlow_output_v1", seq)
            for row in (r for r in synthetic if r["sequence"] == seq):
                frame, tid = int(row["frame"]), int(row["track_id"])
                classification = "TP" if row["clear_gt"] else "FP"
                box = new[frame, tid]
                (best_iou, best_id), nearest = observed_geometry(box, marks[frame])
                assert math.isclose(best_iou, float(row["best_gt_iou"]), abs_tol=1e-6), (seq, frame, tid)
                key = seq, frame, tid
                d = diag[key]
                if classification == "FP":
                    if not marks[frame]:
                        bin_ = "no_annotated_target"
                    elif best_iou == 0 and nearest[0] > 1:
                        bin_ = "zero_iou_and_far_center"
                    elif best_iou == 0:
                        bin_ = "zero_iou_but_near_center"
                    elif best_iou < .25:
                        bin_ = "iou_0_to_025"
                    elif best_iou < .5:
                        bin_ = "iou_025_to_050"
                    else:
                        bin_ = "iou_above_050_clear_conflict"
                    by_class[bin_] += 1
                    sequence_bins[seq][bin_] += 1
                    closest_fp.append(dict(seq=seq, frame=frame, tid=tid, iou=best_iou, nearest=nearest,
                                           box=box[:4], target=marks[frame].get(best_id),
                                           target_gt=best_id, prior_det_gt=row["prior_det_gt"],
                                           prior_output_gt=row["prior_output_gt"]))
                else:
                    by_class["TP"] += 1
                    sequence_bins[seq][row["attribution"]] += 1
                group = ("TP" if classification == "TP" else "FP_zero_iou" if best_iou == 0 else
                         "FP_partial_overlap" if best_iou < .50 else "FP_competition")
                historical_support[group].update(
                    total=1,
                    previous_det_and_native_output_support_same_gt=bool(
                        row["prior_det_gt"] and row["prior_det_gt"] == row["prior_output_gt"]),
                    recovery_raw_det_labelled=bool(d["recovery_gt_id"]),
                    prior_raw_det_labelled=bool(row["prior_det_gt"]),
                    recovery_and_previous_raw_det_support_same_gt=bool(
                        d["recovery_gt_id"] and d["recovery_gt_id"] == row["prior_det_gt"]),
                    recovered_gt_absent_from_original_matches=row["gt_absent_from_base_clear_matches"] == "True",
                    matched_truth_equals_recovery_truth=bool(
                        row["clear_gt"] and row["clear_gt"] == d["recovery_gt_id"]),
                )
                if key in known_keys:
                    observed_prior = native[frame - 1, tid]
                    restore_frame = int(row["last_recovery_frame"])
                    competing = []
                    if best_id is not None:
                        competing = sorted((iou(candidate, marks[frame][best_id]), other_id,
                                            iou(box, candidate))
                                           for (f, other_id), candidate in native.items()
                                           if f == frame and iou(candidate, marks[frame][best_id]) >= .5)
                    cases[str(key)] = {
                        "clear_gt": row["clear_gt"], "diagnostic_gt": diag[key]["emitted_gt_id"],
                        "score": box[4], "prediction_iou": float(row["prediction_iou"]),
                        "raw_box": box[:4], "gt_box": marks[frame].get(int(row["clear_gt"] or best_id or -1)),
                        "best_iou": best_iou, "nearest_center": nearest,
                        "previous_native": observed_prior, "previous_max_iou": observed_geometry(
                            observed_prior, marks[frame - 1])[0],
                        "recovery_box": native.get((restore_frame, tid)), "recovery_frame": restore_frame,
                        "recovery_gt": diag[key]["recovery_gt_id"],
                        "native_outputs_also_covering_best_gt": competing,
                    }
            for key in (k for k in ids_events if k[0] == seq):
                tid = key[1]
                history = [(f, b, observed_geometry(b, marks[f])[0][0])
                           for (f, i), b in native.items() if i == tid]
                ids_hist[str(key)] = dict(
                    hits=len(history), start=history[0][0], end=history[-1][0],
                    max_gt_iou=max(v for _, _, v in history),
                    min_max_native_score=score_bounds(history),
                ) if history else {}
    data = dict(bins=by_class, sequence_bins=sequence_bins, histories=ids_hist,
                historical_GT_support=historical_support,
                closest_fp=sorted(closest_fp, key=lambda r: r["iou"], reverse=True)[:8], cases=cases)
    print(json.dumps(data, ensure_ascii=False, indent=2))


def score_bounds(history):
    return min(b[4] for _, b, _ in history), max(b[4] for _, b, _ in history)


if __name__ == "__main__":
    main()
