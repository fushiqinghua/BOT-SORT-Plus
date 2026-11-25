"""Offline one-to-one coverage of a fixed detection cache, never an online oracle."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .diagnostics import match_boxes, size_bin
from .io import detection_files, dump_json, load_detections, load_gt, sha256


POOLS = ("all_cached", "high", "association_eligible", "birth_eligible")


def candidate_pools(detections):
    """Emulate frozen replay's score representation and strict comparisons.

    The runner casts cache rows to float32, then BoxMOT v16.0.11 appends int64
    indices with ``hstack``, promoting rows to float64 before thresholding.
    Consequently a JSON literal 0.1 rounds slightly above the 0.1 threshold;
    exact 0.5 is representable and remains outside both association stages.
    Do not compare directly in float32 or use epsilon-adjusted thresholds.
    """
    dets = np.asarray(detections, dtype=float).reshape(-1, 6)
    scores = dets[:, 4].astype(np.float32).astype(np.float64)
    return {
        "all_cached": np.arange(len(dets)),
        "high": np.flatnonzero(scores > .5),
        "association_eligible": np.flatnonzero((scores > .5) | ((scores > .1) & (scores < .5))),
        "birth_eligible": np.flatnonzero(scores >= .6),
    }


def diagnose_detection_coverage(cache_dir, gt_root, output, *, sequences, thresholds=(.3, .5, .75)):
    """Maximum-cardinality IoU matching for each frame/pool/threshold.

    The cardinality bonus dominates all IoU tie-breaking. Counting each GT's
    best overlap independently would let one detection cover several targets.
    Size rows partition one global matching: they are not separately optimized
    per-size upper bounds. This bound applies only to selected raw cache boxes,
    not to filtered/predicted boxes, fresh detections, MOTA, HOTA or identity.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Use a new coverage output: {output}")
    thresholds = tuple(float(x) for x in thresholds)
    if not thresholds or len(set(thresholds)) != len(thresholds) or any(not 0 < x <= 1 for x in thresholds):
        raise ValueError("IoU thresholds must be unique and within (0,1]")
    files = detection_files(cache_dir)
    names = list(sequences)
    if not names or len(set(names)) != len(names) or set(names) - set(files):
        raise ValueError("Select unique available sequences")
    output.mkdir(parents=True)
    manifest = {"status": "RUNNING", "interpretation": "retrospective_development_only",
                "thresholds": list(thresholds), "sequences": names, "inputs": {},
                "definition": "maximum-cardinality one-to-one IoU matching of fixed raw detections",
                "limitations": ["Not an upper bound on filtered boxes, MOTA, HOTA or identity metrics.",
                                "No temporal lifecycle or birth/recovery restrictions are simulated.",
                                "Size strata partition a single global matching; no separate size optimization."]}
    dump_json(output / "manifest.json", manifest)
    counts = defaultdict(lambda: {"gt": 0, "matched": 0})
    try:
        with (output / "frames.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["sequence", "frame", "pool", "iou_threshold",
                                                       "gt", "detections", "matched", "unmatched_gt"])
            writer.writeheader()
            for name in names:
                cache = load_detections(files[name])
                gt = load_gt(gt_root, name)
                frame_count = len(cache["frames"])
                if set(gt) - set(range(1, frame_count + 1)):
                    raise ValueError(f"GT frame outside detection cache: {name}")
                manifest["inputs"][name] = {"detections_sha256": sha256(files[name]),
                    "gt_sha256": sha256(Path(gt_root) / name / "gt/gt.txt"), "frames": frame_count}
                for frame in cache["frames"]:
                    truth = gt.get(frame["frame"], np.empty((0, 5)))
                    dets = np.asarray(frame["detections"], dtype=float).reshape(-1, 6)
                    for pool, indices in candidate_pools(dets).items():
                        for threshold in thresholds:
                            matches, _ = match_boxes(truth[:, :4], dets[indices, :4], threshold)
                            writer.writerow({"sequence": name, "frame": frame["frame"], "pool": pool,
                                "iou_threshold": threshold, "gt": len(truth), "detections": len(indices),
                                "matched": len(matches), "unmatched_gt": len(truth) - len(matches)})
                            # Include empty-GT sequences explicitly in the summary.
                            for seq in (name, "COMBINED_SEQ"):
                                counts[(seq, pool, threshold, "all")]["gt"] += len(truth)
                                counts[(seq, pool, threshold, "all")]["matched"] += len(matches)
                                for i, box in enumerate(truth):
                                    entry = counts[(seq, pool, threshold, size_bin(box[:4]))]
                                    entry["gt"] += 1
                                    entry["matched"] += int(i in matches)
        rows = []
        for (name, pool, threshold, size), count in sorted(counts.items()):
            rows.append({"sequence": name, "pool": pool, "iou_threshold": threshold, "size": size,
                         **count, "unmatched_gt": count["gt"] - count["matched"],
                         "coverage_percent": 100 * count["matched"] / count["gt"] if count["gt"] else None})
        with (output / "coverage.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        manifest["status"] = "COMPLETE"
        manifest["results"] = rows
        dump_json(output / "summary.json", manifest)
        dump_json(output / "manifest.json", manifest)
        return manifest
    except Exception as exc:
        manifest.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        dump_json(output / "manifest.json", manifest)
        raise
