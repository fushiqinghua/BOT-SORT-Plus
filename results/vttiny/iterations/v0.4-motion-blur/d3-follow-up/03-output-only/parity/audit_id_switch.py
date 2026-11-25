"""Locate CLEAR ID switches and separate scoring history from tracker state."""
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import sys

import numpy as np

from check_output_only import (BASE, NEW, OLD, ROW_EVENT, clear_row_assignment,
                               config_paths, event_pairs, frame_track, protocol)


def switches(assignments, synthetic):
    previous = {}
    changes = {}
    for key, row in sorted(assignments.items()):
        frame, track = key
        gt = row["gt_id"]
        if gt is None:
            continue
        before = previous.get(gt)
        if before is not None and before[1] != track:
            changes[frame, gt] = {
                "frame": frame, "gt": gt, "from_id": before[1], "to_id": track,
                "from_frame": before[0], "from_supplement": before in synthetic,
                "to_supplement": key in synthetic,
            }
        previous[gt] = key
    return changes


def make_diff(base, candidate):
    return {
        "introduced": [event for k, event in candidate.items() if k not in base],
        "removed": [event for k, event in base.items() if k not in candidate],
        "different_transition_same_frame": [
            {"before": base[k], "after": candidate[k]} for k in candidate.keys() & base.keys()
            if (candidate[k]["from_id"], candidate[k]["to_id"]) != (base[k]["from_id"], base[k]["to_id"])
        ],
    }


def main():
    p = ArgumentParser(description=__doc__)
    for field in ("project", "base_root", "old_root", "new_root", "output"):
        p.add_argument(field, type=Path)
    a = p.parse_args()
    sys.path.insert(0, str(a.project / "external/TrackEval"))
    for name, value in (("int", int), ("float", float), ("bool", bool)):
        if name not in np.__dict__:
            setattr(np, name, value)
    import trackeval
    from trackeval.metrics.clear import linear_sum_assignment
    roots = {BASE: a.base_root, OLD: a.old_root, NEW: a.new_root}
    seqs = protocol(a.new_root, NEW)["sequences"]
    gt = a.project / "data/official_gt/GT/GT_00/test"
    reports = []
    clear = trackeval.metrics.CLEAR({"PRINT_CONFIG": False})
    with TemporaryDirectory(prefix=".idsw_", dir=a.output.parent) as temp:
        for name, root in roots.items():
            (Path(temp) / name).symlink_to(root / "trackers" / name, target_is_directory=True)
        dataset = trackeval.datasets.MotChallenge2DBox({
            "GT_FOLDER": str(gt), "TRACKERS_FOLDER": temp, "TRACKERS_TO_EVAL": list(roots),
            "BENCHMARK": "VT-Tiny-MOT", "SPLIT_TO_EVAL": "test", "SKIP_SPLIT_FOL": True,
            "SEQ_INFO": seqs, "DO_PREPROC": False, "PRINT_CONFIG": False,
            "CLASSES_TO_EVAL": ["pedestrian"],
        })
        dataset.valid_classes = dataset.class_list = ["object"]
        dataset.class_name_to_class_id["object"] = 1
        for seq in seqs:
            new_trace, _ = config_paths(a.new_root, NEW, seq)
            supplemental = {frame_track(e) for _, e in event_pairs(new_trace) if e["event"] == ROW_EVENT}
            runs = {}
            for name in roots:
                assigned, metrics = clear_row_assignment(dataset, name, seq, clear, linear_sum_assignment)
                runs[name] = switches(assigned, supplemental if name == NEW else set())
                assert len(runs[name]) == metrics["IDSW"], (seq, name, metrics)
            diff = make_diff(runs[BASE], runs[NEW])
            reports.append({"sequence": seq, "IDSW": {k: len(v) for k, v in runs.items()}, **diff})
    payload = {"sequences": reports, "definition": "CLEAR counts an identity switch against the last matched "
              "ID for a GT, across any length of no match. A supplemental output can shift the frame scored."}
    with a.output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print("Total IDSW", dict(sum((Counter(row["IDSW"]) for row in reports), Counter())))
    for row in reports:
        if row["introduced"] or row["removed"]:
            print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
