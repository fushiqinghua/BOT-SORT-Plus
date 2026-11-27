"""Register the unchanged development split, and optionally draw a diagnostic review sample."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import random


def read_json(path):
    with path.open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def evidence(path):
    return {"path": str(path), "sha256": sha256(path)}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def load_development_grouping(path, official_development):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    mapping = {}
    for row in rows:
        sequence, source = row["sequence"].strip(), row["original_video"].strip()
        if not sequence or not source or sequence in mapping:
            raise ValueError(f"Blank or duplicate development grouping entry: {sequence!r}")
        mapping[sequence] = source
    if set(mapping) != official_development:
        raise ValueError("Archived development grouping and official test sequence lists differ")
    return dict(sorted(mapping.items()))


def inspect_coco(path):
    data = read_json(path)
    sequences = {}
    metadata_fields = set()
    for row in data["videos"]:
        directory = PurePosixPath(row["name"])
        if len(directory.parts) != 2 or directory.parts[-1] != "00" or row["id"] in sequences:
            raise ValueError(f"Unexpected RGB video metadata: {row!r}")
        sequences[row["id"]] = directory.parts[0]
        metadata_fields.update(row)
    if len(set(sequences.values())) != len(sequences):
        raise ValueError("Official COCO contains duplicate RGB sequence names")
    frame_counts = Counter()
    for image in data["images"]:
        sequence = sequences[image["video_id"]]
        if PurePosixPath(image["file_name"]).parts[:2] != (sequence, "00"):
            raise ValueError(f"Image video_id and RGB file path disagree for {sequence}")
        frame_counts[sequence] += 1
    return {"evidence": evidence(path), "video_fields": sorted(metadata_fields),
            "registered_video_count": len(sequences), "image_count": len(data["images"]),
            "registered_sequences": sorted(sequences.values()), "frames_by_sequence": dict(sorted(frame_counts.items()))}


def manifest(args):
    config = read_json(args.settings)
    assets = config["paths"]
    historical = Path(assets["split_manifest"])
    splits = read_json(historical)["splits"]
    prior_training = set(splits["train"]) | set(splits["val"])
    development = set(splits["test"])
    if len(development) != 30 or len(splits["test"]) != 30 or development & prior_training:
        raise ValueError("Unexpected official development scope or repeated sequence entries")
    grouping = load_development_grouping(args.grouping, development)
    annotation_dir = Path(assets["images_root"]) / "annotations"
    train = inspect_coco(annotation_dir / "instances_00_train2017.json")
    test = inspect_coco(annotation_dir / "instances_00_test2017.json")
    if set(test["frames_by_sequence"]) != development or test["image_count"] != 12590:
        raise ValueError("Official COCO test clips disagree with the registered 30-clip development scope")
    registered_train = set(train["registered_sequences"])
    if prior_training - registered_train:
        raise ValueError("Historical detector split contains unregistered training clips")
    missing_images = sorted(s for s in train["frames_by_sequence"]
                            if not (Path(assets["train_images_root"]) / s / "00").is_dir())
    return {
        "schema": "tco_v0.1_split_manifest",
        "status": "DEVELOPMENT_SCOPE_LOCKED_TRAINING_PARENT_PROVENANCE_PENDING",
        "training_or_calibration_ran": False,
        "seed": None,
        "fit": None,
        "calibration": None,
        "holdout": None,
        "development": sorted(development),
        "development_sequence_to_source_video": grouping,
        "development_grouping": {
            "evidence": evidence(args.grouping),
            "source_video_count": len(set(grouping.values())),
            "scope": "Historical E3 original_video grouping for descriptive development reporting",
            "provenance": "E3 derives its group labels from the sequence naming proxy. The labels do not certify official training mother-video separation."
        },
        "official_training_annotation": train,
        "official_development_annotation": test,
        "historical_detector_split": {
            "evidence": evidence(historical),
            "train_and_val_sequence_count": len(prior_training),
            "registered_annotated_train_clips_excluded_from_legacy_split": sorted(registered_train - prior_training),
            "train_sequences_missing_rgb_directory": missing_images,
        },
        "training_blocker": (
            "The official RGB metadata identifies individual clips and has no mother-video origin field. "
            "A provenance-backed explicit train/development source-video map and complete aligned frozen training caches are needed before fitting."
        )
    }


def read_episodes(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def review_sample(args):
    source_mapping = read_json(args.manifest)["development_sequence_to_source_video"]
    rng = random.Random(args.seed)
    fields = (
        "method", "sequence", "frame", "track_id", "assigned_det_index",
        "primary_bucket", "gt_short_side_bin", "raw_iou", "gt_exact_reachable",
        "feasible_iou", "bounded_cross_050", "identity_status", "raw_xyxy", "gt_xyxy"
    )
    cells = defaultdict(list)
    seen = set()
    for episode in read_episodes(args.episodes):
        key = tuple(episode[name] for name in fields[:4])
        if key in seen:
            raise ValueError(f"Duplicate episode identity {key}")
        seen.add(key)
        group = source_mapping[episode["sequence"]]
        values = {name: episode.get(name) for name in fields}
        stratum = (str(values["method"]), group, str(values["primary_bucket"]),
                   str(values["gt_short_side_bin"]), str(values["bounded_cross_050"]))
        cells[stratum].append({**values, "development_source_video": group})
    if not cells:
        raise ValueError("Episode list is empty")

    # Randomize which diagnostic cells enter when the budget is below the
    # number of cells. Otherwise assign counts evenly, before sampling rows.
    ordered = sorted(cells)
    if len(ordered) > args.budget:
        rng.shuffle(ordered)
    used = ordered[:args.budget]
    quotas = {key: 1 for key in used}
    remaining = min(args.budget, len(seen)) - len(used)
    while remaining > 0:
        for key in used:
            if quotas[key] < len(cells[key]):
                quotas[key] += 1
                remaining -= 1
                if remaining == 0:
                    break
    p_cell = len(used) / len(ordered)
    sample = []
    for key, quota in quotas.items():
        options = sorted(cells[key], key=lambda row: (row["sequence"], row["frame"], row["track_id"]))
        for row in rng.sample(options, quota):
            sample.append({
                **row,
                "stratum_population": len(options),
                "stratum_sampled": quota,
                "inclusion_probability": p_cell * quota / len(options),
                "review_status": "PENDING_HUMAN_REVIEW",
            })
    for index, row in enumerate(sample, 1):
        row["case_id"] = f"TCO_T1_{index:03d}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8-sig", newline="") as stream:
        out = csv.DictWriter(stream, fieldnames=["case_id"] + [k for k in sample[0] if k != "case_id"])
        out.writeheader()
        out.writerows(sample)
    write_json(args.output.with_suffix(".json"), {
        "status": "SAMPLE_DRAWN_NOT_REVIEWED",
        "episodes": evidence(args.episodes),
        "seed": args.seed,
        "budget": args.budget,
        "sample_count": len(sample),
        "population": len(seen),
        "all_cells": len(cells),
        "sampled_cells": len(used),
        "strata": ["method", "archived_development_source_video", "primary_bucket",
                    "gt_short_side_bin", "bounded_cross_050"],
        "draw": "Uniform sample of cells if budget is restrictive, then uniform draw of an evenly allocated count within each selected cell",
        "human_reviews": 0,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    split = commands.add_parser("manifest")
    split.add_argument("--settings", type=Path, required=True)
    split.add_argument("--grouping", type=Path, required=True)
    split.add_argument("--output", type=Path, required=True)
    review = commands.add_parser("review")
    review.add_argument("--episodes", type=Path, required=True)
    review.add_argument("--manifest", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)
    review.add_argument("--budget", type=int, default=160, choices=range(1, 201), metavar="1..200")
    review.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()
    if args.command == "manifest":
        result = manifest(args)
        write_json(args.output, result)
        print(result["status"], len(result["development"]), result["development_grouping"]["source_video_count"])
    else:
        review_sample(args)


if __name__ == "__main__":
    main()
