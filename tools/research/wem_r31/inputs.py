"""Validate existing R3 caches without images, models or tracker imports.

Selection inputs and offline labels remain separate. Hashes cover the exact
bytes decoded, including compressed streams, rather than a second later read.
"""
from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re

from .protocol import LABELS, PARTITIONS, PINS, REFERENCE, SETTINGS


def digest(data):
    return hashlib.sha256(data).hexdigest()


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def decode(data):
    def bad_constant(value):
        raise ValueError(f"Non-finite JSON number: {value}")
    return json.loads(data, object_pairs_hook=_pairs, parse_constant=bad_constant)


def read_json(path):
    payload = Path(path).read_bytes()
    return decode(payload), digest(payload)


def number(value, name, low=None, high=None, nullable=False):
    if nullable and value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"{name} is outside its permitted range")


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def boolean(value, name):
    if type(value) is not bool:
        raise ValueError(f"{name} must be boolean")


def edge_key(row):
    return row["sequence"], row["frame"], row["track_id"], row["det_index"]


def group_key(row):
    return row["sequence"], row["frame"], row["track_id"], tuple(row["raw_box_xyxy"])


def _box(box):
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError("raw_box_xyxy must have four numbers")
    for value in box:
        number(value, "box coordinate")
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("Candidate box has non-positive area")


def _identity(row):
    if not isinstance(row, dict):
        raise ValueError("A candidate must be an object")
    if not isinstance(row["sequence"], str) or not re.fullmatch(r"[A-Za-z0-9_-]+", row["sequence"]):
        raise ValueError("Invalid sequence identifier")
    integer(row["frame"], "frame", 1)
    integer(row["track_id"], "track_id", 0)
    integer(row["det_index"], "det_index", 0)
    _box(row["raw_box_xyxy"])
    number(row["raw_score"], "S0 raw score", 0, 0.1)


FEATURE_FIELDS = (
    "reference", "sequence", "frame", "track_id", "det_index", "raw_box_xyxy", "raw_score",
    "native_iou", "normalized_score", "cosine", "motion", "history_score",
    "geometry_group_rank", "row_unknown", "column_unknown", "row_margin", "column_margin",
    "rgb_weak_valid", "valid_history_rgb", "previous_regular_observations", "crop_wh", "challengers",
)


def clean_feature(row):
    _identity(row)
    if row["reference"] != REFERENCE:
        raise ValueError("Candidate reference changed")
    number(row["native_iou"], "native_iou", 0.5, 1)
    number(row["normalized_score"], "normalized_score", 0, 1)
    expected = max(0., min(1., (row["raw_score"] - .05) / .05))
    if abs(row["normalized_score"] - expected) > 1e-12:
        raise ValueError("Normalized score differs from the frozen R3 definition")
    number(row["cosine"], "cosine", -1, 1, nullable=True)
    number(row["motion"], "motion", 0, 1, nullable=True)
    number(row["history_score"], "history_score", 0, 1, nullable=True)
    for name in ("row_margin", "column_margin"):
        number(row[name], name, -1, 1, nullable=True)
    for name in ("row_unknown", "column_unknown", "rgb_weak_valid"):
        boolean(row[name], name)
    for name in ("valid_history_rgb", "previous_regular_observations"):
        integer(row[name], name)
    integer(row["geometry_group_rank"], "geometry_group_rank", 1)
    if row["cosine"] is not None and (not row["rgb_weak_valid"] or row["valid_history_rgb"] < 2):
        raise ValueError("Own cosine has no two-observation RGB support")
    expected_history = None
    if row["cosine"] is not None and row["motion"] is not None:
        expected_history = (row["native_iou"] + expected + (row["cosine"] + 1) / 2 + row["motion"]) / 4
    if ((expected_history is None) != (row["history_score"] is None)
            or expected_history is not None and abs(expected_history - row["history_score"]) > 1e-10):
        raise ValueError("Historical score differs from original R3 components")
    crop = row["crop_wh"]
    if not isinstance(crop, list) or len(crop) != 2:
        raise ValueError("crop_wh must contain two integers")
    for value in crop:
        integer(value, "crop side")
    if not isinstance(row["challengers"], list):
        raise ValueError("challengers must be a list")
    rivals, seen = [], set()
    for item in row["challengers"]:
        integer(item["track_id"], "challenger track_id")
        if item["track_id"] == row["track_id"] or item["track_id"] in seen:
            raise ValueError("Repeated or self challenger")
        seen.add(item["track_id"])
        number(item["cosine"], "challenger cosine", -1, 1, nullable=True)
        number(item["native_iou"], "challenger native_iou", 0.5, 1)
        integer(item["history_observations"], "challenger history_observations")
        if not isinstance(item["source"], str):
            raise ValueError("Challenger source must be a string")
        rivals.append({k: item[k] for k in ("track_id", "source", "history_observations", "native_iou", "cosine")})
    # This whitelist intentionally excludes every offline label field.
    result = {name: row[name] for name in FEATURE_FIELDS}
    result["challengers"] = rivals
    return result


def load_settings(path, root):
    data, fingerprint = read_json(path)
    if set(data) != set(SETTINGS) or data["schema"] != SETTINGS["schema"]:
        raise ValueError("Expected wem_r31_settings with schema, scope and paths")
    if data["scope"] not in ("smoke", "full"):
        raise ValueError("scope must be smoke or full")
    if not isinstance(data["paths"], dict) or set(data["paths"]) != set(SETTINGS["paths"]):
        raise ValueError("paths must contain features_dir, labels_dir, split_file and output_dir")
    paths = {}
    for name, value in data["paths"].items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Fill paths.{name} in the local settings")
        if re.match(r"^[A-Za-z]:[\\/]", value) and os.name != "nt":
            raise ValueError(f"Windows path supplied on a non-Windows host: {name}")
        local = Path(value).expanduser()
        paths[name] = (local if local.is_absolute() else Path(root) / local).resolve()
    for name in ("features_dir", "labels_dir"):
        if not paths[name].is_dir():
            raise ValueError(f"Missing input directory: paths.{name}")
    if not paths["split_file"].is_file():
        raise ValueError("Missing paths.split_file")
    output = paths["output_dir"]
    if output.exists():
        raise ValueError("output_dir already exists; use a new directory for this frozen comparison")
    for name in ("features_dir", "labels_dir", "split_file"):
        source = paths[name]
        if output == source or output in source.parents or source in output.parents:
            raise ValueError("output_dir overlaps an input path")
    return {"scope": data["scope"], "paths": paths, "settings_sha256": fingerprint}


@dataclass
class Bundle:
    rows: list
    labels: dict
    partitions: dict
    sources: dict
    receipts: dict
    scope: str


def load_bundle(settings):
    paths, scope = settings["paths"], settings["scope"]
    receipts = {}

    def get_json(path, logical, expected=None):
        data, signature = read_json(path)
        if expected is not None and signature != expected:
            raise ValueError(f"Frozen fingerprint differs: {logical}")
        receipts[logical] = signature
        return data

    def get_rows(path, logical, expected):
        raw = path.read_bytes()
        signature = digest(raw)
        if signature != expected:
            raise ValueError(f"Compressed stream fingerprint differs: {logical}")
        receipts[logical] = signature
        payload = gzip.decompress(raw).decode("utf-8-sig")
        for n, line in enumerate(payload.splitlines(), 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL record: {logical}:{n}")
            yield decode(line)

    split = get_json(paths["split_file"], "training_split", PINS["split"])
    manifest = get_json(paths["features_dir"] / "manifest.json", "feature_manifest",
                        PINS["full_feature_manifest"] if scope == "full" else None)
    provenance = get_json(paths["labels_dir"] / "summary.json", "label_receipt",
                          PINS["full_label_receipt"] if scope == "full" else None)
    if split["status"] != "LOCKED":
        raise ValueError("Training split is not locked")
    training = split["training_sequence_to_source_video"]
    development = split["development_sequence_to_source_video"]
    if set(training) & set(development) or set(training.values()) & set(development.values()):
        raise ValueError("Training and official-test proxy groups overlap")
    partitions, sources, lengths, groups = {}, {}, {}, set()
    for part in PARTITIONS:
        detail = split["split"][part]
        part_groups = set()
        for seq, length in detail["sequences"].items():
            if seq in partitions or not re.fullmatch(r"[A-Za-z0-9_-]+", seq):
                raise ValueError("Invalid or repeated sequence in training split")
            integer(length, "sequence length", 1)
            group = training[seq]
            if not isinstance(group, str) or not group:
                raise ValueError("Missing proxy source group")
            partitions[seq], sources[seq], lengths[seq] = part, group, length
            part_groups.add(group)
        if part_groups != set(detail["source_videos"]) or part_groups & groups:
            raise ValueError("Source groups conflict across training partitions")
        groups.update(part_groups)
    if set(training) != set(partitions):
        raise ValueError("Training roster is not fully partitioned")
    if scope == "full":
        expected = {"fit": (51, 18999), "calibration": (11, 4366), "holdout": (12, 5566)}
        for part, size in expected.items():
            names = [seq for seq in partitions if partitions[seq] == part]
            if (len(names), sum(lengths[s] for s in names)) != size:
                raise ValueError("Full run must cover the original 74 sequences and 28,931 frames")
    if (manifest["status"] != "COMPLETE" or manifest["scope"] != scope
            or manifest["reference"] != REFERENCE or manifest["encoder"]["sha256"] != PINS["encoder"]
            or manifest["input_split_sha256"] != PINS["split"]
            or manifest["reference_protocol_sha256"] != PINS["reference_protocol"]):
        raise ValueError("Feature manifest reference/scope/encoder is inconsistent")
    if (provenance["status"] != "TRAINING_DIAGNOSTIC_LABELS_ONLY" or provenance["stage"] != scope
            or provenance["split_sha256"] != PINS["split"] or provenance["reference"] != REFERENCE
            or provenance["official_test_gt_used"] is not False
            or provenance["source"]["capture"] != manifest["parity_sha256"]):
        raise ValueError("Offline label provenance differs from the frozen feature source")
    selected = set(manifest["sequences"])
    if scope == "full" and selected != set(partitions):
        raise ValueError("Incomplete full feature coverage")
    if scope == "smoke" and (not selected <= set(partitions) or len(selected) != 3
                              or {partitions[s] for s in selected} != set(PARTITIONS)):
        raise ValueError("Smoke caches must cover one sequence in each partition")
    labels, label_snapshots = {}, {}
    label_sequence_set = set()
    for part in PARTITIONS:
        proof = provenance["partitions"][part]
        for info in proof["sequences"]:
            seq = info["sequence"]
            if seq not in selected or partitions[seq] != part or seq in label_sequence_set:
                raise ValueError("Offline label sequence coverage differs")
            label_sequence_set.add(seq)
            label_snapshots[seq] = info["provenance"]["snapshot_sha256"]
        for label in get_rows(paths["labels_dir"] / part / "candidate_edges.jsonl.gz",
                              f"labels/{part}", proof["labels_sha256"]):
            _identity(label)
            seq = label["sequence"]
            if (seq not in selected or partitions[seq] != part or label["partition"] != part
                    or label["source_video"] != sources[seq] or label["frame"] > lengths[seq]
                    or label["reference"] != REFERENCE):
                raise ValueError("Offline label sequence/frame/reference/source mismatch")
            for name in ("formal_geometry_allowed", "gmc_guard_allowed", "hard_ownership_conflict",
                         "near_duplicate_ownership_risk", "opportunity"):
                boolean(label[name], name)
            if label["identity_label"] not in LABELS:
                raise ValueError("Unknown offline label category")
            if label["opportunity"]:
                if label["identity_label"] != "same_identity":
                    raise ValueError("Only same-identity edges can be known opportunities")
                integer(label["gt_id"], "opportunity gt_id")
                integer(label["fn_episode_start_frame"], "fn_episode_start_frame", 1)
                if label["fn_episode_start_frame"] > label["frame"]:
                    raise ValueError("FN episode starts after current frame")
            qualified = (label["formal_geometry_allowed"] and label["gmc_guard_allowed"]
                         and not label["hard_ownership_conflict"] and not label["near_duplicate_ownership_risk"])
            if qualified:
                key = edge_key(label)
                if key in labels:
                    raise ValueError("Repeated offline candidate edge")
                labels[key] = label
    if label_sequence_set != selected:
        raise ValueError("Offline receipt lacks a selected sequence")
    rows, seen, per_track = [], set(), {}
    for seq in sorted(selected):
        proof = manifest["sequences"][seq]
        if (proof["partition"] != partitions[seq] or proof["snapshot"] != label_snapshots[seq]
                or scope == "full" and not proof.get("decoded_rgb_sha256")):
            raise ValueError("Feature snapshot or image proof differs from labels")
        count = 0
        for row in get_rows(paths["features_dir"] / (seq + ".jsonl.gz"), f"scores/{seq}", proof["scores_sha256"]):
            row = clean_feature(row)
            key = edge_key(row)
            if row["sequence"] != seq or key in seen or key not in labels or row["frame"] > lengths[seq]:
                raise ValueError("Unlabeled, repeated or out-of-scope score row")
            label = labels[key]
            if row["raw_box_xyxy"] != label["raw_box_xyxy"] or row["raw_score"] != label["raw_score"]:
                raise ValueError("Raw score/coordinates changed between features and labels")
            seen.add(key)
            rows.append(row)
            per_track.setdefault(key[:3], []).append(row)
            count += 1
        if count != proof["counts"].get("edges", 0):
            raise ValueError("Feature row count differs from sequence receipt")
    if seen != set(labels):
        raise ValueError("Eligible labels and cached feature rows are not in one-to-one correspondence")
    for members in per_track.values():
        order = sorted(members, key=lambda r: (-r["native_iou"], -r["raw_score"], r["det_index"]))
        boxes = list(dict.fromkeys(tuple(row["raw_box_xyxy"]) for row in order))
        grouped_labels = {}
        for row in members:
            box = tuple(row["raw_box_xyxy"])
            if row["geometry_group_rank"] != boxes.index(box) + 1:
                raise ValueError("Cached top-k rank differs from original geometry ordering")
            label = labels[edge_key(row)]
            signature = tuple(label.get(name) for name in ("identity_label", "opportunity", "gt_id", "fn_episode_start_frame"))
            if box in grouped_labels and grouped_labels[box] != signature:
                raise ValueError("Exact-coordinate duplicates have inconsistent offline labels")
            grouped_labels[box] = signature
    rows.sort(key=edge_key)
    return Bundle(rows, labels, {s: partitions[s] for s in sorted(selected)},
                  {s: sources[s] for s in sorted(selected)}, receipts, scope)
