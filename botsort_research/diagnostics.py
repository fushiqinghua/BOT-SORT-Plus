"""Offline M3 evidence collection. Ground truth is never used by the tracker.

The matches in this module describe diagnostic cohorts, not official TrackEval
metrics. In particular, a gap is a missing output, not proof of physical occlusion.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def _read_json(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def _sha(path, decompressed=False):
    opener = gzip.open if decompressed else open
    digest = hashlib.sha256()
    with opener(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def box_iou(a, b):
    """Pairwise XYXY IoU, including empty inputs."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    wh = np.maximum(0., np.minimum(a[:, None, 2:], b[None, :, 2:]) -
                    np.maximum(a[:, None, :2], b[None, :, :2]))
    intersection = wh.prod(axis=2)
    aa = np.maximum(a[:, 2:] - a[:, :2], 0.).prod(axis=1)
    ab = np.maximum(b[:, 2:] - b[:, :2], 0.).prod(axis=1)
    return intersection / np.maximum(aa[:, None] + ab[None, :] - intersection, 1e-12)


def match_boxes(gt_boxes, candidate_boxes, threshold=0.5, previous=None,
                gt_ids=None, candidate_ids=None):
    """Maximum-cardinality valid matching, then IoU / continuity tie-breaking.

    ``previous`` is only used for the optional output-gap diagnostic. This is
    deliberately not exported as CLEAR/IDSW: TrackEval remains authoritative.
    """
    overlap = box_iou(gt_boxes, candidate_boxes)
    if not overlap.size:
        return {}, overlap
    valid = overlap >= threshold - np.finfo(float).eps
    cardinality = 2. * (min(overlap.shape) + 1.)
    score = np.where(valid, cardinality + overlap, 0.)
    if previous is not None:
        for i, gid in enumerate(gt_ids):
            score[i] += valid[i] * (np.asarray(candidate_ids) == previous.get(int(gid), -1)) * .25
    rows, columns = linear_sum_assignment(-score)
    return {int(i): int(j) for i, j in zip(rows, columns) if valid[i, j]}, overlap


def size_bin(box):
    w, h = np.maximum(np.asarray(box)[2:4] - np.asarray(box)[:2], 0.)
    side = math.sqrt(float(w * h))
    return "<8" if side < 8 else "8-16" if side < 16 else "16-32" if side < 32 else "32+"


def gap_bin(gap):
    return "0" if gap == 0 else "1-3" if gap <= 3 else "4-10" if gap <= 10 else "11-30" if gap <= 30 else "31+"


def candidate_label(score):
    """Preserve frozen BoxMOT strict .1/.5 confidence comparisons."""
    if score is None:
        return "no_aligned_candidate"
    if score >= .6:
        return "birth_eligible"
    if score > .5:
        return "high_below_birth"
    if .1 < score < .5:
        return "low_association"
    return "outside_association_gate"


def candidate_from_scores(scores):
    """Find a usable candidate even when another candidate lies exactly at .5."""
    scores = np.asarray(scores, dtype=float)
    if not len(scores):
        return "no_aligned_candidate", None
    high = scores[scores > .5]
    low = scores[(scores > .1) & (scores < .5)]
    usable = high if len(high) else low if len(low) else scores
    score = float(usable.max())
    return candidate_label(score), score


def _load_mot(path, ground_truth):
    frames = defaultdict(list)
    with Path(path).open(encoding="utf-8-sig") as stream:
        for row in csv.reader(stream):
            if not row:
                continue
            if len(row) < 6:
                raise ValueError(f"Invalid MOT row in {path}: {row}")
            values = [float(x) for x in row[:7]]
            if ground_truth and len(values) > 6 and values[6] <= 0:
                continue
            frame, identity, x, y, w, h = values[:6]
            if not np.isfinite(values).all() or frame < 1 or frame != int(frame) or identity != int(identity) or w <= 0 or h <= 0:
                raise ValueError(f"Invalid MOT geometry/identity in {path}")
            frames[int(frame)].append([x, y, x + w, y + h, int(identity)])
    result = {f: np.asarray(rows, dtype=float).reshape(-1, 5) for f, rows in frames.items()}
    for frame, rows in result.items():
        if len(np.unique(rows[:, 4])) != len(rows):
            raise ValueError(f"Duplicate identity at frame {frame}: {path}")
    return result


def _cache_path(directory, sequence):
    for suffix in (".json", ".json.gz"):
        path = Path(directory) / (sequence + suffix)
        if path.is_file():
            return path
    raise FileNotFoundError(f"No detection cache for {sequence} in {directory}")


def load_feature_cache(cache_path, frames, embeddings_dir, sequence):
    """Return normalized valid vectors by frame/detection index, or a skip reason.

    Both bytes and provenance must agree. For an archived .json.gz, the exact
    decompressed-byte SHA is also accepted; canonicalizing JSON or ignoring a
    hash mismatch would hide a changed detection order and is not allowed.
    """
    if embeddings_dir is None:
        return {}, {"status": "skipped", "reason": "embeddings_not_supplied"}
    path = Path(embeddings_dir) / (sequence + ".npz")
    metadata_path = path.with_suffix(".json")
    if not path.is_file() or not metadata_path.is_file():
        return {}, {"status": "skipped", "reason": "npz_or_metadata_missing"}
    metadata = _read_json(metadata_path)
    acceptable = {_sha(cache_path)}
    if Path(cache_path).suffix == ".gz":
        acceptable.add(_sha(cache_path, decompressed=True))
    if metadata.get("detection_sha256") not in acceptable:
        return {}, {"status": "skipped", "reason": "detection_sha256_mismatch",
                    "action": "Use the original matching detection/feature caches or rebuild features."}
    if metadata.get("features_sha256") != _sha(path):
        return {}, {"status": "skipped", "reason": "features_sha256_mismatch"}
    with np.load(path, allow_pickle=False) as archive:
        required = {"features", "indices", "offsets", "frame_ids"}
        if not required.issubset(archive.files):
            raise ValueError(f"Feature cache missing {required - set(archive.files)}: {path}")
        values = {key: archive[key] for key in required}
    features = np.asarray(values["features"], dtype=float)
    offsets, indices, frame_ids = (values[x] for x in ("offsets", "indices", "frame_ids"))
    if features.ndim != 2 or features.shape[1] < 1:
        raise ValueError(f"Features must have shape (N,D), D>0: {path}")
    for value in (offsets, indices, frame_ids):
        # The original producer uses np.array([]) for an all-empty index list.
        if value.ndim != 1 or (len(value) and not np.issubdtype(value.dtype, np.integer)):
            raise ValueError(f"Feature indices, offsets and frame IDs must be integer vectors: {path}")
    if not np.array_equal(frame_ids, [f["frame"] for f in frames]):
        raise ValueError(f"Feature frame IDs do not match detections: {path}")
    if len(offsets) != len(frames) + 1 or offsets[0] != 0 or offsets[-1] != len(features) or np.any(np.diff(offsets) < 0) or len(indices) != len(features):
        raise ValueError(f"Invalid feature offsets: {path}")
    result, valid_count, zero_count, nonfinite_count = {}, 0, 0, 0
    for i, frame in enumerate(frames):
        a, b = map(int, offsets[i:i + 2])
        selected = indices[a:b]
        if np.any(selected < 0) or np.any(selected >= len(frame["detections"])) or len(np.unique(selected)) != len(selected):
            raise ValueError(f"Invalid/duplicate detection index at frame {frame['frame']}: {path}")
        vectors = {}
        for index, vector in zip(selected, features[a:b]):
            if not np.isfinite(vector).all():
                nonfinite_count += 1
                continue
            norm = np.linalg.norm(vector)
            if norm <= 1e-12:
                zero_count += 1
                continue
            vectors[int(index)] = vector / norm
            valid_count += 1
        result[int(frame["frame"])] = vectors
    return result, {"status": "available", "valid": valid_count,
                    "zero_vectors_excluded": zero_count,
                    "nonfinite_vectors_excluded": nonfinite_count,
                    "dimension": int(features.shape[1]),
                    "encoder_sha256": metadata.get("reid_sha256"),
                    "features_sha256": metadata["features_sha256"]}


class _Reservoir:
    def __init__(self, limit, seed):
        self.limit, self.seen, self.values = int(limit), 0, []
        self.rng = np.random.default_rng(seed)

    def add(self, value):
        self.seen += 1
        if len(self.values) < self.limit:
            self.values.append(value)
        else:
            index = int(self.rng.integers(0, self.seen))
            if index < self.limit:
                self.values[index] = value


def _distribution(values):
    if not values:
        return {"samples": 0}
    array = np.asarray(values, dtype=float)
    return {"samples": len(values), "mean": float(array.mean()),
            "p10": float(np.quantile(array, .1)), "p50": float(np.median(array)),
            "p90": float(np.quantile(array, .9)),
            "fraction_above_0_4": float(np.mean(array > .4))}


def diagnose_appearance(cache_dir, gt_root, output_dir, sequences=None,
                        embeddings_dir=None, predictions_dir=None,
                        iou_threshold=0.5, geometry_radius_px=64.0,
                        max_pairs_per_sequence=10000, seed=0):
    """Run M3 diagnostics on fixed detections and optional RGB feature caches.

    ``predictions_dir`` points directly to a tracker ``data`` directory.
    Appearance pairs use the previous valid observation of each identity, at
    most 30 frames old, and the same fixed center-distance gate for positives
    and negatives. Negatives are the nearest five eligible other identities.
    This probes local confusion; it is not a complete ReID benchmark.
    """
    if not 0 < iou_threshold <= 1 or geometry_radius_px <= 0 or max_pairs_per_sequence < 1 or seed < 0:
        raise ValueError("Invalid diagnostic thresholds or sample limit")
    cache_dir, gt_root, output = Path(cache_dir), Path(gt_root), Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    if sequences is None:
        sequences = sorted({p.name.removesuffix(".gz").removesuffix(".json")
                            for p in cache_dir.glob("*.json*")})
    if not sequences or len(set(sequences)) != len(sequences):
        raise ValueError("Provide nonempty, unique sequence names")
    counts = defaultdict(Counter)
    sequence_reports, all_pair_values = [], defaultdict(list)
    fields = {
        "gt_coverage.csv": ["sequence", "frame", "gt_id", "size", "candidate", "best_iou", "aligned_score", "track_id"],
        "low_score_runs.csv": ["sequence", "gt_id", "first_frame", "last_frame", "length"],
        "prediction_gaps.csv": ["sequence", "gt_id", "before", "after", "gap", "missing_annotations", "same_track_id", "size"],
        "appearance_pairs.csv": ["sequence", "kind", "frame", "previous_frame", "gt_id", "other_gt_id", "distance", "size", "gap", "center_distance"],
    }
    streams, writers = {}, {}
    try:
        for name, names in fields.items():
            streams[name] = (output / name).open("w", newline="", encoding="utf-8")
            writers[name] = csv.DictWriter(streams[name], fieldnames=names)
            writers[name].writeheader()
        for sequence_index, sequence in enumerate(sequences):
            path = _cache_path(cache_dir, sequence)
            cache = _read_json(path)
            frames = cache["frames"]
            if [f["frame"] for f in frames] != list(range(1, len(frames) + 1)):
                raise ValueError(f"Expected contiguous 1-based detection frames: {sequence}")
            gt = _load_mot(gt_root / sequence / "gt" / "gt.txt", True)
            if any(f > len(frames) for f in gt):
                raise ValueError(f"GT contains frames outside detection cache: {sequence}")
            pred = _load_mot(Path(predictions_dir) / (sequence + ".txt"), False) if predictions_dir else None
            if pred is not None and any(f > len(frames) for f in pred):
                raise ValueError(f"Predictions contain frames outside detection cache: {sequence}")
            features, feature_report = load_feature_cache(path, frames, embeddings_dir, sequence)
            history, memory, previous = defaultdict(list), {}, {}
            pair_reservoirs = {kind: _Reservoir(max_pairs_per_sequence, seed + 2 * sequence_index + index)
                               for index, kind in enumerate(("same_id", "different_id"))}
            local = Counter()
            for record in frames:
                frame = int(record["frame"])
                local["frames_with_thermal_path"] += int(bool(record.get("thermal")))
                detection = np.asarray(record["detections"], dtype=float).reshape(-1, 6)
                if not np.isfinite(detection).all() or np.any(detection[:, 2:4] <= detection[:, :2]) or np.any((detection[:, 4] < 0) | (detection[:, 4] > 1)):
                    raise ValueError(f"Invalid detection values: {sequence}, frame {frame}")
                truth = gt.get(frame, np.empty((0, 5)))
                match, overlaps = match_boxes(truth[:, :4], detection[:, :4], iou_threshold)
                frame_features = features.get(frame, {})
                for index, det in enumerate(detection):
                    band = candidate_label(float(det[4]))
                    local[f"detections_{band}"] += 1
                    counts["native_detection_crop_size"][size_bin(det[:4])] += 1
                    if index in frame_features:
                        local[f"valid_features_{band}"] += 1
                        counts["native_valid_feature_crop_size"][size_bin(det[:4])] += 1
                output_match = {}
                if pred is not None:
                    boxes = pred.get(frame, np.empty((0, 5)))
                    output_match, _ = match_boxes(truth[:, :4], boxes[:, :4], iou_threshold,
                                                  previous, truth[:, 4], boxes[:, 4])
                    output_match = {i: int(boxes[j, 4]) for i, j in output_match.items()}
                    previous = {int(truth[i, 4]): tid for i, tid in output_match.items()}
                updates = []
                for i, row in enumerate(truth):
                    gid = int(row[4])
                    aligned = overlaps[i] >= iou_threshold - np.finfo(float).eps
                    label, score = candidate_from_scores(detection[aligned, 4])
                    size = size_bin(row[:4])
                    tid = output_match.get(i)
                    local["gt_instances"] += 1
                    counts["gt_by_size"][size] += 1
                    counts["candidate_by_size"][size + "|" + label] += 1
                    if pred is not None and tid is None:
                        counts["unmatched_output_by_candidate"][label] += 1
                        counts["unmatched_output_by_size"][size] += 1
                    history[gid].append((frame, label, tid, size))
                    writers["gt_coverage.csv"].writerow(dict(sequence=sequence, frame=frame, gt_id=gid,
                        size=size, candidate=label, best_iou=float(overlaps[i].max(initial=0.)),
                        aligned_score=score, track_id=tid))
                    if i not in match or match[i] not in frame_features:
                        continue
                    index = match[i]
                    # Reject ambiguous detection-to-GT attribution for appearance analysis.
                    alternatives = np.delete(overlaps[:, index], i)
                    if alternatives.size and float(overlaps[i, index] - alternatives.max()) < .05:
                        local["ambiguous_feature_assignments_excluded"] += 1
                        continue
                    vector = frame_features[index]
                    center = (detection[index, :2] + detection[index, 2:4]) / 2
                    neighbors = []
                    for other_gid, old in memory.items():
                        age = frame - old["frame"]
                        distance = float(np.linalg.norm(center - old["center"]))
                        if 1 <= age <= 30 and distance <= geometry_radius_px:
                            neighbors.append((distance, other_gid, old))
                    negatives = 0
                    for center_distance, other_gid, old in sorted(neighbors, key=lambda x: (x[0], x[1])):
                        kind = "same_id" if other_gid == gid else "different_id"
                        if kind == "different_id":
                            if negatives >= 5:
                                continue
                            negatives += 1
                        cosine_distance = float(np.clip(1. - np.dot(vector, old["vector"]), 0., 2.))
                        pair_reservoirs[kind].add(dict(sequence=sequence, kind=kind, frame=frame,
                            previous_frame=old["frame"], gt_id=gid, other_gt_id=other_gid,
                            distance=cosine_distance, size=size, gap=frame - old["frame"] - 1,
                            center_distance=center_distance))
                    updates.append((gid, dict(frame=frame, vector=vector, center=center)))
                # No same-frame vectors enter the temporal gallery.
                memory = {gid: value for gid, value in memory.items() if frame - value["frame"] < 30}
                memory.update(updates)
            for gid, observations in history.items():
                run = []
                for observation in observations + [(None, None, None, None)]:
                    f, label, tid, size = observation
                    if run and (label != "low_association" or f != run[-1] + 1):
                        writers["low_score_runs.csv"].writerow(dict(sequence=sequence, gt_id=gid,
                            first_frame=run[0], last_frame=run[-1], length=len(run)))
                        counts["low_score_runs"][gap_bin(len(run))] += 1
                        counts["low_score_run_frames"][gap_bin(len(run))] += len(run)
                        run = []
                    if label == "low_association":
                        run.append(f)
                hits = [(i, observation) for i, observation in enumerate(observations) if observation[2] is not None]
                if pred is not None:
                    counts["trajectories"]["has_output_match" if hits else "never_output_matched"] += 1
                for (i1, first), (i2, second) in zip(hits, hits[1:]):
                    gap = second[0] - first[0] - 1
                    same = first[2] == second[2]
                    counts["matched_transitions"][gap_bin(gap)] += 1
                    if not same:
                        counts["identity_changes_in_diagnostic_matching"][gap_bin(gap)] += 1
                    if gap:
                        writers["prediction_gaps.csv"].writerow(dict(sequence=sequence, gt_id=gid,
                            before=first[0], after=second[0], gap=gap,
                            missing_annotations=gap - (i2 - i1 - 1), same_track_id=int(same), size=second[3]))
            sampled = {}
            for kind, reservoir in pair_reservoirs.items():
                sampled[kind] = {"eligible_pairs_seen": reservoir.seen, "sampled_pairs": len(reservoir.values)}
                for pair in reservoir.values:
                    writers["appearance_pairs.csv"].writerow(pair)
                    all_pair_values[kind].append(pair["distance"])
                    all_pair_values[kind + "|size=" + pair["size"]].append(pair["distance"])
                    all_pair_values[kind + "|gap=" + gap_bin(pair["gap"])].append(pair["distance"])
            sequence_reports.append(dict(sequence=sequence, frames=len(frames),
                detection_sha256=_sha(path), model_sha256=cache.get("model_sha256"),
                gt_sha256=_sha(gt_root / sequence / "gt" / "gt.txt"),
                counts=dict(local), features=feature_report, appearance_sampling=sampled,
                feature_coverage={band: {"detections": local[f"detections_{band}"],
                    "valid_features": local[f"valid_features_{band}"],
                    "missing_or_invalid": local[f"detections_{band}"] - local[f"valid_features_{band}"]}
                    for band in ("birth_eligible", "high_below_birth", "low_association", "outside_association_gate")}))
            counts["totals"].update(local)
    finally:
        for stream in streams.values():
            stream.close()
    summary = {
        "status": "COMPLETE", "purpose": "OFFLINE_DIAGNOSTIC_NOT_TRACKER_OR_OFFICIAL_METRICS",
        "gt_reference": str(gt_root), "feature_modality": "RGB supplied feature cache",
        "thermal_identity_diagnostics": {"status": "not_implemented", "reason": "No validated thermal identity encoder/cache is assumed."},
        "parameters": dict(iou_threshold=iou_threshold, geometry_radius_px=geometry_radius_px,
                           max_pairs_per_sequence_per_kind=max_pairs_per_sequence, seed=seed,
                           memory_max_age_frames=30, max_negative_neighbors=5),
        "counts": {key: dict(value) for key, value in counts.items()},
        "appearance_distributions": {key: _distribution(value) for key, value in sorted(all_pair_values.items())},
        "sequences": sequence_reports,
        "interpretation": [
            "Ground truth is used only in this offline diagnostic process.",
            "Candidate coverage is a per-GT opportunity count, not a claim that the tracker rejected that pair.",
            "Output gaps are not evidence that a target was physically occluded.",
            "Identity changes use this diagnostic matching, not TrackEval CLEAR IDSW.",
            "Appearance pairs are locally gated, bounded sampled observations; .4 is the frozen first-stage threshold reference, not a calibrated optimum.",
            "Historical official test sequences already informed selection/diagnosis; these results are retrospective development evidence.",
        ],
    }
    _write_json(output / "summary.json", summary)
    return summary
