"""Offline RGB appearance calibration and strict cached-feature replay I/O.

No function in this module extracts new features. Threshold fitting uses only
the calibration recordings; it is an empirical selected-pair diagnostic, not
probability calibration or a guarantee about online identity errors.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from .diagnostics import _Reservoir, gap_bin, load_feature_cache, match_boxes
from .io import detection_files, dump_json, load_detections, load_gt, load_json, sha256


APPEARANCE_MODEL_KIND = "rgb_high_only_gap_thresholds_v1"
GAP_BINS = ("0", "1-3", "4-10", "11-30")


def load_replay_features(cache_path, frames, embeddings_dir, sequence):
    """Require the exact archived high-score feature rows and retain FP32 bytes.

    The diagnostic loader validates hashes, frame IDs, indices and norms. Its
    normalized float64 vectors are deliberately not replayed: the archive's
    native replay used the original float32 rows, normalized by STrack itself.
    Low-score rows in the dense replay array are placeholders, never evidence.
    """
    _, report = load_feature_cache(cache_path, frames, embeddings_dir, sequence)
    if report.get("status") != "available":
        raise ValueError(f"Feature cache unavailable for {sequence}: {report}")
    if report["zero_vectors_excluded"] or report["nonfinite_vectors_excluded"]:
        raise ValueError(f"Invalid high-score feature vectors in {sequence}; rebuild the cache explicitly")
    encoder = report.get("encoder_sha256")
    if not isinstance(encoder, str) or len(encoder) != 64 or any(c not in "0123456789abcdef" for c in encoder.lower()):
        raise ValueError(f"Feature cache lacks a valid encoder SHA256: {sequence}")
    path = Path(embeddings_dir) / (sequence + ".npz")
    with np.load(path, allow_pickle=False) as archive:
        values = archive["features"]
        indices, offsets = archive["indices"], archive["offsets"]
        if values.dtype != np.float32:
            raise ValueError(f"Frozen RGB feature replay requires float32, got {values.dtype}: {sequence}")
        result = {}
        for i, frame in enumerate(frames):
            a, b = map(int, offsets[i:i + 2])
            chosen = indices[a:b].astype(int)
            dets = np.asarray(frame["detections"], np.float32).reshape(-1, 6)
            expected = np.flatnonzero(dets[:, 4] > .5)
            if not np.array_equal(chosen, expected):
                raise ValueError(f"Features must exactly follow score > .5 detection indices: {sequence}:{frame['frame']}")
            result[int(frame["frame"])] = {int(j): vector.copy() for j, vector in zip(chosen, values[a:b])}
    return result, {**report, "metadata_sha256": sha256(path.with_suffix(".json")),
                    "scope": "score_strictly_above_0.5_rgb_only", "dtype": "float32"}


def dense_frame_features(frame, feature_map, dimension):
    """Construct native N x D alignment, failing on missing high-score vectors."""
    if not isinstance(dimension, (int, np.integer)) or dimension <= 0:
        raise ValueError("Feature dimension must be positive")
    dets = np.asarray(frame["detections"], np.float32).reshape(-1, 6)
    features = feature_map.get(int(frame["frame"]), {})
    expected = set(np.flatnonzero(dets[:, 4] > .5).tolist())
    if set(features) != expected:
        raise ValueError(f"High-score features missing or misaligned at frame {frame['frame']}")
    dense = np.zeros((len(dets), int(dimension)), dtype=np.float32)
    for index, vector in features.items():
        value = np.asarray(vector, dtype=np.float32)
        if value.shape != (dimension,) or not np.isfinite(value).all() or np.linalg.norm(value) <= 1e-12:
            raise ValueError(f"Invalid feature at frame {frame['frame']}, detection {index}")
        dense[index] = value
    return dense


def _wilson_upper(successes, total, z=1.959963984540054):
    if not total:
        return None
    p = successes / total
    return float((p + z*z/(2*total) + z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))) / (1+z*z/total))


def fit_threshold(same, different, *, max_empirical_far=.005, min_same=50,
                  min_different=200, max_threshold=.8):
    """Largest observed positive distance within a sampled negative FAR budget.

    A null threshold disables appearance acceptance. Wilson intervals are
    descriptive: pairs share identities/recordings and are not independent.
    Sparse bins must back off rather than silently claim an unconstrained fit.
    """
    same, different = (np.asarray(x, dtype=float).reshape(-1) for x in (same, different))
    if not 0 <= max_empirical_far < 1 or not 0 <= max_threshold <= 2:
        raise ValueError("FAR must be in [0,1), maximum cosine distance in [0,2]")
    if min_same < 1 or min_different < 1:
        raise ValueError("Minimum pair counts must be positive")
    if any(not np.isfinite(x).all() or np.any((x < 0) | (x > 2)) for x in (same, different)):
        raise ValueError("Cosine distances must be finite and in [0,2]")
    counts = {"same_pairs": len(same), "different_pairs": len(different)}
    if len(same) < min_same or len(different) < min_different:
        return {**counts, "status": "insufficient_pairs", "threshold": None}
    negatives = np.sort(different)
    candidates = np.unique(same[same <= max_threshold])
    false_counts = np.searchsorted(negatives, candidates, side="right")
    admissible = candidates[false_counts / len(negatives) <= max_empirical_far]
    threshold = float(admissible[-1]) if len(admissible) else None
    false_accepts = int(np.sum(different <= threshold)) if threshold is not None else 0
    true_accepts = int(np.sum(same <= threshold)) if threshold is not None else 0
    return {**counts, "status": "fitted" if threshold is not None else "no_admissible_positive",
            "threshold": threshold, "same_accepted": true_accepts, "different_accepted": false_accepts,
            "empirical_tpr": true_accepts / len(same), "empirical_far": false_accepts / len(different),
            "descriptive_wilson95_far_upper": _wilson_upper(false_accepts, len(different))}


def appearance_threshold(model, gap, *, conditioned=True):
    """Return a raw cosine threshold; None means no appearance evidence allowed."""
    if gap < 0:
        raise ValueError("Appearance gap must be nonnegative")
    if gap > 30:
        return None
    if model.get("kind") != APPEARANCE_MODEL_KIND:
        raise ValueError("Unexpected appearance calibration model")
    result = model["global"]
    if conditioned:
        result = model["by_gap"][gap_bin(gap)]
    value = result.get("threshold")
    if value is not None and (not np.isfinite(value) or not 0 <= value <= 2):
        raise ValueError("Invalid calibrated appearance threshold")
    return value


def _gt_classes(path):
    result = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.reader(stream):
            if not row or not row[0].strip():
                continue
            values = [float(v) for v in row]
            if len(values) >= 7 and values[6] > 0:
                result[(int(values[0]), int(values[1]))] = int(values[7]) if len(values) > 7 else None
    return result


def calibrate_appearance(cache_dir, gt_root, embeddings_dir, split, output, *,
                         max_pairs_per_sequence=20000, seed=0, max_empirical_far=.005,
                         min_same=50, min_different=200, max_threshold=.8):
    """Fit global/gap controls on the declared calibration recordings only.

    One latest GT-aligned feature per identity, <=30 frame age, <=64 native-px
    center distance, at most five nearest same-class negatives per observation.
    Ambiguous GT attribution is excluded. Same/different streams have separate
    bounded uniform reservoirs per sequence; priors are not estimated.
    """
    from .calibration import validate_split
    files = detection_files(cache_dir)
    validate_split(split, available=files)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Use a new appearance calibration directory: {output}")
    if max_pairs_per_sequence < 1:
        raise ValueError("max_pairs_per_sequence must be positive")
    # Validate fitting options before doing any expensive I/O.
    fit_kwargs = dict(max_empirical_far=max_empirical_far, min_same=min_same,
                      min_different=min_different, max_threshold=max_threshold)
    fit_threshold([], [], **fit_kwargs)
    all_pairs, inputs, sampling = [], {}, {}
    encoder_hashes, detector_hashes, feature_dimensions = set(), set(), set()
    rejected_ambiguous, unknown_class_negatives = 0, 0
    for sequence_number, sequence in enumerate(split["calibration"]):
        cache_path = files[sequence]
        payload = load_detections(cache_path)
        frames = payload["frames"]
        features, report = load_replay_features(cache_path, frames, embeddings_dir, sequence)
        gt = load_gt(gt_root, sequence)
        gt_path = Path(gt_root) / sequence / "gt/gt.txt"
        classes = _gt_classes(gt_path)
        if gt and max(gt) > len(frames):
            raise ValueError(f"GT frame exceeds detection sequence: {sequence}")
        encoder_hashes.add(report["encoder_sha256"])
        feature_dimensions.add(report["dimension"])
        detector_hashes.add(payload.get("model_sha256"))
        inputs[sequence] = {"detections_sha256": sha256(cache_path),
                            "features_sha256": report["features_sha256"],
                            "metadata_sha256": report["metadata_sha256"],
                            "gt_sha256": sha256(gt_path)}
        reservoirs = {kind: _Reservoir(max_pairs_per_sequence, seed + sequence_number*2 + i)
                      for i, kind in enumerate(("same_id", "different_id"))}
        memory = {}
        for frame in frames:
            number = int(frame["frame"])
            # Match the replay's float32 cache-to-tracker conversion before
            # doing offline geometry in float64 (including threshold edges).
            detections = np.asarray(frame["detections"], dtype=np.float32).reshape(-1, 6).astype(float)
            truth = gt.get(number, np.empty((0, 5)))
            aligned, overlap = match_boxes(truth[:, :4], detections[:, :4])
            updates = []
            for i, index in aligned.items():
                if index not in features[number]:
                    continue
                others = np.delete(overlap[:, index], i)
                if len(others) and overlap[i, index] - others.max() < .05:
                    rejected_ambiguous += 1
                    continue
                identity = int(truth[i, 4])
                cls = classes.get((number, identity))
                vector = np.asarray(features[number][index], dtype=float)
                vector = vector / np.linalg.norm(vector)
                center = (detections[index, :2] + detections[index, 2:4]) / 2
                neighbors = []
                for old_id, old in memory.items():
                    age = number - old["frame"]
                    center_distance = float(np.linalg.norm(center - old["center"]))
                    if not 1 <= age <= 30 or center_distance > 64:
                        continue
                    if old_id != identity:
                        if cls is None or old["class"] is None:
                            unknown_class_negatives += 1
                            continue
                        if old["class"] != cls:
                            continue
                    neighbors.append((center_distance, old_id, old))
                negatives = 0
                for center_distance, old_id, old in sorted(neighbors, key=lambda x: (x[0], x[1])):
                    kind = "same_id" if old_id == identity else "different_id"
                    if kind == "different_id":
                        if negatives >= 5:
                            continue
                        negatives += 1
                    reservoirs[kind].add({"sequence": sequence, "frame": number,
                        "previous_frame": old["frame"], "gt_id": identity, "other_gt_id": old_id,
                        "class": cls, "kind": kind, "gap": number-old["frame"]-1,
                        "distance": float(np.clip(1-np.dot(vector, old["vector"]), 0, 2)),
                        "center_distance_px": center_distance})
                updates.append((identity, {"frame": number, "vector": vector, "center": center, "class": cls}))
            memory = {gid: old for gid, old in memory.items() if number-old["frame"] < 30}
            memory.update(updates)
        sampling[sequence] = {kind: {"eligible": value.seen, "retained": len(value.values)}
                              for kind, value in reservoirs.items()}
        for value in reservoirs.values():
            all_pairs.extend(value.values)
    if len(encoder_hashes) != 1 or len(detector_hashes) != 1 or None in detector_hashes or len(feature_dimensions) != 1:
        raise ValueError("Appearance calibration requires one fixed encoder, feature dimension and detector")
    def fit(pairs):
        return fit_threshold([p["distance"] for p in pairs if p["kind"] == "same_id"],
                             [p["distance"] for p in pairs if p["kind"] == "different_id"], **fit_kwargs)
    global_fit = fit(all_pairs)
    if global_fit["status"] == "insufficient_pairs":
        raise ValueError(f"Insufficient global same-class calibration pairs: {global_fit}")
    bins = {}
    for label in GAP_BINS:
        fitted = fit([p for p in all_pairs if gap_bin(p["gap"]) == label])
        if fitted["status"] == "insufficient_pairs":
            fitted.update(threshold=global_fit["threshold"], fallback="global_insufficient_bin_samples")
        else:
            fitted["fallback"] = None
        bins[label] = fitted
    model = {"schema_version": 1, "kind": APPEARANCE_MODEL_KIND,
             "interpretation": "retrospective_development_only",
             "global": global_fit, "by_gap": bins, "fit_options": fit_kwargs,
             "sampling": {"max_age_frames": 30, "radius_native_px": 64,
                          "max_same_class_negatives": 5, "per_sequence_per_kind_cap": max_pairs_per_sequence,
                          "seed": seed, "sequence_counts": sampling,
                          "ambiguous_gt_assignments_excluded": rejected_ambiguous,
                          "unknown_class_negative_pairs_excluded": unknown_class_negatives},
             "provenance": {"sequences": list(split["calibration"]), "split": split, "inputs": inputs,
                            "encoder_sha256": next(iter(encoder_hashes)),
                            "feature_dimension": next(iter(feature_dimensions)),
                            "detector_model_sha256": next(iter(detector_hashes))},
             "limitations": ["Selected GT-aligned high-score RGB pairs; no low-score or thermal features.",
                             "Offline latest-vector pairs differ from online EMA/protected-gallery features.",
                             "Sampled-pair FAR and Wilson intervals do not guarantee online recovery precision.",
                             "The calibration recordings are development data, not independent test data."]}
    output.mkdir(parents=True)
    with (output / "pairs.csv").open("w", encoding="utf-8", newline="") as stream:
        columns = ["sequence", "frame", "previous_frame", "gt_id", "other_gt_id", "class", "kind", "gap", "distance", "center_distance_px"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_pairs)
    model["provenance"]["pairs_sha256"] = sha256(output / "pairs.csv")
    dump_json(output / "model.json", model)
    dump_json(output / "summary.json", {k: model[k] for k in ("kind", "interpretation", "global", "by_gap", "fit_options", "sampling", "limitations")})
    return model


def validate_appearance_model(model, split, cache_dir, embeddings_dir, replay_sequences):
    """Validate fitted inputs and disjoint replay selection without reading GT."""
    from .calibration import validate_split
    files = detection_files(cache_dir)
    validate_split(split, available=files)
    if model.get("kind") != APPEARANCE_MODEL_KIND:
        raise ValueError("Unexpected appearance calibration model")
    provenance = model.get("provenance", {})
    fitted = provenance.get("sequences", [])
    if not fitted or len(set(fitted)) != len(fitted) or not set(fitted).issubset(split["calibration"]):
        raise ValueError("Appearance fit sequences must belong to the declared calibration partition")
    if provenance.get("split") != split:
        raise ValueError("Appearance model was fitted with a different split")
    if not replay_sequences or not set(replay_sequences).issubset(split["evaluation"]) or set(fitted) & set(replay_sequences):
        raise ValueError("Calibrated appearance replay requires disjoint evaluation sequences")
    for name in fitted:
        expected = provenance.get("inputs", {}).get(name, {})
        path = Path(embeddings_dir) / (name + ".npz")
        for field, candidate in (("detections_sha256", files[name]), ("features_sha256", path), ("metadata_sha256", path.with_suffix(".json"))):
            if not expected.get(field) or sha256(candidate) != expected[field]:
                raise ValueError(f"Appearance calibration input changed: {name}/{field}")
        metadata = load_json(path.with_suffix(".json"))
        if metadata.get("reid_sha256") != provenance.get("encoder_sha256"):
            raise ValueError(f"Appearance encoder provenance mismatch: {name}")
        if load_detections(files[name]).get("model_sha256") != provenance.get("detector_model_sha256"):
            raise ValueError(f"Appearance detector provenance mismatch: {name}")
    for name in replay_sequences:
        payload = load_detections(files[name])
        if payload.get("model_sha256") != provenance.get("detector_model_sha256"):
            raise ValueError(f"Replay detector differs from appearance calibration: {name}")
        _, report = load_replay_features(files[name], payload["frames"], embeddings_dir, name)
        if report["encoder_sha256"] != provenance.get("encoder_sha256"):
            raise ValueError(f"Replay encoder differs from appearance calibration: {name}")
        if report["dimension"] != provenance.get("feature_dimension"):
            raise ValueError(f"Replay feature dimension differs from appearance calibration: {name}")
    for gap in (0, 1, 4, 11):
        appearance_threshold(model, gap)
    appearance_threshold(model, 0, conditioned=False)
    return {"status": "PASS", "fit_sequences": fitted,
            "encoder_sha256": provenance.get("encoder_sha256"),
            "detector_model_sha256": provenance.get("detector_model_sha256")}
