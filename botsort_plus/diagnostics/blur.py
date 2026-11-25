"""D4: offline observation quality and RGB identity evidence diagnostics.

No quality proxy is called a motion-blur label.  Natural blur comparisons use
explicit human labels; missing features/GT/comparators stay missing.  This module
does not import BoxMOT, change tracking, or extract replacement embeddings.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

from botsort_plus.io import (
    FeatureCache, detection_files, detection_fingerprint, read_detection_cache,
    resolve_image, sequence_name, sha256, write_json,
)


SCHEMA = "botsort_blur_diagnostics_v1"
LABELS = {"unreviewed", "clear", "motion_blur", "defocus", "occlusion",
          "low_contrast", "too_small", "uncertain"}
OCCLUSION_LABELS = {"yes", "no", "uncertain"}
OBS_FIELDS = [
    "sequence", "frame", "det_index", "gt_id", "gt_match_status", "gt_class",
    "gt_visibility", "gt_visibility_bin", "image",
    "x1", "y1", "x2", "y2", "score", "short_side_px", "size_bin", "score_bin",
    "rgb_valid", "rgb_quality", "rgb_quality_bin", "roi_status", "roi_width_px",
    "roi_height_px", "in_frame_area_fraction", "intensity_mean", "contrast_std",
    "contrast_bin", "gradient_rms", "laplacian_variance", "gradient_coherence",
    "gradient_axis_deg", "ring_contrast_std", "det_center_error_px",
    "det_center_error_norm", "accepted_track_id", "accepted_stage",
    "previous_same_gt_frame", "same_gt_gap_frames", "previous_rgb_quality",
    "previous_contrast_std", "previous_short_side_px", "previous_human_label",
    "same_gt_cosine_across_gap",
    "same_gt_cosine_adjacent", "hard_negative_cosine", "hard_negative_gt_id",
    "hard_negative_det_index", "negative_count", "appearance_margin",
    "appearance_status", "human_label", "human_occlusion", "reviewer", "note",
]
METRICS = ("det_center_error_px", "det_center_error_norm", "same_gt_cosine_adjacent",
           "hard_negative_cosine", "appearance_margin", "gradient_rms",
           "laplacian_variance", "rgb_quality")
REVIEW_FIELDS = ["run_protocol_sha256", "sequence", "frame", "det_index", "label",
                 "occlusion", "reviewer", "note", "contact_sheet", "image",
                 "x1", "y1", "x2", "y2", "short_side_px", "score"]


def _number(value):
    if value in (None, ""):
        return None
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Diagnostic numbers must be finite or empty")
    return value


def _integer(value, minimum=0):
    result = _number(value)
    if result is None or int(result) != result or result < minimum:
        raise ValueError(f"Expected integer >= {minimum}: {value!r}")
    return int(result)


def _boolean(value):
    if value in (True, "True", "true", "1", 1):
        return True
    if value in (False, "False", "false", "0", 0):
        return False
    raise ValueError(f"Invalid boolean: {value!r}")


def size_bin(short_side):
    return "lt8" if short_side < 8 else "8to16" if short_side < 16 else "16to32" if short_side < 32 else "ge32"


def score_bin(score):
    return "le0.1" if score <= .1 else "0.1to0.5" if score < .5 else "eq0.5" if score == .5 else "gt0.5"


def contrast_bin(contrast):
    return "missing" if contrast is None else "lt0.03" if contrast < .03 else "0.03to0.10" if contrast < .1 else "ge0.10"


def quality_bin(quality):
    return "missing" if quality is None else "lt0.15" if quality < .15 else "0.15to0.35" if quality < .35 else "0.35to0.70" if quality < .7 else "ge0.70"


def visibility_bin(visibility):
    if visibility is None or not 0 <= visibility <= 1:
        return "unknown"
    return "lt0.5" if visibility < .5 else "0.5to0.9" if visibility < .9 else "ge0.9"


def native_roi_metrics(image, xyxy):
    """Native-pixel quality proxies, without resizing or declaring blur.

    Intensity is normalized to [0,1].  The gradient axis describes image edges,
    NOT an estimated blur direction.  Very small ROIs have missing derivatives.
    """
    import cv2
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim not in (2, 3):
        raise ValueError("Quality diagnostics require uint8 original pixels")
    if image.ndim == 3 and image.shape[2] != 3:
        raise ValueError("Expected BGR image with three channels")
    box = np.asarray(xyxy, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
        raise ValueError("Expected finite positive xyxy box")
    height, width = image.shape[:2]
    x1, y1 = np.floor(box[:2]).astype(int)
    x2, y2 = np.ceil(box[2:]).astype(int)
    x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
    y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
    clipped = np.maximum(0., np.minimum(box[2:], [width, height]) - np.maximum(box[:2], 0.))
    result = dict(roi_status="outside_image", roi_width_px=x2-x1, roi_height_px=y2-y1,
                  in_frame_area_fraction=float(np.prod(clipped) / np.prod(box[2:]-box[:2])),
                  intensity_mean=None, contrast_std=None, gradient_rms=None,
                  laplacian_variance=None, gradient_coherence=None,
                  gradient_axis_deg=None, ring_contrast_std=None)
    if x2 <= x1 or y2 <= y1:
        return result
    patch = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else patch
    gray = gray.astype(np.float64) / 255.
    result.update(roi_status="ok", intensity_mean=float(gray.mean()), contrast_std=float(gray.std()))
    gradients = []
    if gray.shape[1] > 1:
        gradients.append(np.diff(gray, axis=1).ravel())
    if gray.shape[0] > 1:
        gradients.append(np.diff(gray, axis=0).ravel())
    if gradients:
        result["gradient_rms"] = float(np.sqrt(np.mean(np.concatenate(gradients)**2)))
    if min(gray.shape) >= 3:
        lap = 4*gray[1:-1, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:] - gray[:-2, 1:-1] - gray[2:, 1:-1]
        result["laplacian_variance"] = float(np.var(lap))
        gx, gy = (gray[1:-1, 2:]-gray[1:-1, :-2])*.5, (gray[2:, 1:-1]-gray[:-2, 1:-1])*.5
        xx, yy, xy = float(np.mean(gx*gx)), float(np.mean(gy*gy)), float(np.mean(gx*gy))
        if xx+yy > 1e-12:
            result["gradient_coherence"] = float(np.clip(np.hypot(xx-yy, 2*xy)/(xx+yy), 0., 1.))
            result["gradient_axis_deg"] = float((.5*np.degrees(np.arctan2(2*xy, xx-yy))) % 180)
    pad = max(2, int(math.ceil(max(x2-x1, y2-y1)*.5)))
    ox1, oy1, ox2, oy2 = max(0, x1-pad), max(0, y1-pad), min(width, x2+pad), min(height, y2+pad)
    outer = image[oy1:oy2, ox1:ox2]
    outer = cv2.cvtColor(outer, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else outer
    mask = np.ones(outer.shape, dtype=bool)
    mask[y1-oy1:y2-oy1, x1-ox1:x2-ox1] = False
    if mask.any():
        result["ring_contrast_std"] = float((outer[mask].astype(float)/255).std())
    return result


def appearance_evidence(frame, current, previous):
    """Compare a previous same-GT anchor against this frame's positive/negatives.

    ``current`` maps unique, unambiguously assigned GT IDs to det_index/vector.
    ``previous`` stores the last *detection*, including invalid features.  Missing
    frames or invalid intervening features cannot masquerade as adjacent pairs.
    """
    result = {}
    for gt_id, observation in current.items():
        prior = previous.get(gt_id)
        vector = observation.get("vector")
        entry = dict(previous_same_gt_frame=prior["frame"] if prior else None,
                     same_gt_gap_frames=frame-prior["frame"] if prior else None,
                     same_gt_cosine_across_gap=None, same_gt_cosine_adjacent=None,
                     hard_negative_cosine=None, hard_negative_gt_id=None,
                     hard_negative_det_index=None, negative_count=0,
                     appearance_margin=None, appearance_status="no_previous_gt_detection")
        for name in ("rgb_quality", "contrast_std", "short_side_px", "human_label"):
            entry["previous_"+name] = prior.get(name) if prior else None
        if vector is None:
            entry["appearance_status"] = "missing_current_feature"
        elif prior is not None and prior.get("vector") is None:
            entry["appearance_status"] = "missing_previous_feature"
        elif prior is not None:
            distance = float(np.clip(1-np.dot(prior["vector"], vector), 0., 2.))
            entry["same_gt_cosine_across_gap"] = distance
            if frame-prior["frame"] != 1:
                entry["appearance_status"] = "nonconsecutive_gt_detection"
            else:
                entry["same_gt_cosine_adjacent"] = distance
                entry["appearance_status"] = "adjacent_no_negative"
                negatives = [(float(np.clip(1-np.dot(prior["vector"], other["vector"]), 0., 2.)),
                              int(other_gt), int(other["det_index"]))
                             for other_gt, other in current.items()
                             if other_gt != gt_id and other.get("vector") is not None]
                entry["negative_count"] = len(negatives)
                if negatives:
                    negative, other_gt, other_index = min(negatives)
                    entry.update(hard_negative_cosine=negative, hard_negative_gt_id=other_gt,
                                 hard_negative_det_index=other_index,
                                 appearance_margin=negative-distance, appearance_status="adjacent_comparable")
        result[gt_id] = entry
    for gt_id, observation in current.items():
        vector = observation.get("vector")
        previous[gt_id] = {"frame": int(frame), "vector": None if vector is None else vector.copy(),
                           **{name: observation.get(name) for name in
                              ("rgb_quality", "contrast_std", "short_side_px", "human_label")}}
    return result


def _stats(values):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(values):
        return {"n": 0, "mean": None, "median": None, "p90": None}
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite diagnostic statistic")
    return {"n": int(len(values)), "mean": float(values.mean()),
            "median": float(np.median(values)), "p90": float(np.quantile(values, .9))}


def _bucket():
    return {"n": 0, "matched": 0, "rgb_valid": 0, "metrics": {name: [] for name in METRICS}}


def _add(bucket, row):
    bucket["n"] += 1
    bucket["matched"] += int(row["gt_match_status"] == "matched")
    bucket["rgb_valid"] += int(row["rgb_valid"])
    for name in METRICS:
        value = row.get(name)
        if value is not None:
            bucket["metrics"][name].append(value)


def _summarize(bucket):
    return {"observations": bucket["n"], "gt_matched": bucket["matched"],
            "valid_rgb": bucket["rgb_valid"],
            "metrics": {name: _stats(values) for name, values in bucket["metrics"].items()}}


def load_manual_labels(path, protocol_hash):
    labels = {}
    if path is None:
        return labels
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"sequence", "frame", "det_index", "label", "occlusion", "run_protocol_sha256"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"Manual labels need columns {sorted(required)}")
        for row in reader:
            key = (row["sequence"], _integer(row["frame"], 1), _integer(row["det_index"]))
            if key in labels:
                raise ValueError(f"Duplicate manual label key: {key}")
            if row["run_protocol_sha256"] != protocol_hash:
                raise ValueError("Manual labels belong to a different replay protocol")
            label, occlusion = row["label"].strip(), row["occlusion"].strip()
            if label not in LABELS or occlusion not in OCCLUSION_LABELS:
                raise ValueError(f"Unknown manual label/occlusion for {key}")
            labels[key] = {"human_label": label, "human_occlusion": occlusion,
                           "reviewer": row.get("reviewer", ""), "note": row.get("note", "")}
    return labels


def _audit_groups(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"sequence", "frame", "det_index", "gt_id", "gt_match_status", "score",
                    "x1", "y1", "x2", "y2", "rgb_valid", "rgb_quality",
                    "det_center_error_px", "det_center_error_norm"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("Incomplete D1 detection_audit.csv schema")
        previous = None
        for sequence, rows in itertools.groupby(reader, key=lambda row: row["sequence"]):
            if previous is not None and sequence <= previous:
                raise ValueError("Detection audit must be sorted by sequence, frame, det_index")
            previous = sequence
            yield sequence, list(rows)


def _trace_frames(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _write_table(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _contact_sheets(rows, output_dir, images_root):
    """Native ROI plus nearest-neighbor context; no image enhancement."""
    import cv2
    if not rows:
        return
    folder = output_dir/"contact_sheets"
    folder.mkdir()
    width, height, page_size = 600, 194, 12
    for page in range(math.ceil(len(rows)/page_size)):
        selected = rows[page*page_size:(page+1)*page_size]
        sheet = np.full((height*len(selected), width, 3), 245, dtype=np.uint8)
        name = f"sheet_{page+1:03d}.png"
        for offset, row in enumerate(selected):
            image = cv2.imread(str(resolve_image(row["image"], images_root)))
            if image is None:
                raise ValueError("Image became unreadable during contact-sheet export")
            h, w = image.shape[:2]
            box = np.array([row[k] for k in ("x1", "y1", "x2", "y2")])
            x1, y1 = np.maximum(np.floor(box[:2]).astype(int), 0)
            x2, y2 = np.minimum(np.ceil(box[2:]).astype(int), [w, h])
            yy = offset*height
            cv2.putText(sheet, f"{row['sequence']}  f={row['frame']}  det={row['det_index']}",
                        (8, yy+20), cv2.FONT_HERSHEY_SIMPLEX, .46, (20,20,20), 1, cv2.LINE_AA)
            cv2.putText(sheet, "native ROI* / nearest ROI / nearest context (no sharpening)",
                        (8, yy+40), cv2.FONT_HERSHEY_SIMPLEX, .40, (20,20,20), 1, cv2.LINE_AA)
            cv2.putText(sheet, "*native: top-left if ROI exceeds 150x130 px", (8, yy+189),
                        cv2.FONT_HERSHEY_SIMPLEX, .37, (20,20,20), 1, cv2.LINE_AA)
            if x2 > x1 and y2 > y1:
                crop = image[y1:y2, x1:x2]
                native = crop[:130, :150]
                sheet[yy+52:yy+52+native.shape[0], 8:8+native.shape[1]] = native
                scale = min(145/crop.shape[1], 130/crop.shape[0], 16.)
                enlarged = cv2.resize(crop, (max(1, round(crop.shape[1]*scale)), max(1, round(crop.shape[0]*scale))), interpolation=cv2.INTER_NEAREST)
                sheet[yy+52:yy+52+enlarged.shape[0], 178:178+enlarged.shape[1]] = enlarged
                pad = max(8, int(max(x2-x1, y2-y1)))
                context = image[max(0,y1-pad):min(h,y2+pad), max(0,x1-pad):min(w,x2+pad)]
                scale = min(245/context.shape[1], 130/context.shape[0])
                context = cv2.resize(context, (max(1, round(context.shape[1]*scale)), max(1, round(context.shape[0]*scale))), interpolation=cv2.INTER_NEAREST)
                sheet[yy+52:yy+52+context.shape[0], 346:346+context.shape[1]] = context
            row["contact_sheet"] = str(Path("contact_sheets")/name).replace("\\", "/")
        if not cv2.imwrite(str(folder/name), sheet):
            raise IOError(f"Could not write {folder/name}")


def analyze_blur(run_root, tracker_name, baseline_root, features_dir, images_root,
                 audit_dir, output_dir, manual_labels=None, max_review=120):
    """Analyze all original detector rows, using D1's offline GT identity audit.

    The replay and D1 audit must be COMPLETE and provenance-linked. Output must
    be a new directory. A second invocation with manual labels also uses a new
    output directory, keeping the unlabeled diagnosis intact.
    """
    import cv2
    if isinstance(max_review, bool) or int(max_review) != max_review or max_review < 0:
        raise ValueError("max_review must be a nonnegative integer")
    run_root, baseline_root, features_dir, images_root = map(Path, (run_root, baseline_root, features_dir, images_root))
    audit_dir, output_dir = Path(audit_dir), Path(output_dir)
    if Path(tracker_name).name != tracker_name or tracker_name in ("", ".", ".."):
        raise ValueError("Invalid tracker name")
    protocol_path = run_root/"protocols"/(tracker_name+".json")
    protocol_hash = sha256(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8-sig"))
    if protocol.get("status") != "COMPLETE":
        raise ValueError("D4 needs a COMPLETE replay protocol")
    if protocol.get("diagnostics", {}).get("status") != "COMPLETE":
        raise ValueError("D4 needs COMPLETE diagnostic traces")
    audit_summary = json.loads((audit_dir/"summary.json").read_text(encoding="utf-8-sig"))
    if audit_summary.get("status") != "COMPLETE" or audit_summary.get("run_protocol_sha256") != protocol_hash:
        raise ValueError("D1 audit must be COMPLETE and belong to the same replay protocol")
    if audit_summary.get("files_sha256", {}).get("detection_audit.csv") != sha256(audit_dir/"detection_audit.csv"):
        raise ValueError("D1 detection audit differs from its COMPLETE summary")
    source_protocol = baseline_root/"protocol.json"
    if sha256(source_protocol) != protocol["source_protocol_sha256"]:
        raise ValueError("Baseline source protocol differs from replay")
    paths = {sequence_name(p): p for p in detection_files(baseline_root/"detections")}
    selected = sorted(protocol["sequences"])
    if not set(selected) <= paths.keys():
        raise ValueError("Missing baseline detection sequences")
    labels = load_manual_labels(manual_labels, protocol_hash)
    unknown_labels = set(labels)
    output_dir.mkdir(parents=True, exist_ok=False)
    source_root = Path(__file__).resolve().parents[2]
    offline_sources = ("botsort_plus/diagnostics/blur.py", "botsort_plus/io.py",
                       "botsort_plus/types.py")
    manifest = {"schema": SCHEMA, "status": "RUNNING", "run_protocol_sha256": protocol_hash,
                "offline_implementation_sha256": {name: sha256(source_root/name) for name in offline_sources},
                "offline_implementation_semantics": "Current D4 source files at analysis time; distinct from the capture implementation recorded in the replay protocol.",
                "audit_csv_sha256": sha256(audit_dir/"detection_audit.csv"),
                "audit_summary_sha256": sha256(audit_dir/"summary.json"),
                "manual_labels_sha256": sha256(manual_labels) if manual_labels else None,
                "sequences": {}, "quality_is_blur_label": False,
                "scope": "Offline GT-assisted diagnostic; no tracker changes, no MOT metric redefinition.",
                "image_hash_semantics": "SHA256 of decoded contiguous uint8 BGR pixels, matched to online trace",
                "limitations": ["Only unambiguous IoU-assigned GT detections enter identity pairs.",
                    "Quality derivatives confound scale, contrast, shape and blur; they are not blur detectors.",
                    "Adjacent identity distances require actual adjacent frames and two valid embeddings.",
                    "Hard negative is another unambiguous GT detection in the current frame using the same previous-frame anchor.",
                    "Manual review is stratified descriptive sampling, not a prevalence estimate or causal experiment.",
                    "Previous observation quality/labels are retained; unknown prior quality and unknown occlusion remain confounds.",
                    "No comparison can identify the cause of complete detection disappearance from detector rows alone."]}
    write_json(output_dir/"summary.json", manifest)
    groups = iter(_audit_groups(audit_dir/"detection_audit.csv"))
    overall, strata, label_buckets = _bucket(), defaultdict(_bucket), defaultdict(_bucket)
    reviewed_strata = defaultdict(lambda: defaultdict(_bucket))
    status_counts, quality_status_counts, human_counts = Counter(), Counter(), Counter()
    review_candidates = {}
    try:
        pending = next(groups, None)
        with (output_dir/"observation_quality.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, OBS_FIELDS, extrasaction="ignore", lineterminator="\n")
            writer.writeheader()
            for seq in selected:
                rows = pending[1] if pending is not None and pending[0] == seq else []
                if pending is not None and pending[0] < seq:
                    raise ValueError("Audit has a sequence outside the replay")
                if rows:
                    pending = next(groups, None)
                cache = read_detection_cache(paths[seq])
                if detection_fingerprint(cache) != protocol["detection_fingerprints"][seq]:
                    raise ValueError(f"Detector fingerprint mismatch: {seq}")
                feature_path = features_dir/(seq+".npz")
                feature_hashes = {"features": sha256(feature_path), "metadata": sha256(feature_path.with_suffix(".json"))}
                if feature_hashes != protocol["feature_hashes"][seq]:
                    raise ValueError(f"ReID cache differs from replay: {seq}")
                features = FeatureCache(feature_path, cache, expected_rgb_sha=protocol["rgb_encoder_sha256"])
                if len(cache["frames"]) != protocol["sequences"][seq]:
                    raise ValueError("Frame count changed after replay")
                expected_count = sum(len(f["detections"]) for f in cache["frames"])
                if len(rows) != expected_count:
                    raise ValueError(f"D1 must retain all detector rows: {seq}")
                trace_path = run_root/"diagnostics"/tracker_name/(seq+".jsonl.gz")
                trace_manifest_path = trace_path.with_name(seq+".json")
                trace_manifest = json.loads(trace_manifest_path.read_text(encoding="utf-8-sig"))
                trace_contract = protocol["diagnostics"]["traces"][seq]
                trace_hash = sha256(trace_path)
                if (trace_manifest.get("schema") != "botsort_motion_diagnostics_v1"
                        or trace_manifest.get("status") != "COMPLETE" or trace_manifest.get("sequence") != seq
                        or trace_manifest.get("records") != protocol["sequences"][seq]
                        or sha256(trace_manifest_path) != trace_contract["manifest_sha256"]
                        or trace_hash != trace_manifest.get("data_sha256")
                        or trace_hash != trace_contract["data_sha256"]):
                    raise ValueError(f"Diagnostic trace integrity failure: {seq}")
                traces = iter(_trace_frames(trace_path))
                cursor, previous, sequence_hash = 0, {}, hashlib.sha256()
                for index, cached_frame in enumerate(cache["frames"]):
                    frame = int(cached_frame["frame"])
                    trace = next(traces, None)
                    if (trace is None or trace.get("schema") != "botsort_motion_diagnostics_v1"
                            or trace.get("sequence") != seq or trace.get("frame") != frame):
                        raise ValueError(f"Incomplete or misaligned diagnostic trace: {seq}:{frame}")
                    image = cv2.imread(str(resolve_image(cached_frame["image"], images_root)))
                    if image is None:
                        raise ValueError(f"Unreadable original image: {cached_frame['image']}")
                    image_digest = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
                    if (trace.get("image_sha256") != image_digest or trace.get("image_shape") != list(image.shape)
                            or trace.get("image_dtype") != str(image.dtype)):
                        raise ValueError(f"Original decoded image differs from replay: {seq}:{frame}")
                    sequence_hash.update(f"{frame}:{image_digest}\n".encode("ascii"))
                    detections = np.asarray(cached_frame["detections"], dtype=np.float32).reshape(-1, 6)
                    first = int(features.arrays["offsets"][index])
                    current, frame_rows = {}, []
                    for det_index, det in enumerate(detections):
                        audit = rows[cursor]
                        cursor += 1
                        key = (seq, frame, det_index)
                        if (audit["sequence"], _integer(audit["frame"], 1), _integer(audit["det_index"])) != key:
                            raise ValueError(f"D1 detector row alignment mismatch: {key}")
                        if not np.allclose([_number(audit[k]) for k in ("x1", "y1", "x2", "y2", "score")], det[:5], rtol=1e-6, atol=1e-6):
                            raise ValueError(f"D1 detector values disagree with frozen cache: {key}")
                        valid = bool(features.arrays["rgb_valid"][first+det_index])
                        cached_quality = float(features.arrays["rgb_quality"][first+det_index])
                        quality = cached_quality if valid else None
                        audit_quality = _number(audit["rgb_quality"])
                        if (_boolean(audit["rgb_valid"]) != valid
                                or (valid and (audit_quality is None or not np.isclose(audit_quality, cached_quality, rtol=1e-6, atol=1e-6)))
                                or (not valid and audit_quality is not None and not np.isclose(audit_quality, cached_quality, rtol=1e-6, atol=1e-6))):
                            raise ValueError(f"D1 RGB metadata disagrees with cache: {key}")
                        match_status = audit["gt_match_status"]
                        if match_status not in ("matched", "ambiguous", "unmatched", "no_gt"):
                            raise ValueError(f"Unknown GT-match status: {key}")
                        gt_id = _integer(audit["gt_id"], 0) if match_status == "matched" else None
                        gt_class = _number(audit.get("gt_class")) if gt_id is not None else None
                        gt_visibility = _number(audit.get("gt_visibility")) if gt_id is not None else None
                        short_side = float(min(det[2]-det[0], det[3]-det[1]))
                        metrics = native_roi_metrics(image, det[:4])
                        row = {"sequence": seq, "frame": frame, "det_index": det_index,
                               "gt_id": gt_id, "gt_match_status": match_status,
                               "gt_class": gt_class, "gt_visibility": gt_visibility,
                               "gt_visibility_bin": visibility_bin(gt_visibility), "image": cached_frame["image"],
                               **dict(zip(("x1", "y1", "x2", "y2", "score"), map(float, det[:5]))),
                               "short_side_px": short_side, "size_bin": size_bin(short_side),
                               "score_bin": score_bin(float(det[4])), "rgb_valid": valid,
                               "rgb_quality": quality, "rgb_quality_bin": quality_bin(quality),
                               **metrics, "contrast_bin": contrast_bin(metrics["contrast_std"]),
                               "det_center_error_px": _number(audit["det_center_error_px"]) if gt_id is not None else None,
                               "det_center_error_norm": _number(audit["det_center_error_norm"]) if gt_id is not None else None,
                               "accepted_track_id": audit.get("accepted_track_id", ""),
                               "accepted_stage": audit.get("accepted_stage", ""),
                               "appearance_status": "unknown_gt_identity", "negative_count": 0,
                               "human_label": "unreviewed", "human_occlusion": "uncertain", "reviewer": "", "note": ""}
                        if key in labels:
                            row.update(labels[key])
                            unknown_labels.discard(key)
                        if gt_id is not None:
                            if gt_id in current:
                                raise ValueError(f"D1 assigned the same GT twice: {seq}:{frame}:{gt_id}")
                            current[gt_id] = {"det_index": det_index,
                                "vector": features.arrays["rgb"][first+det_index].astype(np.float64) if valid else None,
                                **{name: row[name] for name in ("rgb_quality", "contrast_std", "short_side_px", "human_label")}}
                        frame_rows.append(row)
                    evidence = appearance_evidence(frame, current, previous)
                    for row in frame_rows:
                        if row["gt_id"] is not None:
                            row.update(evidence[row["gt_id"]])
                        writer.writerow(row)
                        _add(overall, row)
                        group = (row["size_bin"], row["score_bin"], row["contrast_bin"], row["rgb_quality_bin"])
                        _add(strata[group], row)
                        _add(label_buckets[row["human_label"]], row)
                        if row["human_label"] in ("clear", "motion_blur"):
                            key = (seq, row["size_bin"], row["score_bin"], row["contrast_bin"],
                                   str(row["gt_class"]) if row["gt_class"] is not None else "unknown",
                                   row["gt_visibility_bin"], row["human_occlusion"])
                            _add(reviewed_strata[key][row["human_label"]], row)
                        status_counts[row["appearance_status"]] += 1
                        quality_status_counts[row["roi_status"]] += 1
                        human_counts[row["human_label"]] += 1
                        if max_review:
                            stratum = (seq, row["size_bin"], row["score_bin"], row["contrast_bin"],
                                       str(row["gt_class"]) if row["gt_class"] is not None else "unknown",
                                       row["gt_visibility_bin"])
                            priority = hashlib.sha256(f"{seq}:{frame}:{row['det_index']}".encode("utf-8")).hexdigest()
                            candidates = review_candidates.setdefault(stratum, [])
                            candidates.append((priority, {k: row[k] for k in
                                ("sequence", "frame", "det_index", "image", "x1", "y1", "x2", "y2", "short_side_px", "score")}))
                            candidates.sort(key=lambda item: item[0])
                            del candidates[4:]
                if next(traces, None) is not None:
                    raise ValueError(f"Extra frames in diagnostic trace: {seq}")
                manifest["sequences"][seq] = {"frames": len(cache["frames"]), "observations": cursor,
                    "trace_sha256": trace_hash, "decoded_images_fingerprint": sequence_hash.hexdigest()}
                del features, cache, rows, previous
            if pending is not None or next(groups, None) is not None:
                raise ValueError("D1 audit contains extra sequences")
        if unknown_labels:
            raise ValueError(f"Manual labels reference unknown detector rows: {sorted(unknown_labels)[:3]}")
        summary_rows = []
        for group, bucket in sorted(strata.items()):
            for metric, stats in _summarize(bucket)["metrics"].items():
                summary_rows.append(dict(zip(("size_bin", "score_bin", "contrast_bin", "rgb_quality_bin"), group)) |
                                    {"observations": bucket["n"], "gt_matched": bucket["matched"],
                                     "rgb_valid": bucket["rgb_valid"], "metric": metric, **stats})
        _write_table(output_dir/"quality_strata.csv", summary_rows,
                     ["size_bin", "score_bin", "contrast_bin", "rgb_quality_bin", "observations", "gt_matched", "rgb_valid", "metric", "n", "mean", "median", "p90"])
        comparisons = []
        for group, buckets in sorted(reviewed_strata.items()):
            if "clear" not in buckets or "motion_blur" not in buckets:
                continue
            for metric in METRICS:
                clear, blurred = _stats(buckets["clear"]["metrics"][metric]), _stats(buckets["motion_blur"]["metrics"][metric])
                comparisons.append(dict(zip(("sequence", "size_bin", "score_bin", "contrast_bin", "gt_class", "gt_visibility_bin", "occlusion"), group)) |
                    {"metric": metric, "clear_n": clear["n"], "blur_n": blurred["n"],
                     "clear_mean": clear["mean"], "blur_mean": blurred["mean"],
                     "difference_blur_minus_clear": blurred["mean"]-clear["mean"] if clear["n"] and blurred["n"] else None})
        _write_table(output_dir/"manual_label_comparisons.csv", comparisons,
                     ["sequence", "size_bin", "score_bin", "contrast_bin", "gt_class", "gt_visibility_bin", "occlusion", "metric", "clear_n", "blur_n", "clear_mean", "blur_mean", "difference_blur_minus_clear"])
        review = []
        ordered_strata = sorted(review_candidates, key=lambda group:
            (len(review_candidates[group]) < 2, hashlib.sha256(repr(group).encode("utf-8")).hexdigest()))
        sampled = [item for group in ordered_strata for item in review_candidates[group]][:max_review]
        for _, row in sampled:
            key = (row["sequence"], row["frame"], row["det_index"])
            label = labels.get(key, {})
            review.append({"run_protocol_sha256": protocol_hash, **row,
                           "label": label.get("human_label", "unreviewed"),
                           "occlusion": label.get("human_occlusion", "uncertain"),
                           "reviewer": label.get("reviewer", ""), "note": label.get("note", ""), "contact_sheet": ""})
        _contact_sheets(review, output_dir, images_root)
        _write_table(output_dir/"manual_review.csv", review, REVIEW_FIELDS)
        labels_summary = {"status": "LABELLED_DESCRIPTIVE_ANALYSIS" if comparisons else "NEEDS_MORE_COMPARABLE_MANUAL_LABELS",
                          "counts": dict(human_counts), "within_sequence_matched_strata": len({tuple(row[k] for k in ("sequence", "size_bin", "score_bin", "contrast_bin", "gt_class", "gt_visibility_bin", "occlusion")) for row in comparisons}),
                          "comparison_rows": len(comparisons), "labels": sorted(LABELS),
                          "occlusion_values": sorted(OCCLUSION_LABELS),
                          "scope": "Descriptive differences within sequence/size/score/contrast/GT-class/GT-visibility/occlusion strata; unknown occlusion remains a confound; no causal effect or MOT gain is asserted.",
                          "by_label": {name: _summarize(bucket) for name, bucket in sorted(label_buckets.items())}}
        write_json(output_dir/"manual_label_summary.json", labels_summary)
        manifest.update(status="COMPLETE", overall=_summarize(overall),
                        appearance_status_counts=dict(status_counts), roi_status_counts=dict(quality_status_counts),
                        manual_review_rows=len(review), manual_label_status=labels_summary["status"],
                        review_sampling={"method": "Up to four deterministic hash-sampled rows per sequence/size/score/contrast/GT-class/GT-visibility stratum; paired strata first; no outcome-based selection.",
                                         "candidate_strata": len(review_candidates), "max_review": max_review},
                        derived_files_sha256={p.name: sha256(p) for p in output_dir.iterdir() if p.is_file() and p.name != "summary.json"})
    except Exception as error:
        manifest.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output_dir/"summary.json", manifest)
    return manifest
