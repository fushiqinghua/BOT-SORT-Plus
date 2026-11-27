"""Join frozen TCO review samples to their exact runtime observations and official GT."""

import csv
import gzip
import hashlib
import io
import json
import math
from collections import defaultdict, deque
from pathlib import Path


ROLES = {"B_early_moderate": "B", "evidence_both": "BG"}
FOCUS_SEQUENCES = ("DJI_0327_3", "DJI_0061_2")
FOCUS_STATUS = "identity_unverified_official_miss"


def _csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _records(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def _track_key(row):
    return row["method"], row["sequence"], int(row["frame"]), int(row["track_id"])


def _det_index(row):
    value = row["assigned_det_index"]
    return None if value in (None, "") else int(value)


def _gt_key(row):
    return row["sequence"], int(row["frame"]), int(row["gt_id"])


def _indexed(rows, key, wanted):
    selected = {}
    for row in rows:
        identity = key(row)
        if identity in wanted:
            if identity in selected:
                raise ValueError(f"Duplicate source row: {identity}")
            selected[identity] = row
    missing = wanted.difference(selected)
    if missing:
        raise ValueError(f"Missing {len(missing)} source rows: {sorted(missing)[:5]}")
    return selected


def _focus_draw(rows, original_gt_keys, budget):
    unique = {}
    for row in rows:
        if row["sequence"] not in FOCUS_SEQUENCES or row["case_status"] != FOCUS_STATUS:
            continue
        key = _gt_key(row)
        if key in original_gt_keys:
            continue
        # Prefer B for a GT/frame shared between arms. Native track and detection
        # indices break ties without ranking by how well the oracle aligns to GT.
        order = (ROLES[row["method"]] != "B", int(row["track_id"]), _det_index(row))
        if key not in unique or order < unique[key][0]:
            unique[key] = order, row

    result = []
    for sequence in FOCUS_SEQUENCES:
        options = [row for key, (_, row) in sorted(unique.items()) if key[0] == sequence]
        if len(options) < budget:
            raise ValueError(f"{sequence} only has {len(options)} eligible focus cases for a budget of {budget}")
        seen_gt_ids = set()
        for index in range(budget):
            start = index * len(options) // budget
            end = (index + 1) * len(options) // budget
            midpoint = (int(options[start]["frame"]) + int(options[end - 1]["frame"])) / 2
            row = min(options[start:end], key=lambda row: (
                row["gt_id"] in seen_gt_ids, abs(int(row["frame"]) - midpoint), *_gt_key(row)[1:]
            ))
            seen_gt_ids.add(row["gt_id"])
            result.append({
                **row,
                "case_id": f"TCO_FOCUS_{sequence}_{index + 1:02d}",
                "development_source_video": row["source_video"],
                "review_status": "PENDING_HUMAN_REVIEW",
                "focus_population_after_original_exclusion": len(options),
                "focus_temporal_stratum_size": end - start,
                "focus_selection": "chronological quantile bin; prefer previously unseen GT IDs, then bin midpoint",
            })
    return result


def _trace_subset(root, method, sequence, wanted):
    path = root / "full" / "on" / ROLES[method] / "tco_capture" / method / f"{sequence}.jsonl.gz"
    found = {}
    history = deque(maxlen=12)
    for frame in _records(path):
        number = frame["frame"]
        if frame["sequence"] != sequence or frame["method"] != method or (
                history and number != history[-1][0] + 1):
            raise ValueError(f"Wrong frame order or run in {path} at frame {number}")
        if number in wanted:
            past = dict(history)
            # Each past thumbnail is an actual accepted detection observation.
            # Verify identities, original image records, and boxes via the frozen end-of-frame trace.
            for track in frame["low_tracks"]:
                if track["track_id"] not in wanted[number]:
                    continue
                seen = []
                for obs in track["history"]:
                    previous = obs["frame"]
                    key = track["track_id"], obs["det_index"]
                    if not 1 <= number - previous <= 12 or past.get(previous, {}).get(key) != obs:
                        raise ValueError(f"Unverified historical observation: {path}, {number}, {key}, {previous}")
                    seen.append(previous)
                if seen != sorted(set(seen)) or len(seen) > 3:
                    raise ValueError(f"Unexpected observation order or count: {path}, {number}")
            found[number] = frame
            if len(found) == len(wanted):
                break
        history.append((number, {(obs["track_id"], obs["det_index"]): obs for obs in frame["observations"]}))
    if set(found) != set(wanted):
        raise ValueError(f"Missing trace frames in {path}: {sorted(set(wanted).difference(found))}")
    return found


def _ground_truth(path, expected_sha, wanted):
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha:
        raise ValueError(f"Official GT changed since T1 labeling: {path}")
    result = defaultdict(dict)
    for row in csv.reader(io.StringIO(content.decode("utf-8-sig"))):
        if not row:
            continue
        frame = int(float(row[0]))
        # The project runs class-agnostic TrackEval with DO_PREPROC=False.
        # Its GT mask keeps all nonzero marks, regardless of the semantic class.
        if frame not in wanted or float(row[6]) == 0:
            continue
        gt_id = int(float(row[1]))
        x, y, w, h = map(float, row[2:6])
        if gt_id in result[frame]:
            raise ValueError(f"Duplicate GT ID at {path}, frame {frame}, ID {gt_id}")
        result[frame][gt_id] = [x, y, x + w, y + h]
    return result


def _nearby_gt(boxes, episode, track, detection):
    anchor_id = episode["gt_id"]
    expected = episode["gt_xyxy"]
    actual = boxes.get(anchor_id)
    if (expected is None) != (actual is None) or (expected is not None and not all(
            math.isclose(a, b, abs_tol=1e-8) for a, b in zip(expected, actual, strict=True))):
        raise ValueError(f"Official anchor box differs from its T1 label: {_track_key(episode)}")
    refs = [track["pred_xyxy"]]
    if actual is not None:
        refs.append(actual)
    if detection is not None:
        refs.append(detection["raw_xyxy"])
    # Context annotations include official boxes close to the original evidence.
    pad = max(24, max(max(box[2] - box[0], box[3] - box[1]) for box in refs))
    x0, y0 = min(b[0] for b in refs) - pad, min(b[1] for b in refs) - pad
    x1, y1 = max(b[2] for b in refs) + pad, max(b[3] for b in refs) + pad
    return [{"gt_id": gt_id, "xyxy": box} for gt_id, box in sorted(boxes.items())
            if not (box[2] < x0 or box[0] > x1 or box[3] < y0 or box[1] > y1)]


def assemble_cases(root: Path, focus_per_sequence: int) -> list[dict]:
    """Return the 160 source rows in order, followed by separately drawn focus cases."""
    if focus_per_sequence < 0:
        raise ValueError("Focus budget must be nonnegative")
    t1 = root / "T1"
    original = _csv(t1 / "review_sample.csv")
    if [row["case_id"] for row in original] != [f"TCO_T1_{i:03d}" for i in range(1, 161)]:
        raise ValueError("The registered 160-row review sample is incomplete or out of order")
    original_keys = {_track_key(row) for row in original}
    if len(original_keys) != len(original):
        raise ValueError("The registered review sample has duplicate episode keys")
    focus_pool = _csv(t1 / "feasible_cases.csv") if focus_per_sequence else []
    focus_keys = {_track_key(row) for row in focus_pool
                  if row["sequence"] in FOCUS_SEQUENCES and row["case_status"] == FOCUS_STATUS}
    episodes = _indexed(_records(t1 / "episodes.jsonl.gz"), _track_key, original_keys | focus_keys)
    original_gt_keys = {_gt_key(episodes[key]) for key in original_keys if episodes[key]["gt_id"] is not None}
    focus = _focus_draw(focus_pool, original_gt_keys, focus_per_sequence)
    combined = [("original", row) for row in original] + [("focus", row) for row in focus]
    wanted_edges = {_track_key(row) + (_det_index(row),) for _, row in combined if _det_index(row) is not None}
    labels = _indexed(_records(t1 / "pairs.labels.jsonl.gz"), lambda row: _track_key(row) + (row["det_index"],),
                      wanted_edges)
    wanted_frames = defaultdict(lambda: defaultdict(set))
    wanted_gt_frames = defaultdict(set)
    for _, row in combined:
        method, sequence, frame, track_id = _track_key(row)
        wanted_frames[method, sequence][frame].add(track_id)
        wanted_gt_frames[sequence].add(frame)
    gt_root = Path(_json(root / "offline_settings.json")["paths"]["gt_root"])
    gt_hashes = _json(t1 / "routing_summary.json")["gt_sha256"]
    truths = {seq: _ground_truth(gt_root / seq / "gt/gt.txt", gt_hashes[seq], when)
              for seq, when in wanted_gt_frames.items()}
    frames = {key: _trace_subset(root, *key, wanted) for key, wanted in wanted_frames.items()}

    cases = []
    for scope, row in combined:
        key = method, seq, number, track_id = _track_key(row)
        episode = episodes[key]
        frame = frames[method, seq][number]
        track, = (item for item in frame["low_tracks"] if item["track_id"] == track_id)
        idx = _det_index(row)
        if idx != episode["assigned_det_index"] or idx != dict(frame["assignments"]).get(track_id):
            raise ValueError(f"Sample, episode and runtime have differing low assignments: {key}")
        for field in ("raw_xyxy", "gt_xyxy", "raw_iou", "feasible_iou", "gt_exact_reachable", "identity_status"):
            value = episode[field]
            expected = "" if value is None else str(value)
            if row[field] != expected:
                raise ValueError(f"Sample and episode disagree on {field}: {key}")
        for field in ("run_fingerprint", "frame_image", "image_shape", "image_sha256", "source_video"):
            if frame[field] != episode[field]:
                raise ValueError(f"Runtime and episode disagree on {field}: {key}")
        edge = labels.get(key + (idx,))
        detection = None
        if idx is not None:
            detection, = (item for item in frame["low_dets"] if item["det_index"] == idx)
            if detection["raw_xyxy"] != episode["raw_xyxy"] or any(edge[field] != episode[field] for field in (
                    "run_fingerprint", "gt_id", "gt_xyxy", "raw_iou", "feasible_iou", "identity_status")):
                raise ValueError(f"Assigned edge and episode have differing labels: {key}, index {idx}")
        if scope == "focus" and (not episode["geometry_only_bounded_cross_050"] or episode["anchor_covered"] or
                                 episode["gt_id"] != int(row["gt_id"]) or not edge or not edge["feasible_xyxy"]):
            raise ValueError(f"Supplemental case is not a bounded geometric crossing on an official miss: {key}")
        cases.append({"sample": row, "episode": episode, "edge_label": edge, "frame": frame, "track": track,
                      "detection": detection, "history": track["history"],
                      "nearby_gt": _nearby_gt(truths[seq].get(number, {}), episode, track, detection), "scope": scope})
    return cases
