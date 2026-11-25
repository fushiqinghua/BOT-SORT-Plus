"""Validated cache/annotation I/O; annotations are used only offline."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def load_json(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def detection_files(cache_dir):
    """Map sequence name to JSON or JSON.gz, rejecting ambiguous duplicates."""
    cache_dir = Path(cache_dir)
    result = {}
    for path in sorted(cache_dir.glob("*.json*")):
        if path.name.endswith(".json.gz"):
            name = path.name[:-8]
        elif path.suffix == ".json":
            name = path.stem
        else:
            continue
        if name in result:
            raise ValueError(f"Both compressed and plain caches exist for {name}; choose one directory")
        result[name] = path
    if not result:
        raise FileNotFoundError(f"No sequence detection caches in {cache_dir}")
    return result


def load_detections(path):
    data = load_json(path)
    frames = data.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"Empty or missing frames in {path}")
    if [f.get("frame") for f in frames] != list(range(1, len(frames) + 1)):
        raise ValueError(f"Frames must be contiguous, 1-based, including empty detections: {path}")
    for frame in frames:
        det = np.asarray(frame.get("detections"), dtype=float)
        if det.size == 0:
            det = np.empty((0, 6))
        if det.ndim != 2 or det.shape[1] != 6 or not np.isfinite(det).all():
            raise ValueError(f"Expected finite N x 6 detections at frame {frame['frame']}: {path}")
        if np.any(det[:, 2:4] <= det[:, :2]):
            raise ValueError(f"Non-positive box at frame {frame['frame']}: {path}")
        if np.any((det[:, 4] < 0) | (det[:, 4] > 1)):
            raise ValueError(f"Score outside [0,1]: {path}")
    return data


def load_gt(gt_root, sequence):
    """Read official MOT rows as frame -> N x 5 [x1,y1,x2,y2,id].

    Pool all classes, retaining rows with positive mark (column 7).
    An existing empty gt.txt is valid. No annotation modifications are written.
    """
    path = Path(gt_root) / sequence / "gt" / "gt.txt"
    frames = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.reader(stream):
            if not row or not row[0].strip():
                continue
            values = [float(v) for v in row]
            if len(values) < 7 or not np.isfinite(values).all():
                raise ValueError(f"Invalid MOT annotation in {path}")
            f, identity, x, y, w, h, mark = values[:7]
            if mark <= 0:
                continue
            if f < 1 or f != int(f) or identity != int(identity) or w <= 0 or h <= 0:
                raise ValueError(f"Invalid frame, identity or box in {path}")
            frames.setdefault(int(f), []).append([x, y, x+w, y+h, identity])
    result = {}
    for frame, rows in frames.items():
        value = np.asarray(rows, dtype=float)
        if len(np.unique(value[:, 4])) != len(value):
            raise ValueError(f"Duplicate GT identity at {sequence}:{frame}")
        result[frame] = value
    return result


def image_path(frame, image_root, modality="image"):
    """Resolve archived relative paths or original absolute paths explicitly."""
    if modality not in frame:
        raise ValueError(f"Frame {frame.get('frame')} has no {modality} path")
    saved = str(frame[modality]).replace("\\", "/")
    root = Path(image_root).expanduser().resolve()
    prefix = "VTTINY_ROOT/data/raw/VT-Tiny-MOT/"
    if saved.startswith(prefix):
        saved = saved[len(prefix):]
    original = Path(saved)
    if original.is_absolute() and original.is_file():
        return original
    # Original machines used absolute prefixes. The official split suffix is stable.
    for split in ("test2017/", "train2017/", "val2017/"):
        if split in saved:
            saved = saved[saved.index(split):]
            break
    relative = Path(saved)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Cannot remap cached image path {saved!r}; use the official image root")
    return root / relative


def forbid_gt(event, args):
    """Prediction-only guard, installed by replay after reading its split/config."""
    if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        path = os.fsdecode(args[0]).lower().replace("\\", "/")
        if path.endswith("/gt.txt") or "/annotations/" in path or path.endswith("/instances_test2017.json"):
            raise PermissionError(f"GT cannot be opened during tracking: {path}")


def selected_sequences(files, split=None, subset="evaluation", sequences=None):
    if sequences:
        names = list(sequences)
        if split and not set(names).issubset(split[subset]):
            raise ValueError(f"Explicit sequences must be within split[{subset!r}]")
    elif split:
        names = list(split[subset])
    else:
        names = sorted(files)
    if not names or len(set(names)) != len(names):
        raise ValueError("Need a nonempty set of unique sequence names")
    missing = set(names) - set(files)
    if missing:
        raise FileNotFoundError(f"Missing caches: {sorted(missing)}")
    return names
