#!/usr/bin/env python3
"""Differential SOFTWARE replay against a local prior Git revision.

No dataset accuracy is measured. The reference is read with git archive, never
checked out over the user's work. Requires the same pinned native runtime as the
runner. Both revisions execute in isolated subprocesses to reset global IDs.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]
BASE = "6d9c258556a143036cebde891c93079a772b9945"


def replay(source):
    sys.path.insert(0, str(source))
    import cv2
    import numpy as np
    import torch
    from botsort_plus.config import PlusConfig
    from botsort_plus.native_adapter import BoTSORTPlus
    from botsort_plus.types import Feature
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    params = dict(reid_weights=Path("unused-cache-only.pt"), device=torch.device("cpu"),
                  half=False, with_reid=True, det_thresh=.5, track_high_thresh=.5,
                  track_low_thresh=.1, new_track_thresh=.6, match_thresh=.8,
                  track_buffer=60, frame_rate=15, cmc_method="sof",
                  proximity_thresh=.5, appearance_thresh=.4, fuse_first_associate=False)
    image = np.random.default_rng(416).integers(0, 256, (128, 192, 3), np.uint8)
    def row(x, score, vector, quality=1.):
        return ([x, 50, x + 4, 54, score, 0],
                {"rgb": Feature(np.asarray(vector, np.float32), quality),
                 "thermal": Feature(np.asarray(vector[::-1], np.float32), 1.)})
    frames = []
    for frame in range(1, 61):
        entries = []
        if frame <= 8 or 12 <= frame <= 21 or frame >= 56:
            entries.append(row(20 if frame <= 8 else 23, .3 if 12 <= frame <= 14 else .95,
                               [1., 0.], .2 if frame == 60 else 1.))
        if frame not in (25, 26, 27):
            entries.append(row(90 + min(frame, 20) * .03,
                               .3 if frame % 7 == 0 else .9,
                               [0., 1.]))
        if 3 <= frame <= 13:
            entries.append(row(140, .2, [1., 0.]))
        if frame >= 18:
            entries.append(row(110, .95 if frame != 31 else .5, [.8, .6]))
        if frame == 30:
            entries.append(row(160, float(np.nextafter(np.float32(.1), np.float32(0.))), [1., 0.]))
        frames.append(entries)
    results = {}
    for config_path in sorted((source / "configs/botsort_plus").glob("*.json")):
        if int(config_path.name.split("_", 1)[0]) > 26:
            continue
        cv2.setRNGSeed(0)
        tracker = BoTSORTPlus(PlusConfig.load(config_path), **params)
        output_hash = hashlib.sha256()
        event_hash = hashlib.sha256()
        row_count = 0
        for entries in frames:
            dets = np.asarray([r for r, _ in entries], np.float32).reshape(-1, 6)
            packets = [{k: Feature(f.vector.copy(), f.quality) for k, f in p.items()}
                       for _, p in entries]
            result = tracker.update(dets, image.copy(), features=packets).reshape(-1, 8)
            output_hash.update(np.asarray(result.shape, np.int64).tobytes())
            output_hash.update(result.tobytes())
            row_count += len(result)
            event_hash.update(json.dumps(tracker.last_events, sort_keys=True,
                                         allow_nan=False).encode())
        results[config_path.name] = dict(frames=len(frames), output_rows=row_count,
                                        prediction_sha256=output_hash.hexdigest(),
                                        events_sha256=event_hash.hexdigest())
    print(json.dumps(results, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-sha", default=BASE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker-source", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_source:
        replay(args.worker_source.resolve())
        return 0
    if args.output and args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    base = subprocess.check_output(["git", "rev-parse", "--verify", args.base_sha + "^{commit}"],
                                   cwd=ROOT, text=True).strip()
    archive = subprocess.check_output(["git", "archive", base, "botsort_plus", "configs/botsort_plus"],
                                      cwd=ROOT)
    with tempfile.TemporaryDirectory(prefix="botsort_c2_reference_") as temporary:
        reference = Path(temporary)
        with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
            stream.extractall(reference, filter="data")
        runs = []
        for source in (reference, ROOT):
            completed = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                        "--worker-source", str(source)], cwd=source,
                                       env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
                                       text=True, capture_output=True, check=True)
            runs.append(json.loads(completed.stdout))
    keys = sorted(set(runs[0]) | set(runs[1]))
    report = {"purpose": "SYNTHETIC_SOFTWARE_COMPATIBILITY_NOT_DATASET_ACCURACY",
              "reference_commit": base, "config_count": len(keys),
              "equal": runs[0] == runs[1],
              "comparisons": {key: {"equal": runs[0].get(key) == runs[1].get(key),
                                    "reference": runs[0].get(key), "current": runs[1].get(key)} for key in keys}}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "comparisons"}, indent=2))
    return 0 if report["equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
