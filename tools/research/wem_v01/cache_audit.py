"""Read-only inventory of cached detections excluded by the native score gate.

This module does not run a tracker, identify objects, or calculate recoverable
misses. Its S0 rows are inventory candidates, not accepted W observations.
Only the Python standard library is required.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import sys
import tempfile
from typing import Any, Iterator, Sequence


SCHEMA = "wem_s0_cache_inventory_v1"
VERSION = "0.1.0"
STATUS = "INVENTORY_ONLY_NOT_W_OPPORTUNITY"
REID_EXPORT_THRESHOLD = 0.1
OUTPUT_NAMES = (
    "manifest.json", "summary.json", "per_sequence.csv", "s0_candidates.jsonl",
)
COUNT_FIELDS = (
    "frames", "frames_with_detections", "frames_with_s0", "detections",
    "native_high", "native_low", "s0", "high_threshold_gap",
    "reid_export_policy_eligible",
    "native_eligible_reid_policy_uncovered",
    "s0_reid_export_policy_eligible",
)
LIMITATIONS = [
    "No tracker state, B ownership/reservations, GMC, identity, GT or images were read.",
    "S0 inventory size is not W eligibility, recoverable FN, or expected metric gain.",
    "RGB flags reproduce an exporter score policy only, not actual feature validity.",
    "Only existing postprocessed cache rows are audited; no S1 or pre-NMS S2 is exported.",
    "No detector weights, cache-to-model consistency, mother-video split or real-data parity is verified.",
]


class AuditError(ValueError):
    """Invalid input or a destination that cannot safely be written."""


@dataclass(frozen=True)
class InputAudit:
    path: Path
    sequence: str
    sha256: str
    size_bytes: int
    counts: dict[str, int]
    first_frame: int | None
    last_frame: int | None
    missing_frame_ids_inside_span: int
    frame_gap_ranges: tuple[tuple[int, int], ...]
    s0_score_min: float | None
    s0_score_max: float | None

    def sequence_row(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            **self.counts,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "missing_frame_ids_inside_span": self.missing_frame_ids_inside_span,
            "s0_score_min": self.s0_score_min,
            "s0_score_max": self.s0_score_max,
            "input_sha256": self.sha256,
        }


def _float32(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AuditError(f"{label}: expected a finite numeric value")
    try:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("non-finite")
        # Exactly the cache conversion before BoxMOT appends integer row IDs.
        result = struct.unpack("<f", struct.pack("<f", number))[0]
    except (ValueError, OverflowError, struct.error) as exc:
        raise AuditError(f"{label}: cannot represent a finite float32 value") from exc
    if not math.isfinite(result):
        raise AuditError(f"{label}: cannot represent a finite float32 value")
    return result


def _threshold(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise AuditError(f"{label} must be a finite number in [0, 1]")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AuditError(f"{label} must be a finite number in [0, 1]") from exc
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise AuditError(f"{label} must be a finite number in [0, 1]")
    return number


def validate_thresholds(low_thresh: float, high_thresh: float) -> tuple[float, float]:
    low = _threshold(low_thresh, "low_thresh")
    high = _threshold(high_thresh, "high_thresh")
    if low >= high:
        raise AuditError("low_thresh must be strictly smaller than high_thresh")
    return low, high


def classify_score(score_float32: float, low_thresh: float, high_thresh: float) -> dict[str, bool]:
    """Use the exact float32 value promoted to double by native hstack.

    The RGB exporter compares a float32 array with 0.1 in the pinned policy,
    so its boundary is the float32 representation of 0.1. This is NOT a
    statement that an embedding exists or that its crop is usable.
    """
    native_low = score_float32 > low_thresh and score_float32 < high_thresh
    native_high = score_float32 > high_thresh
    return {
        "native_low": native_low,
        "native_high": native_high,
        "s0": not native_low and not native_high and score_float32 <= low_thresh,
        "high_threshold_gap": score_float32 == high_thresh,
        "rgb_export_score_policy_eligible": (
            score_float32 > _float32(REID_EXPORT_THRESHOLD, "reid_export_threshold")
        ),
    }


def _reject_constant(value: str) -> None:
    raise AuditError(f"Non-finite JSON constant is not allowed: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise AuditError(f"Duplicate JSON object key: {key}")
        value[key] = item
    return value


def _read(path: Path) -> tuple[dict[str, Any], str, int]:
    try:
        raw = path.read_bytes()
        data = json.loads(
            raw.decode("utf-8-sig"), parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"Cannot read detection cache {path}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("frames"), list):
        raise AuditError(f"{path}: expected an object containing a frames list")
    return data, hashlib.sha256(raw).hexdigest(), len(raw)


def _frames(data: dict[str, Any], path: Path) -> Iterator[tuple[int, str | None, list[list[Any]]]]:
    previous = 0
    for frame_offset, frame in enumerate(data["frames"]):
        label = f"{path.name}: frames[{frame_offset}]"
        if not isinstance(frame, dict):
            raise AuditError(f"{label}: expected an object")
        frame_id = frame.get("frame")
        if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 1:
            raise AuditError(f"{label}.frame: expected a positive integer")
        if frame_id <= previous:
            raise AuditError(
                f"{label}.frame: duplicate or unsorted frame ID {frame_id}; "
                "frames must be strictly increasing (gaps are allowed)"
            )
        previous = frame_id
        detections = frame.get("detections")
        if not isinstance(detections, list):
            raise AuditError(f"{label}.detections: expected a list, including [] for an empty frame")
        image = frame.get("image")
        if image is not None and not isinstance(image, str):
            raise AuditError(f"{label}.image: expected a string or null when supplied")
        yield frame_id, image, detections


def _row(values: Any, path: Path, frame_id: int, row_index: int) -> list[float]:
    label = f"{path.name}: frame {frame_id}, row {row_index}"
    if not isinstance(values, list) or len(values) != 6:
        raise AuditError(f"{label}: detections must have exactly six columns [x1,y1,x2,y2,score,class]")
    row = [_float32(value, f"{label}, column {i}") for i, value in enumerate(values)]
    if row[2] <= row[0] or row[3] <= row[1]:
        raise AuditError(f"{label}: xyxy must have positive width and height after float32 conversion")
    if not 0 <= row[4] <= 1:
        raise AuditError(f"{label}: confidence must be in [0, 1]")
    if row[5] < 0 or not row[5].is_integer():
        raise AuditError(f"{label}: class must be a non-negative integer")
    return row


def _inventory(
    path: Path, low_thresh: float, high_thresh: float, candidate_stream: Any = None,
    expected_sha256: str | None = None,
) -> InputAudit:
    data, digest, size = _read(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise AuditError(f"Input changed after validation: {path}")
    sequence = path.stem
    counts = {key: 0 for key in COUNT_FIELDS}
    first_frame = last_frame = None
    gaps: list[tuple[int, int]] = []
    s0_min = s0_max = None
    for frame_id, image, detections in _frames(data, path):
        if first_frame is None:
            first_frame = frame_id
        if last_frame is not None and frame_id > last_frame + 1:
            gaps.append((last_frame + 1, frame_id - 1))
        last_frame = frame_id
        counts["frames"] += 1
        counts["frames_with_detections"] += bool(detections)
        frame_s0 = 0
        for index, values in enumerate(detections):
            row = _row(values, path, frame_id, index)
            flags = classify_score(row[4], low_thresh, high_thresh)
            counts["detections"] += 1
            for key in ("native_high", "native_low", "s0", "high_threshold_gap"):
                counts[key] += flags[key]
            policy = flags["rgb_export_score_policy_eligible"]
            counts["reid_export_policy_eligible"] += policy
            counts["native_eligible_reid_policy_uncovered"] += (
                (flags["native_high"] or flags["native_low"]) and not policy
            )
            if not flags["s0"]:
                continue
            frame_s0 += 1
            counts["s0_reid_export_policy_eligible"] += policy
            s0_min = row[4] if s0_min is None else min(s0_min, row[4])
            s0_max = row[4] if s0_max is None else max(s0_max, row[4])
            if candidate_stream is not None:
                item = {
                    "schema": "wem_s0_candidate_v1",
                    "observation_key": {
                        "sequence": sequence, "frame_id": frame_id,
                        "source": "native_cache", "original_row": index,
                    },
                    "sequence": sequence,
                    "frame_id": frame_id,
                    "original_row": index,
                    "image": image,
                    "raw_xyxy": row[:4],
                    "raw_score_float32": row[4],
                    "raw_class": int(row[5]),
                    "native_high_mask": flags["native_high"],
                    "native_low_mask": flags["native_low"],
                    "source_flags": {"S0_below_native_low": True},
                    "rgb_export_score_policy_eligible": policy,
                    "actual_rgb_feature_valid": None,
                    "owner_or_reservation": "NOT_AUDITED",
                    "track_eligibility": "NOT_AUDITED",
                    "identity_label": "NOT_AUDITED",
                }
                candidate_stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
        counts["frames_with_s0"] += frame_s0 > 0
    if counts["detections"] != sum(
        counts[key] for key in ("native_high", "native_low", "s0", "high_threshold_gap")
    ):
        raise AuditError(f"Internal score partition error for {path}")
    return InputAudit(
        path=path, sequence=sequence, sha256=digest, size_bytes=size, counts=counts,
        first_frame=first_frame, last_frame=last_frame,
        missing_frame_ids_inside_span=sum(end - start + 1 for start, end in gaps),
        frame_gap_ranges=tuple(gaps), s0_score_min=s0_min, s0_score_max=s0_max,
    )


def _inputs(detections_dir: Path, sequences: Sequence[str] | None) -> list[Path]:
    if not detections_dir.is_dir():
        raise AuditError(f"Detection cache directory does not exist: {detections_dir}")
    choices = {path.stem: path for path in sorted(detections_dir.glob("*.json")) if path.is_file()}
    if not choices:
        raise AuditError(f"No *.json cache files found directly inside {detections_dir}")
    if sequences is None:
        return list(choices.values())
    if not sequences or len(set(sequences)) != len(sequences):
        raise AuditError("--sequence selection must be non-empty and contain no duplicates")
    missing = sorted(set(sequences) - choices.keys())
    if missing:
        raise AuditError(f"Requested sequence cache files are missing: {', '.join(missing)}")
    # Deterministic output independent of CLI option order.
    return [choices[name] for name in sorted(sequences)]


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _validate_destination(output: Path, input_root: Path, paths: list[Path]) -> None:
    if _is_within(output, input_root) or _is_within(input_root, output):
        raise AuditError("output_dir and detections_dir must not contain one another")
    for path in paths:
        if _is_within(path.resolve(), output):
            raise AuditError(f"output_dir overlaps an input cache: {path}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise AuditError(f"Refusing to overwrite a non-empty or non-directory output: {output}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_directory(
    detections_dir: str | Path, output_dir: str | Path, *,
    low_thresh: float, high_thresh: float,
    sequences: Sequence[str] | None = None,
    cli_arguments: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate all input caches, then publish one non-destructive audit directory.

    Relative paths are resolved against the current working directory.
    Existing non-empty output directories and overlaps with inputs are rejected.
    All file hashes are checked again before candidate rows are emitted.
    """
    low_thresh, high_thresh = validate_thresholds(low_thresh, high_thresh)
    input_root = Path(detections_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    paths = _inputs(input_root, sequences)
    _validate_destination(output, input_root, paths)

    # Validate the entire selected batch before creating any output files.
    audits = [_inventory(path, low_thresh, high_thresh) for path in paths]
    totals = {key: sum(item.counts[key] for item in audits) for key in COUNT_FIELDS}
    summary = {
        "schema": SCHEMA,
        "tool_version": VERSION,
        "status": STATUS,
        "sequences": len(audits),
        "totals": totals,
        "score_thresholds": {"low": low_thresh, "high": high_thresh},
        "score_partition": {
            "native_high": "float64(float32(score)) > high",
            "native_low": "low < float64(float32(score)) < high",
            "S0": "not high and not low and float64(float32(score)) <= low",
            "high_threshold_gap": "float64(float32(score)) == high",
        },
        "reid_export_policy": {
            "comparison": "float32(score) > float32(0.1)",
            "threshold_float32": _float32(REID_EXPORT_THRESHOLD, "reid_export_threshold"),
            "actual_feature_files_read": False,
            "interpretation": "score-policy coverage only; actual feature validity is unknown",
        },
        "frame_gaps": [
            {
                "sequence": item.sequence,
                "ranges": [list(pair) for pair in item.frame_gap_ranges],
                "missing_frame_ids_inside_span": item.missing_frame_ids_inside_span,
            }
            for item in audits if item.frame_gap_ranges
        ],
        "gt_read": False,
        "tracker_executed": False,
        "w_online_enabled": False,
        "limitations": LIMITATIONS,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        with (stage / "s0_candidates.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for first in audits:
                second = _inventory(
                    first.path, low_thresh, high_thresh, candidate_stream=stream,
                    expected_sha256=first.sha256,
                )
                if second != first:
                    raise AuditError(f"Input changed between validation and emission: {first.path}")
        fieldnames = list(audits[0].sequence_row())
        with (stage / "per_sequence.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(item.sequence_row() for item in audits)
        _write_json(stage / "summary.json", summary)
        manifest = {
            "schema": "wem_s0_manifest_v1",
            "tool_version": VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": STATUS,
            "arguments": {
                "detections_dir": str(input_root), "output_dir": str(output),
                "low_thresh": low_thresh, "high_thresh": high_thresh,
                "sequences": list(sequences) if sequences is not None else None,
            },
            "cli_arguments": list(cli_arguments) if cli_arguments is not None else None,
            "source_script_sha256": _sha256_file(Path(__file__)),
            "inputs": [
                {
                    "sequence": item.sequence, "path": str(item.path.resolve()),
                    "size_bytes": item.size_bytes, "sha256": item.sha256,
                }
                for item in audits
            ],
            "outputs": [
                {
                    "path": name, "size_bytes": (stage / name).stat().st_size,
                    "sha256": _sha256_file(stage / name),
                }
                for name in OUTPUT_NAMES if name != "manifest.json"
            ],
            "gt_read": False,
            "tracker_executed": False,
            "w_online_enabled": False,
        }
        _write_json(stage / "manifest.json", manifest)
        # Refuse new contents if the destination changed while inputs were read.
        _validate_destination(output, input_root, paths)
        if output.exists():
            output.rmdir()  # succeeds only while still empty
        stage.rename(output)
        return summary
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inventory existing S0 cache rows without GT, images, GPU or tracking. "
            "Output counts are NOT W opportunities or recoverable FN."
        ),
    )
    parser.add_argument("--detections-dir", required=True, type=Path, help="Flat directory of sequence JSON caches.")
    parser.add_argument("--output-dir", required=True, type=Path, help="New or empty audit directory; never overwrites results.")
    parser.add_argument("--low-thresh", required=True, type=float, help="Actual reference track_low_thresh, e.g. 0.1.")
    parser.add_argument("--high-thresh", required=True, type=float, help="Actual reference track_high_thresh, e.g. 0.5.")
    parser.add_argument("--sequence", action="append", dest="sequences", help="Select one sequence stem; repeat for a subset.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_args)
    try:
        summary = audit_directory(
            args.detections_dir, args.output_dir, low_thresh=args.low_thresh,
            high_thresh=args.high_thresh, sequences=args.sequences,
            cli_arguments=raw_args,
        )
    except (AuditError, OSError) as exc:
        print(f"S0 audit failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": summary["status"], "output_dir": str(args.output_dir.resolve()),
        "sequences": summary["sequences"], "totals": summary["totals"],
    }, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
