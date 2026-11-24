"""Export real, row-aligned RGB/(optional) thermal embeddings for A/B/C replay.

The frozen detections are never changed. Every detection receives a cache row;
skipped/invalid features have validity=False and a zero placeholder. Unlike the
baseline high-confidence-only cache, this exporter extracts scores > 0.1 as well.
No detector, ReID training, or image/GT fabrication occurs here.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Callable

import numpy as np

from .appearance import estimate_quality


SCHEMA = "botsort_plus_features_v1"


def validate_spatial_mapping(mapping: dict | None) -> dict:
    """Require an explicit RGB-coordinate -> thermal-coordinate declaration."""
    if not isinstance(mapping, dict) or mapping.get("declared_by_user") is not True:
        raise ValueError("Thermal features require explicit aligned or RGB-to-thermal homography declaration")
    mode = mapping.get("mode")
    if mode == "aligned":
        return {"mode": mode, "declared_by_user": True}
    if mode != "homography":
        raise ValueError("spatial_mapping.mode must be aligned or homography")
    matrix = np.asarray(mapping.get("rgb_to_thermal"), dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("RGB-to-thermal homography must be a finite 3x3 matrix")
    if np.linalg.matrix_rank(matrix) < 3:
        raise ValueError("RGB-to-thermal homography must be nonsingular")
    return {"mode": mode, "rgb_to_thermal": matrix.tolist(), "declared_by_user": True}


def map_boxes_to_thermal(boxes: np.ndarray, thermal_shape: tuple, mapping: dict,
                         return_visibility: bool = False):
    """Transform all four corners, then clip; horizon-crossing boxes are invalid.

    Mapping does not establish temporal synchronization or identity alignment;
    these remain a property of the provided paired image paths and calibration.
    """
    mapping = validate_spatial_mapping(mapping)
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError("Boxes must be finite positive-area xyxy")
    if len(thermal_shape) < 2 or min(thermal_shape[:2]) <= 0:
        raise ValueError("Thermal image must have positive height and width")
    corners = np.stack((boxes[:, [0, 1]], boxes[:, [2, 1]],
                        boxes[:, [2, 3]], boxes[:, [0, 3]]), axis=1)
    valid = np.ones(len(boxes), dtype=bool)
    if mapping["mode"] == "homography":
        matrix = np.asarray(mapping["rgb_to_thermal"], dtype=float)
        homogeneous = np.concatenate((corners, np.ones((*corners.shape[:2], 1))), axis=2)
        mapped = homogeneous @ matrix.T
        denominators = mapped[..., 2]
        valid &= np.all(np.abs(denominators) > 1e-10, axis=1)
        valid &= np.all(denominators > 0, axis=1) | np.all(denominators < 0, axis=1)
        safe = np.where(np.abs(denominators) > 1e-10, denominators, 1.0)
        corners = mapped[..., :2] / safe[..., None]
    mapped_boxes = np.concatenate((corners.min(axis=1), corners.max(axis=1)), axis=1)
    valid &= np.isfinite(mapped_boxes).all(axis=1)
    original_area = np.prod(np.maximum(mapped_boxes[:, 2:] - mapped_boxes[:, :2], 0), axis=1)
    h, w = thermal_shape[:2]
    mapped_boxes[:, [0, 2]] = np.clip(mapped_boxes[:, [0, 2]], 0, w)
    mapped_boxes[:, [1, 3]] = np.clip(mapped_boxes[:, [1, 3]], 0, h)
    valid &= np.all(mapped_boxes[:, 2:] > mapped_boxes[:, :2], axis=1)
    visible_area = np.prod(np.maximum(mapped_boxes[:, 2:] - mapped_boxes[:, :2], 0), axis=1)
    visibility = np.divide(visible_area, original_area, out=np.zeros_like(visible_area),
                           where=(original_area > 0) & np.isfinite(original_area))
    visibility = np.clip(visibility, 0, 1)
    visibility[~valid] = 0
    mapped_boxes[~valid] = 0
    if return_visibility:
        return mapped_boxes, valid, visibility
    return mapped_boxes, valid


def _valid_rgb_crop(boxes: np.ndarray, image: np.ndarray) -> np.ndarray:
    # BoxMOT 16.0.11 rounds before cropping. Preserve exactly that behavior for
    # the old baseline; do not silently enlarge a subpixel crop to make it valid.
    rounded = np.rint(boxes).copy()
    h, w = image.shape[:2]
    rounded[:, [0, 2]] = np.clip(rounded[:, [0, 2]], 0, w)
    rounded[:, [1, 3]] = np.clip(rounded[:, [1, 3]], 0, h)
    return np.all(rounded[:, 2:] > rounded[:, :2], axis=1)


def _normalized_batch(output, rows: int, modality: str) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(output, "detach"):
        output = output.detach().cpu().numpy()
    vectors = np.asarray(output, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != rows or vectors.shape[1] == 0:
        raise ValueError(f"{modality} encoder must return N x D features; got {vectors.shape} for N={rows}")
    if not np.isfinite(vectors).all():
        raise ValueError(f"{modality} encoder returned NaN/infinity")
    norms = np.linalg.norm(vectors.astype(float), axis=1)
    valid = norms > 1e-12
    vectors = vectors.copy()
    vectors[valid] /= norms[valid, None]
    vectors[~valid] = 0
    return vectors, valid


def _encode_batches(encoder, boxes, image, modality, batch_size):
    features, masks = [], []
    start = time.perf_counter()
    for offset in range(0, len(boxes), batch_size):
        batch = boxes[offset:offset + batch_size]
        vectors, valid = _normalized_batch(encoder.get_features(batch, image), len(batch), modality)
        features.append(vectors)
        masks.append(valid)
    elapsed = time.perf_counter() - start
    if features and len({f.shape[1] for f in features}) != 1:
        raise ValueError(f"{modality} encoder dimension changed between batches")
    return np.concatenate(features), np.concatenate(masks), elapsed


def _default_image_loader(path: Path, modality: str) -> np.ndarray:
    import cv2
    image = cv2.imread(str(path), cv2.IMREAD_COLOR if modality == "rgb" else cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not decode {modality} image: {path}")
    return image


def export_sequence(
    cache_path: Path,
    output_dir: Path,
    rgb_encoder,
    rgb_metadata: dict,
    image_root: Path,
    thermal_encoder=None,
    thermal_metadata: dict | None = None,
    spatial_mapping: dict | None = None,
    overwrite: bool = False,
    image_loader: Callable | None = None,
    batch_size: int = 64,
) -> dict:
    """Write one .npz and checksum-bearing .json, one row per frozen detection.

    Encoders implement get_features(boxes_xyxy, image) -> N x D and may provide
    output_dim for an entirely empty sequence. Both files are replaced atomically
    individually; readers validate the NPZ hash against the sidecar and reject an
    incomplete/mixed pair. ``image_loader`` injection exists for mechanism tests.
    """
    from .io import read_detection_cache, detection_fingerprint, sha256, write_json, resolve_image
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    cache_path, output_dir, image_root = Path(cache_path), Path(output_dir), Path(image_root)
    name = cache_path.name.removesuffix(".gz").removesuffix(".json")
    npz_path, json_path = output_dir / f"{name}.npz", output_dir / f"{name}.json"
    if not overwrite and (npz_path.exists() or json_path.exists()):
        raise FileExistsError(f"Feature output already exists for {name}; use --overwrite explicitly")
    if not isinstance(rgb_metadata, dict) or not rgb_metadata.get("sha256"):
        raise ValueError("RGB encoder provenance must contain sha256")
    thermal_enabled = thermal_encoder is not None
    if thermal_enabled:
        if not isinstance(thermal_metadata, dict) or not thermal_metadata.get("sha256"):
            raise ValueError("Thermal encoder provenance must contain sha256")
        spatial_mapping = validate_spatial_mapping(spatial_mapping)
    elif thermal_metadata is not None or spatial_mapping is not None:
        raise ValueError("Thermal metadata/mapping requires a real thermal encoder")
    cache = read_detection_cache(cache_path)
    loader = image_loader if image_loader is not None else _default_image_loader
    modalities = ["rgb", "thermal"] if thermal_enabled else ["rgb"]
    pieces = {m: [] for m in modalities}
    dimensions = {"rgb": getattr(rgb_encoder, "output_dim", rgb_metadata.get("output_dim"))}
    if thermal_enabled:
        dimensions["thermal"] = getattr(thermal_encoder, "output_dim", thermal_metadata.get("output_dim"))
    offsets, frame_ids = [0], []
    seconds = {m: 0.0 for m in modalities}
    counts = {"detections": 0, "eligible_score_gt_0_1": 0, "high_score_gt_0_5": 0,
              **{f"{m}_valid": 0 for m in modalities}, **{f"{m}_zero_outputs": 0 for m in modalities}}
    wall_start = time.perf_counter()
    for frame in cache["frames"]:
        frame_id = int(frame["frame"])
        frame_ids.append(frame_id)
        # The export policy is float32 score > .1. Native BoxMOT subsequently
        # hstacks integer row indices and may promote scores to float64, so exact
        # lower-boundary behavior is not claimed bit-identical to its low gate.
        # The high gate .5 is exactly representable and must always have RGB.
        detections = np.asarray(frame["detections"], dtype=np.float32).reshape(-1, 6)
        n = len(detections)
        offsets.append(offsets[-1] + n)
        counts["detections"] += n
        eligible = detections[:, 4] > 0.1
        high = detections[:, 4] > 0.5
        counts["eligible_score_gt_0_1"] += int(eligible.sum())
        counts["high_score_gt_0_5"] += int(high.sum())
        rgb = loader(resolve_image(frame["image"], image_root), "rgb")
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("Baseline BoxMOT RGB encoding expects native uint8 BGR HxWx3 images")
        images, boxes, usable = {"rgb": rgb}, {"rgb": detections[:, :4]}, {"rgb": eligible & _valid_rgb_crop(detections[:, :4], rgb)}
        visibility = {"rgb": np.ones(n)}
        if np.any(high & ~usable["rgb"]):
            raise ValueError(f"{name} frame {frame_id}: invalid RGB crop for high-score detection; baseline ReID cannot be reproduced")
        if thermal_enabled:
            if "thermal" not in frame:
                raise ValueError(f"{name} frame {frame_id}: no paired thermal image path")
            thermal = loader(resolve_image(frame["thermal"], image_root), "thermal")
            if spatial_mapping["mode"] == "aligned" and thermal.shape[:2] != rgb.shape[:2]:
                raise ValueError("--thermal-aligned requires the same RGB/thermal image dimensions; supply a homography otherwise")
            thermal_boxes, valid_mapping, visible_fraction = map_boxes_to_thermal(
                detections[:, :4], thermal.shape, spatial_mapping, return_visibility=True)
            images["thermal"], boxes["thermal"] = thermal, thermal_boxes
            usable["thermal"] = eligible & valid_mapping
            visibility["thermal"] = visible_fraction
        for modality in modalities:
            indices = np.flatnonzero(usable[modality])
            valid = np.zeros(n, dtype=bool)
            quality = np.zeros(n, dtype=np.float32)
            vectors = None
            if len(indices):
                encoder = rgb_encoder if modality == "rgb" else thermal_encoder
                vectors, actual_valid, elapsed = _encode_batches(encoder, boxes[modality][indices], images[modality], modality, batch_size)
                seconds[modality] += elapsed
                dim = vectors.shape[1]
                if dimensions[modality] is not None and dimensions[modality] != dim:
                    raise ValueError(f"{modality} embedding dimension changed/does not match declared output_dim")
                dimensions[modality] = dim
                valid[indices] = actual_valid
                counts[f"{modality}_zero_outputs"] += int((~actual_valid).sum())
                for index in indices[actual_valid]:
                    quality[index] = estimate_quality(images[modality], boxes[modality][index], modality) * visibility[modality][index]
                counts[f"{modality}_valid"] += int(actual_valid.sum())
            if modality == "rgb" and np.any(high & ~valid):
                raise ValueError(f"{name} frame {frame_id}: zero/missing RGB features for high-score detection")
            pieces[modality].append((n, indices, vectors, valid, quality))
    arrays = {"frame_ids": np.asarray(frame_ids, dtype=np.int64), "offsets": np.asarray(offsets, dtype=np.int64)}
    for modality in modalities:
        dimension = dimensions[modality]
        if dimension is None or int(dimension) <= 0:
            raise ValueError(f"No {modality} features in sequence and encoder output_dim is unknown")
        features = np.zeros((offsets[-1], int(dimension)), dtype=np.float32)
        valid = np.zeros(offsets[-1], dtype=bool)
        quality = np.zeros(offsets[-1], dtype=np.float32)
        position = 0
        for n, indices, vectors, frame_valid, frame_quality in pieces[modality]:
            if vectors is not None:
                features[position + indices] = vectors
            valid[position:position + n] = frame_valid
            quality[position:position + n] = frame_quality
            position += n
        arrays[modality], arrays[f"{modality}_valid"], arrays[f"{modality}_quality"] = features, valid, quality
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_paths = []
    try:
        with tempfile.NamedTemporaryFile(dir=output_dir, prefix=f".{name}-", suffix=".npz.tmp", delete=False) as stream:
            npz_temp = Path(stream.name)
            temporary_paths.append(npz_temp)
            np.savez_compressed(stream, **arrays)
        metadata = {
            "schema": SCHEMA, "sequence": name,
            "detection_fingerprint": detection_fingerprint(cache),
            "detection_file_sha256": sha256(cache_path), "features_sha256": sha256(npz_temp),
            "encoders": {"rgb": {**rgb_metadata, "output_dim": int(dimensions["rgb"])}},
            "spatial_mapping": spatial_mapping,
            "selection": "float32 score > 0.1; ALL detection rows retained; RGB score > 0.5 must be valid",
            "threshold_boundary_note": "Native BoxMOT may promote scores during hstack; float32 .1 boundary is not claimed identical to its low gate",
            "encoder_batch_size": batch_size,
            "quality_method": "native-pixel size/clip/contrast/gradient proxy v1; uncalibrated",
            "extraction_seconds": float(sum(seconds.values())),
            "encoder_seconds": seconds, "export_wall_seconds": time.perf_counter() - wall_start,
            "coverage": counts,
        }
        if thermal_enabled:
            metadata["encoders"]["thermal"] = {**thermal_metadata, "output_dim": int(dimensions["thermal"])}
        with tempfile.NamedTemporaryFile(dir=output_dir, prefix=f".{name}-", suffix=".json.tmp", delete=False) as stream:
            json_temp = Path(stream.name)
            temporary_paths.append(json_temp)
        write_json(json_temp, metadata)
        # A reader's checksum validation rejects the transient interval between
        # these two replacements, rather than accepting stale mixed artifacts.
        os.replace(npz_temp, npz_path)
        os.replace(json_temp, json_path)
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)
    return metadata


def load_rgb_encoder(weights: Path, device: str = "cuda:0", runtime_root: Path | None = None):
    """Load and verify the supplied baseline CLIP-VeRi image encoder checkpoint."""
    from .io import sha256
    weights = Path(weights).resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)
    if "clip" not in weights.name.lower() or "veri" not in weights.name.lower() or weights.suffix != ".pt":
        raise ValueError("Keep the baseline RGB checkpoint filename clip_veri.pt: BoxMOT infers architecture and 256x256 crops from its name")
    if runtime_root is not None:
        runtime = Path(runtime_root).resolve() / "tracking_runtime"
        if not runtime.is_dir():
            raise FileNotFoundError(f"Missing baseline tracking runtime: {runtime}")
        sys.path.insert(0, str(runtime))
    import torch
    from boxmot.reid.core.auto_backend import ReidAutoBackend
    from boxmot.reid.backbones.clip.clip import clip
    original_download = clip._download
    if runtime_root is not None:
        # Reuse the same CLIP backbone cache location as baseline
        # tools/vttiny_reid_adapter.py instead of requiring a new download.
        clip_cache = Path(runtime_root).resolve() / "reid_weights"
        clip._download = lambda url: original_download(url, root=str(clip_cache))
    try:
        backend = ReidAutoBackend(weights=weights, device=torch.device(device), half=False).model
    finally:
        clip._download = original_download
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict", checkpoint)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    current = backend.model.state_dict()
    relevant = [key for key in current if key.startswith(("image_encoder.", "bottleneck.", "bottleneck_proj."))]
    if len(relevant) <= 100:
        raise ValueError("The loaded RGB model is not the expected baseline CLIP-ReID visual encoder")
    for key in relevant:
        if key not in state or current[key].shape != state[key].shape:
            raise ValueError(f"RGB checkpoint is missing/incompatible with visual state key {key}")
        if not torch.equal(current[key].cpu(), state[key].to(current[key].dtype)):
            raise ValueError(f"RGB encoder state did not load exactly: {key}")
    if tuple(backend.input_shape) != (256, 256):
        raise ValueError("Baseline CLIP-VeRi input size must be 256x256")
    backend.output_dim = 1280
    return backend, {"sha256": sha256(weights), "weights_name": weights.name,
                     "encoder": "baseline CLIP-ReID ViT-B/16 VeRi, supplied fine-tuned checkpoint",
                     "input_hw": [256, 256], "output_dim": 1280, "precision": "float32",
                     "verified_visual_state_keys": len(relevant), "crop_source": "RGB BGR-to-RGB",
                     "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}


class TorchScriptCropEncoder:
    """Run a user-supplied thermal ReID model with explicit preprocessing.

    The model must accept NCHW and return an N x D Tensor. One-channel input is
    grayscale; three-channel input converts BGR to RGB or replicates grayscale.
    uint16 is divided by 65535 (uint8 by 255); exported training preprocessing
    must agree with this declared convention. No pretrained weights are included.
    """

    def __init__(self, weights: Path, input_size: tuple[int, int], channels: int,
                 mean: list[float], std: list[float], device: str = "cuda:0"):
        import torch
        from .io import sha256
        self.weights = Path(weights).resolve()
        if not self.weights.is_file():
            raise FileNotFoundError(self.weights)
        if channels not in (1, 3) or len(input_size) != 2 or min(input_size) <= 0:
            raise ValueError("Thermal input requires channels=1 or 3 and positive H W")
        mean, std = np.asarray(mean, float), np.asarray(std, float)
        if mean.shape != (channels,) or std.shape != (channels,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
            raise ValueError("Thermal mean/std must have one finite value per channel; std must be positive")
        self.device = torch.device(device)
        self.channels, self.input_shape = channels, tuple(input_size)
        self.mean = torch.tensor(mean, dtype=torch.float32, device=self.device).view(1, channels, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32, device=self.device).view(1, channels, 1, 1)
        self.model = torch.jit.load(str(self.weights), map_location=self.device).eval()
        # This probe determines dimensions only. It is never written as an image
        # observation, feature row, quality, or evidence in the output cache.
        with torch.inference_mode():
            probe = self.model(torch.zeros((1, channels, *self.input_shape), device=self.device))
        if not isinstance(probe, torch.Tensor) or probe.ndim != 2 or probe.shape[0] != 1 or probe.shape[1] < 1:
            raise ValueError("Thermal TorchScript model must return one N x D feature Tensor")
        self.output_dim = int(probe.shape[1])
        self.metadata = {"sha256": sha256(self.weights), "weights_name": self.weights.name,
                         "encoder": "user-supplied thermal TorchScript ReID", "input_hw": list(self.input_shape),
                         "channels": channels, "mean": mean.tolist(), "std": std.tolist(),
                         "output_dim": self.output_dim, "precision": "float32",
                         "resize": "cv2.INTER_LINEAR", "crop_rounding": "floor min, ceil max",
                         "intensity_scaling": "integer/dtype_max; float inputs must already be [0,1]",
                         "color_order": "grayscale" if channels == 1 else "RGB or repeated grayscale"}

    def get_features(self, boxes: np.ndarray, image: np.ndarray) -> np.ndarray:
        import cv2
        import torch
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        if not len(boxes):
            return np.empty((0, self.output_dim), np.float32)
        if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (1, 3)):
            raise ValueError("Thermal image must have 1 or 3 channels")
        h, w = image.shape[:2]
        crops = []
        for box in boxes:
            if not np.isfinite(box).all():
                raise ValueError("Thermal crop box is nonfinite")
            x1, y1 = np.maximum(np.floor(box[:2]), (0, 0)).astype(int)
            x2, y2 = np.minimum(np.ceil(box[2:]), (w, h)).astype(int)
            if x2 <= x1 or y2 <= y1:
                raise ValueError("Thermal crop has no native pixels")
            crop = image[y1:y2, x1:x2]
            if crop.ndim == 3 and crop.shape[2] == 1:
                crop = crop[..., 0]
            if crop.ndim == 3:
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY if self.channels == 1 else cv2.COLOR_BGR2RGB)
            elif self.channels == 3:
                crop = np.repeat(crop[..., None], 3, axis=2)
            crop = cv2.resize(crop, (self.input_shape[1], self.input_shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            if not np.isfinite(crop).all() or np.min(crop) < 0:
                raise ValueError("Thermal crop contains invalid intensities")
            if np.issubdtype(image.dtype, np.integer):
                crop /= float(np.iinfo(image.dtype).max)
            elif not np.issubdtype(image.dtype, np.floating) or np.max(crop) > 1:
                raise ValueError("Float thermal input must be explicitly normalized to [0,1]")
            if crop.ndim == 2:
                crop = crop[..., None]
            crops.append(np.moveaxis(crop, -1, 0))
        tensor = torch.from_numpy(np.stack(crops)).to(self.device)
        tensor = (tensor - self.mean) / self.std
        with torch.inference_mode():
            output = self.model(tensor)
        if not isinstance(output, torch.Tensor):
            raise ValueError("Thermal TorchScript encoder did not return an N x D Tensor")
        return _normalized_batch(output, len(boxes), "thermal")[0]


def forbid_ground_truth_reads(event, args):
    """Same inference-only GT boundary as the frozen baseline scripts."""
    if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        path = os.fsdecode(args[0]).lower().replace("\\", "/")
        if path.endswith("/gt.txt") or "/annotations/" in path or path.endswith("/instances_test2017.json"):
            raise PermissionError(f"Ground truth is forbidden during feature extraction: {path}")


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections-dir", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--rgb-weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sequence", action="append", help="Optional sequence name; repeat to select several")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--thermal-torchscript", type=Path)
    parser.add_argument("--thermal-input-size", type=int, nargs=2, metavar=("H", "W"))
    parser.add_argument("--thermal-channels", type=int, choices=(1, 3))
    parser.add_argument("--thermal-mean", type=float, nargs="+")
    parser.add_argument("--thermal-std", type=float, nargs="+")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--thermal-aligned", action="store_true")
    group.add_argument("--thermal-homography", type=Path, help="JSON 3x3 matrix mapping RGB -> thermal coordinates")
    args = parser.parse_args(argv)
    if not args.detections_dir.is_dir() or not args.images_root.is_dir():
        parser.error("--detections-dir and --images-root must exist")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    thermal_options = [args.thermal_input_size, args.thermal_channels, args.thermal_mean, args.thermal_std]
    if args.thermal_torchscript:
        if any(v is None for v in thermal_options) or not (args.thermal_aligned or args.thermal_homography):
            parser.error("Thermal export requires explicit size, channels, mean, std and spatial mapping")
    elif any(v is not None for v in thermal_options) or args.thermal_aligned or args.thermal_homography:
        parser.error("Thermal preprocessing/mapping requires --thermal-torchscript")
    sys.addaudithook(forbid_ground_truth_reads)
    paths = sorted([*args.detections_dir.glob("*.json"), *args.detections_dir.glob("*.json.gz")])
    by_name = {}
    for path in paths:
        name = path.name.removesuffix(".gz").removesuffix(".json")
        if name in by_name:
            parser.error(f"Ambiguous duplicate detection files for {name}")
        by_name[name] = path
    if args.sequence:
        missing = set(args.sequence) - set(by_name)
        if missing:
            parser.error(f"Unknown sequences: {sorted(missing)}")
        by_name = {name: by_name[name] for name in dict.fromkeys(args.sequence)}
    if not by_name:
        parser.error("No detection cache files found")
    if not args.overwrite:
        existing = [name for name in by_name if (args.output_dir / f"{name}.npz").exists()
                    or (args.output_dir / f"{name}.json").exists()]
        if existing:
            parser.error(f"Feature outputs already exist: {existing}; use --overwrite explicitly")
    rgb, rgb_metadata = load_rgb_encoder(args.rgb_weights, args.device, args.runtime_root)
    thermal, thermal_metadata, mapping = None, None, None
    if args.thermal_torchscript:
        mapping = ({"mode": "aligned", "declared_by_user": True} if args.thermal_aligned else
                   {"mode": "homography", "declared_by_user": True,
                    "rgb_to_thermal": json.loads(args.thermal_homography.read_text())})
        mapping = validate_spatial_mapping(mapping)
        thermal = TorchScriptCropEncoder(args.thermal_torchscript, tuple(args.thermal_input_size),
                                         args.thermal_channels, args.thermal_mean, args.thermal_std, args.device)
        thermal_metadata = thermal.metadata
    for name, path in by_name.items():
        metadata = export_sequence(path, args.output_dir, rgb, rgb_metadata, args.images_root,
                                   thermal, thermal_metadata, mapping, args.overwrite, batch_size=args.batch_size)
        print(json.dumps({"sequence": name, "coverage": metadata["coverage"],
                          "extraction_seconds": metadata["extraction_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
