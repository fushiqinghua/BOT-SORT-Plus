"""Run the selected GMC tracker on the frozen VT-Tiny-MOT detection and ReID caches."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
EXPERIMENT = REPOSITORY / "experiments/vttiny_gmc"


def check_hashes(directory, hashes):
    for name, expected in hashes.items():
        path = directory / name
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"SHA-256 mismatch: {path}")


def main():
    versions = json.loads((EXPERIMENT / "versions.json").read_bytes())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=versions["methods"], default=versions["default"],
                        help="tracking policy (default: %(default)s)")
    parser.add_argument("--settings", type=Path, required=True, help="local frozen-input settings JSON")
    parser.add_argument("--vendor", type=Path, required=True, help="pinned LightGlue vendor directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="new or empty output directory")
    parser.add_argument("--sequences", nargs="+", help="sequence names (default: full split)")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"output directory is already in use: {output}")
    vendor = args.vendor.resolve()
    settings = json.loads(args.settings.read_bytes())
    for field in ("runtime_root", "baseline_root", "features_dir", "images_root"):
        if not Path(settings[field]).is_absolute() or not Path(settings[field]).is_dir():
            parser.error(f"{field} must be an existing absolute directory")
    reference_file = EXPERIMENT / "reference.json"
    model = json.loads(reference_file.read_bytes())["gmc_factorial"]["model"]
    lightglue = vendor / ("LightGlue-" + model["commit"])
    check_hashes(vendor / "torch/hub/checkpoints", model["weights"])
    check_hashes(lightglue / "lightglue", model["source_sha256"])
    os.environ["TORCH_HOME"] = str(vendor / "torch")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(output / "runtime_cfg"))
    for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    paths = [EXPERIMENT / "native_botsort_gmc_0030_fp_v1", EXPERIMENT / "native_botsort_gmc_factorial_v1",
             REPOSITORY, REPOSITORY / "tools", Path(settings["runtime_root"]) / "tracking_runtime",
             vendor / "dependencies", lightglue]
    sys.path[:0] = list(map(str, paths))
    from live_arm import main as track
    return track(args.method, "full", output_dir=output, settings_file=args.settings,
                 repository=REPOSITORY, reference_file=reference_file, sequences=args.sequences)


if __name__ == "__main__":
    raise SystemExit(main())
