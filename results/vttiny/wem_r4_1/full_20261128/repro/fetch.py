"""Rebuild the local CLEAR ledgers from a completed W R4.1 run."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys


SEQUENCES = ("DJI_0073_3", "DJI_0327_3")
MODES = ("off", "W2", "W3")
STREAMS = {
    "native_events": ("events", ".jsonl"),
    "weak_events": ("wem_events", ".jsonl"),
    "wem_decisions": ("wem_decisions", ".jsonl.gz"),
}


def checked(path, digest):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"Archived file changed: {path}")
    return data


def load_json(path):
    return json.loads(path.read_bytes())


def write_json(path, value):
    path.write_text(json.dumps(value, separators=(",", ":")) + "\n", encoding="utf-8")


def find_repo():
    return next((parent for parent in Path(__file__).resolve().parents
                 if (parent / "tools/run_wem_r41.py").is_file()), None)


def rebuild(repo, batch, output, gt_root=None, trackeval_root=None):
    if not (repo / "tools/run_wem_r41.py").is_file():
        raise ValueError(f"Expected a W R4.1 checkout at {repo}")
    if output.is_relative_to(batch):
        raise ValueError("Choose an audit output directory outside the completed run")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Audit output must be empty: {output}")
    sys.path.insert(0, str(repo))
    from experiments.vttiny_gmc.round3_joint.analysis import evaluate_sequence, load_trackeval
    from tools.research.wem_r41.assets import METHODS

    complete = load_json(batch / "complete.json")
    evaluated = load_json(batch / "evaluation_complete.json")
    if complete["status"] != "FULL_BENCHMARK_COMPLETE" or evaluated["status"] != "EVALUATION_COMPLETE":
        raise ValueError("W R4.1 evaluation must be complete")
    verified = json.loads(checked(batch / "verified.json", complete["checks"]))
    if verified["status"] != "PASS" or verified["fingerprint"] != complete["fingerprint"]:
        raise ValueError("W R4.1 verification receipt does not match the completed run")
    metrics_path = "metrics/summary.json"
    metrics = json.loads(checked(batch / metrics_path, evaluated["files"][metrics_path]))
    official = {(r["tracker"], r["sequence"]): r for r in metrics["results"]}
    inputs = load_json(batch / "evaluation_inputs.json")
    gt_root = gt_root or Path(inputs["ground_truth"]["root"])
    trackeval_root = trackeval_root or Path(inputs["trackeval"]["root"])
    trackeval, scorer = load_trackeval(trackeval_root)
    tracker_root = batch / "eval_input/trackers"

    for sequence in SEQUENCES:
        checked(gt_root / sequence / "gt/gt.txt", inputs["ground_truth"]["sha256"][sequence])
        shared_gt = None
        for role, method in METHODS.items():
            for mode in MODES:
                arm = f"{role}_{mode}"
                receipt = verified["arms"][arm]["per_sequence"][sequence]
                path = tracker_root / arm / "data" / f"{sequence}.txt"
                checked(path, receipt["mot"])
                result = evaluate_sequence(trackeval, gt_root, tracker_root, arm, sequence,
                                           inputs["sequences"][sequence], output)
                for name in ("FP", "FN", "IDSW"):
                    if result["counts"][name] != official[arm, sequence][name]:
                        raise ValueError(f"{arm}/{sequence}: {name} disagrees with the archived evaluation")
                gt = [frame.pop("gt_boxes") for frame in result["frames"]]
                destination = output / sequence / arm
                destination.mkdir(parents=True, exist_ok=True)
                if shared_gt is None:
                    shared_gt = gt
                    write_json(destination.parent / "gt_frames.json", gt)
                elif gt != shared_gt:
                    raise ValueError(f"{arm}/{sequence}: inconsistent official GT")
                for frame in result["frames"]:
                    del frame["gt_ids"]
                    for row in frame["outputs"].values():
                        del row["gt_ious"]
                write_json(destination / "clear.json", result)
                for name, (folder, suffix) in STREAMS.items():
                    path = batch / arm / folder / method / (sequence + suffix)
                    data = checked(path, receipt[name])
                    (destination / f"{name}.jsonl").write_bytes(gzip.decompress(data) if suffix.endswith(".gz") else data)

    write_json(output / "receipt.json", {"complete": complete, "scorer": scorer})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True, type=Path, help="Completed rollout's full/ directory")
    parser.add_argument("--output", required=True, type=Path, help="Empty private directory for local ledgers")
    parser.add_argument("--repo-root", type=Path, default=find_repo(), help="Override the W R4.1 checkout")
    parser.add_argument("--gt-root", type=Path, help="Override the location of unchanged official GT files")
    parser.add_argument("--trackeval-root", type=Path, help="Override the location of the pinned TrackEval checkout")
    args = parser.parse_args()
    if args.repo_root is None:
        parser.error("--repo-root is required when this script is used outside the checkout")
    rebuild(args.repo_root.resolve(), args.batch.resolve(), args.output.resolve(), args.gt_root, args.trackeval_root)
    print(f"Verified CLEAR ledgers for {len(SEQUENCES)} sequences and {len(MODES) * 2} arms per sequence.")


if __name__ == "__main__":
    main()
