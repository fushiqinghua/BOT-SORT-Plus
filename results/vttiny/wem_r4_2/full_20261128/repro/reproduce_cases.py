"""Rebuild the three published W R4.2 case ledgers from a completed local run."""

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys


COMMIT = "eded63f9079734560966d3df59ba5c938c3e9ac4"
CASES = {
    "lower_low_DJI_0339_2.json": dict(seq="DJI_0339_2", start=47, end=53, gt_ids=[], track_ids=[2],
                                      arms=("BG_W0", "BG_lower_low")),
    "relaxed_anchor_DJI_0339_2.json": dict(seq="DJI_0339_2", start=95, end=99, gt_ids=[], track_ids=[2],
                                           arms=("BG_W0", "BG_W0_relaxed_anchor")),
    "state_feedback_DJI_0327_3.json": dict(seq="DJI_0327_3", start=227, end=237, gt_ids=[16], track_ids=[55],
                                           arms=("BG_off", "BG_W0", "BG_W0_emit_only")),
}
WEAK_FIELDS = {"event", "det_index", "raw_score", "native_iou", "motion", "state_committed", "output_only",
               "decision_policy", "previous_regular_observations", "history_frames", "anchor_box_mismatch",
               "anchor_not_ready", "anchor_prediction_mismatch", "anchor_support_bypassed"}


def load_json(path):
    return json.loads(path.read_bytes())


def check(path, expected):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"File differs from its completed audit receipt: {path}")


def window_events(path, start, end):
    with path.open(encoding="utf-8") as stream:
        for row in map(json.loads, stream):
            if start <= row.get("frame", -1) <= end:
                yield row


def weak_fields(event):
    row = {key: value for key, value in event.items() if key in WEAK_FIELDS}
    if event["event"] == "committed":
        row.setdefault("state_committed", True)
    return row


def frame_ledger(frame, gt_ids, track_ids, published, event_stages, weak_rows):
    time = frame["frame"]
    coverage = {gid: dict(present=gid in frame["gt_ids"], matched_track=frame["coverage"].get(gid))
                for gid in gt_ids}
    selected = set(track_ids)
    selected.update(row["matched_track"] for row in coverage.values() if row["matched_track"] is not None)
    tracks = []
    for tid in sorted(selected):
        output = frame["outputs"].get(tid)
        key = time, tid
        events = weak_rows.get(key, [])
        if output is None:
            tracks.append(dict(track_id=tid, status="no_output", weak_events=events))
            continue
        if key not in published:
            raise ValueError(f"Missing published detection index: frame {time}, track {tid}")
        index = published[key]
        stages = event_stages.get((time, tid, None if index == -1 else index), [])
        committed = [row for row in events if row["event"] == "committed" and row["det_index"] == index]
        if committed:
            stages = ["weak_measurement" if row["state_committed"] else "weak_emit_only" for row in committed]
        tracks.append(dict(track_id=tid, status=output["status"], matched_gt=output["matched_gt"],
                           nearest_gt=output["nearest_gt"], max_iou=output["max_iou"],
                           matched_iou=output["gt_ious"].get(output["matched_gt"]),
                           requested_gt_ious={gid: output["gt_ious"].get(gid) for gid in gt_ids},
                           det_index=index, stage=stages[0] if len(stages) == 1 else None,
                           stage_events=stages, weak_events=events))
    return dict(frame=time, clear_counts=frame["counts"], coverage=coverage, tracks=tracks)


def arm_ledger(batch, arm, receipt, evaluate, official, files, seq, start, end, gt_ids, track_ids):
    folder = batch / arm
    method = receipt["method"]
    native = folder / "events" / method / (seq + ".jsonl")
    weak = folder / "wem_events" / method / (seq + ".jsonl")
    state = folder / "states" / method / (seq + ".sha256")
    decisions = folder / "decisions" / method / (seq + ".jsonl")
    mot = batch / "eval_input" / "trackers" / arm / "data" / (seq + ".txt")
    details = receipt["per_sequence"][seq]
    for path, field in ((mot, "mot"), (native, "native_events"), (weak, "weak_events"), (state, "states")):
        check(path, details[field])
    protocol_file = f"eval_input/protocols/{arm}.json"
    check(batch / protocol_file, files[protocol_file])
    protocol = load_json(batch / protocol_file)
    check(decisions, protocol["joint_batch"]["decision_manifests"][seq]["data_sha256"])

    result = evaluate(arm, seq)
    if any(result["counts"][key] != official[key] for key in ("FP", "FN", "IDSW")):
        raise ValueError(f"CLEAR ledger differs from completed official metrics: {arm}/{seq}")
    event_stages = defaultdict(list)
    for row in window_events(native, start, end):
        stage = {"match": row.get("stage"), "recovered": "B_recovery", "native_birth": "native_birth",
                 "motion_output": "motion_output"}.get(row["event"])
        if stage is not None:
            event_stages[row["frame"], row["track_id"], row.get("det_index")].append(stage)
    weak_rows = defaultdict(list)
    for row in window_events(weak, start, end):
        weak_rows[row["frame"], row["track_id"]].append(weak_fields(row))
    published = {(row["frame"], item["track_id"]): item["det_index"]
                 for row in window_events(decisions, start, end) if row["event"] == "output_index" for item in row["rows"]}
    counters = details["counts"]
    return dict(sequence_counts=result["counts"], official_confirmed=True,
                verified_mot_sha256=details["mot"], verified_state_chain_sha256=details["states"],
                verified_native_event_sha256=details["native_events"],
                state_chain_basis=protocol.get("wem_r42", {}).get("state_chain_output_basis", "published_output"),
                weak_outputs_emitted=counters.get("weak_outputs_emitted", counters.get("committed", 0)),
                weak_state_updates=counters.get("weak_state_updates", counters.get("committed", 0)),
                frames=[frame_ledger(frame, gt_ids, track_ids, published, event_stages, weak_rows)
                        for frame in result["frames"][start - 1:end]])


def reproduce(batch, repo, gt_root=None, trackeval_root=None, source_receipt=None):
    sys.dont_write_bytecode = True
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    sys.path.insert(0, str(repo))
    from experiments.vttiny_gmc.round3_joint.analysis import evaluate_sequence, load_trackeval

    complete = load_json(batch / "complete.json")
    evaluation = load_json(batch / "evaluation_complete.json")
    source = load_json(source_receipt or batch.parent.parent / "source_receipt.json")
    if (source["github_commit"] != COMMIT or complete["status"] != "FULL_BENCHMARK_COMPLETE"
            or evaluation["status"] != "EVALUATION_COMPLETE"):
        raise ValueError(f"Expected a completed W R4.2 run at {COMMIT}")
    files = evaluation["files"]
    check(batch / "verified.json", complete["checks"])
    check(batch / "metrics/summary.json", files["metrics/summary.json"])
    check(batch / "metrics/evaluation_audit.json", files["metrics/evaluation_audit.json"])
    verified = load_json(batch / "verified.json")
    access = load_json(batch / "metrics/evaluation_audit.json")
    lengths = access["validation"]["sequences"]
    if (verified["status"] != "PASS" or verified["fingerprint"] != complete["fingerprint"]
            or set(verified["scope"]) != set(lengths) or len(lengths) != 30 or sum(lengths.values()) != 12590
            or access["summary_sha256"] != files["metrics/summary.json"]):
        raise ValueError("Mismatched W R4.2 completion or frozen scope receipts")
    gt_root = gt_root or Path(access["ground_truth"]["root"])
    trackeval_root = trackeval_root or Path(access["evaluator"]["root"])
    tracker_root = batch / "eval_input" / "trackers"
    trackeval, audit = load_trackeval(trackeval_root)
    if any(audit[field] != access["evaluator"][field] for field in ("commit", "manifest_sha256")):
        raise ValueError("TrackEval source identity differs from the completed evaluation")
    official = {(row["tracker"], row["sequence"]): row for row in load_json(batch / "metrics/summary.json")["results"]}

    def evaluate(arm, seq):
        return evaluate_sequence(trackeval, gt_root, tracker_root, arm, seq, lengths[seq], batch / "attribution")

    cases = {}
    for name, spec in CASES.items():
        seq, start, end = spec["seq"], spec["start"], spec["end"]
        if end > lengths[seq]:
            raise ValueError(f"Frame window exceeds archived sequence length for {seq}")
        check(gt_root / seq / "gt" / "gt.txt", access["ground_truth"]["sha256"][seq])
        gt_ids, track_ids = spec["gt_ids"], spec["track_ids"]
        arms = {arm: arm_ledger(batch, arm, verified["arms"][arm], evaluate, official[arm, seq],
                                files, seq, start, end, gt_ids, track_ids) for arm in spec["arms"]}
        cases[name] = dict(sequence=seq, start=start, end=end, gt_ids=gt_ids, track_ids=track_ids,
                           commit=COMMIT, evaluator=audit["commit"], arms=arms)
    return cases


def find_repo():
    probe = Path("tools/run_wem_r42.py")
    return next((parent for parent in Path(__file__).resolve().parents if (parent / probe).is_file()), None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True, type=Path, help="Completed R4.2 run/full directory")
    parser.add_argument("--output", required=True, type=Path, help="Directory for three derived case JSON files")
    parser.add_argument("--repo-root", type=Path, default=find_repo(), help="Repository root if the script was moved")
    parser.add_argument("--gt-root", type=Path, help="Relocated official GT root with identical GT bytes")
    parser.add_argument("--trackeval-root", type=Path, help="Relocated checkout of the pinned TrackEval")
    parser.add_argument("--source-receipt", type=Path, help="Source receipt when run/full was relocated")
    args = parser.parse_args()
    if args.repo_root is None or not (args.repo_root / "tools/run_wem_r42.py").is_file():
        parser.error("--repo-root must identify the R4.2 repository checkout")
    batch, output = args.batch.resolve(), args.output.resolve()
    if output.is_relative_to(batch):
        parser.error("Choose an output directory outside the original completed run/full")
    for name, data in reproduce(batch, args.repo_root.resolve(), args.gt_root, args.trackeval_root,
                                args.source_receipt).items():
        text = json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_text(encoding="utf-8") != text:
                raise FileExistsError(f"Existing case evidence differs: {path}")
        else:
            path.write_text(text, encoding="utf-8", newline="\n")
        print(path)


if __name__ == "__main__":
    main()
