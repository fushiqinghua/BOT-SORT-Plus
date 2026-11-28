"""Build offline W R3 identity labels on the locked official-training subset."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys


METHOD = "B_early_moderate"
PARTS = ("fit", "calibration", "holdout")
SPLIT_SHA256 = "43329496f50105b87a9175d312f70b65b1dd3e050307697b20a0656709398c04"
TCO_CAPTURE_PROTOCOL = "ddb7f476ad97d870aae3acf8aa1c071ef190846cd62fcb267f8105014b0aa69a"
DETECTION_MANIFEST_SHA256 = "b69d05eea9c39e09e941d5a890addf0e87c78a2028eacab0accbb9ecb6ba81b1"
EXPECTED_CLIPS_AND_FRAMES = {"fit": (51, 18999), "calibration": (11, 4366), "holdout": (12, 5566)}
BASE = Path(__file__).resolve().parent


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class LabelStream:
    """Add the split, summarize exclusions, and write the exact request keys."""
    def __init__(self, stream, partition):
        self.stream = stream
        self.partition = partition
        self.counts = Counter()

    def write(self, payload):
        row = json.loads(payload)
        row["partition"] = self.partition
        row["schema"] = "wem_r3_training_candidate_label_v1"
        identity = row["identity_label"]
        self.counts["all_edges"] += 1
        self.counts["all_" + identity] += 1
        if row["formal_geometry_allowed"] and row["gmc_guard_allowed"]:
            self.counts["geometry_feasible_edges"] += 1
            self.counts["geometry_feasible_" + identity] += 1
            if not row["hard_ownership_conflict"] and not row["near_duplicate_ownership_risk"]:
                self.counts["unowned_geometry_feasible_edges"] += 1
                self.counts["unowned_geometry_feasible_" + identity] += 1
        self.stream.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n")


class NoOutput:
    """Leave R1b summary checks in place without duplicating auxiliary ledgers."""
    def write(self, _):
        pass


def locked_scope(split_path):
    if sha(split_path) != SPLIT_SHA256:
        raise ValueError("Training split differs from pinned TCO proxy-group split")
    split = read(split_path)
    groups = split["training_sequence_to_source_video"]
    dev = split["development_sequence_to_source_video"]
    if split["status"] != "LOCKED" or set(groups) & set(dev) or set(groups.values()) & set(dev.values()):
        raise ValueError("Training and development groups overlap or split is unlocked")
    if len(groups) != 74 or "DJI_0061_2" in groups:
        raise ValueError("Unexpected training sequence scope")
    seen_seq, seen_group = set(), set()
    for part in PARTS:
        detail = split["split"][part]
        lengths = detail["sequences"]
        if (len(lengths), sum(lengths.values())) != EXPECTED_CLIPS_AND_FRAMES[part]:
            raise ValueError(f"Unexpected sequence/frame counts in {part}")
        group_ids = {groups[seq] for seq in lengths}
        if group_ids != set(detail["source_videos"]) or group_ids & seen_group or set(lengths) & seen_seq:
            raise ValueError(f"Proxy groups overlap or have conflicting membership in {part}")
        seen_seq.update(lengths)
        seen_group.update(group_ids)
    if seen_seq != set(groups):
        raise ValueError("Split failed to account for all listed training clips")
    return split


def run(args):
    sys.path[:0] = [str(args.reuse_r1b), str(args.repo)]
    import label_r1b as r1b
    from experiments.vttiny_gmc.round3_joint import analysis

    split = locked_scope(args.split)
    parts = PARTS if args.partition is None else (args.partition,)
    names = {seq: n for p in parts for seq, n in split["split"][p]["sequences"].items()}
    capture = read(args.capture)
    stage = "smoke" if args.smoke else "full"
    if args.smoke:
        names = {min(split["split"][p]["sequences"]): split["split"][p]["sequences"][min(split["split"][p]["sequences"])]
                 for p in parts}
    if not (capture["status"] == "PASS" and capture["stage"] == stage
            and not capture["gt_open_allowed"] and not capture["online_w_updates"]):
        raise ValueError("Training observer parity not verified or GT reached online")
    if not names.keys() <= set(capture["sequences"]):
        raise ValueError("Observer parity does not cover the requested training partition")
    gt_manifest = read(args.gt_manifest)
    det_manifest = read(args.detection_manifest)
    if sha(args.detection_manifest) != DETECTION_MANIFEST_SHA256:
        raise ValueError("Training detection/ReID manifest changed")
    if gt_manifest["status"] != "PASS" or det_manifest["status"] != "PASS":
        raise ValueError("Required official training GT and cache histories were not verified")
    if not names.keys() <= gt_manifest["output"].keys() or not names.keys() <= det_manifest["per_sequence"].keys():
        raise ValueError("Official training assets do not cover the locked partition")
    old_proto_path = args.trackers.parent / "protocols" / (METHOD + ".json")
    if sha(old_proto_path) != TCO_CAPTURE_PROTOCOL:
        raise ValueError("The historical B training protocol changed")
    old_proto = read(old_proto_path)
    if any(old_proto[k] != det_manifest[k] for k in ("detector_sha256", "rgb_encoder_sha256")):
        raise ValueError("Detector or ReID model differs from the frozen training cache")
    if capture["references"]["B"]["method"] != METHOD:
        raise ValueError("Reference is not B_early_moderate")
    trackeval, audit = analysis.load_trackeval(args.trackeval)
    reviews = r1b.read_reviews(args.background_review)
    sequence_split = {seq: p for p in PARTS for seq in split["split"][p]["sequences"]}
    if any(key[0] not in sequence_split for key in reviews):
        raise ValueError("Independent review contains non-training candidates")
    checked = {}
    for seq, length in names.items():
        observation = capture["references"]["B"]["files"][seq]
        if observation["frames"] != length or old_proto["sequences"][seq] != length:
            raise ValueError("Trajectory scope or snapshot length differs")
        path = Path(observation["snapshot_file"])
        if not path.resolve(strict=True).is_relative_to(args.capture.parent.parent.resolve()):
            raise ValueError("Snapshot is outside the verified isolated training observer")
        trace_hash = sha(path)
        mot = args.trackers / METHOD / "data" / (seq + ".txt")
        mot_hash = sha(mot)
        gt = args.gt / seq / "gt" / "gt.txt"
        gt_hash = sha(gt)
        det = r1b.frozen_path(args.detections, seq)
        det_hash = sha(det)
        cache = det_manifest["per_sequence"][seq]
        if not (trace_hash == observation["observation_proof"]["snapshots_sha256"]
                and mot_hash == observation["mot_sha256"] == old_proto["output_prediction_hashes"][seq]
                and gt_hash == gt_manifest["output"][seq]["gt_sha256"]
                and det_hash == cache["detection_sha256"] and cache["frames"] == length):
            raise ValueError(f"Frozen data contract mismatch for {seq}")
        checked[seq] = path, det, gt, dict(snapshot_sha256=trace_hash, reference_mot_sha256=mot_hash,
                                           official_training_gt_sha256=gt_hash, detection_sha256=det_hash)
    args.output.mkdir(parents=True, exist_ok=False)
    counts_map = {}
    discard = NoOutput()
    for part in parts:
        out = args.output / part
        out.mkdir()
        by_sequence = []
        with gzip.open(out / "candidate_edges.jsonl.gz", "xt", compresslevel=4, encoding="utf-8") as stream:
            labeled = LabelStream(stream, part)
            for seq, length in sorted(split["split"][part]["sequences"].items()):
                if seq not in names:
                    continue
                path, det, gt, proof = checked[seq]
                truth = analysis.evaluate_sequence(trackeval, args.gt, args.trackers, METHOD, seq, length, out)
                ignored = r1b.ignored_boxes(gt)
                cache = r1b.read_json(det)["frames"]
                case = r1b.SequenceLabels("B", seq, split["training_sequence_to_source_video"][seq], reviews)
                for snapshot, detection_frame, gt_frame in zip(r1b.digestable_snapshots(path), cache,
                                                               truth["frames"], strict=True):
                    case.annotate(snapshot, detection_frame, gt_frame, ignored.get(gt_frame["frame"], []),
                                  labeled, discard, discard)
                report = case.summary()
                if report["reference_gt_missed"] != truth["counts"]["FN"]:
                    raise ValueError("Framewise FN denominator differs from official TrackEval")
                by_sequence.append(dict(report, reference_clear=truth["counts"], provenance=proof))
                print(part, seq, "edges", report["candidate_edges"], "opportunities", report["uncovered_opportunity_gt_frames"], flush=True)
        for scope in ("all", "geometry_feasible", "unowned_geometry_feasible"):
            for label in ("same_identity", "other_identity", "confirmed_background", "unknown", "ignored"):
                labeled.counts.setdefault(scope + "_" + label, 0)
        counts_map[part] = dict(proxy_groups=sorted({row["source_video"] for row in by_sequence}), sequences=by_sequence,
                                edge_counts=dict(labeled.counts), aggregate=r1b.group_summary(by_sequence),
                                labels_sha256=sha(out / "candidate_edges.jsonl.gz"))
        write(out / "summary.json", counts_map[part])
    payload = dict(status="TRAINING_DIAGNOSTIC_LABELS_ONLY", reference=METHOD, stage=stage, split_sha256=sha(args.split),
                   proxy_group_provenance_verified=split["source_video_provenance_verified"],
                   grouping_limit=split["protocol_status"],
                   excluded_overlap_groups=split["excluded_indexed_training_sequences"],
                   excluded_unindexed=split["excluded_unindexed_training_sequences"],
                   official_test_gt_used=False, human_background_reviews_supplied=bool(args.background_review),
                   low_iou_disposition="unknown_without_independent_confirmation", trackeval=audit,
                   source={"labeler": sha(Path(__file__)), "r1b_labeler": sha(args.reuse_r1b / "label_r1b.py"),
                           "r1b_rules": sha(args.reuse_r1b / "label_rules.py"), "capture": sha(args.capture),
                           "official_training_gt_manifest": sha(args.gt_manifest), "detection_manifest": sha(args.detection_manifest)},
                   partitions=counts_map)
    write(args.output / "summary.json", payload)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "split", "gt", "gt-manifest", "detections", "detection-manifest",
                 "trackers", "trackeval", "repo", "reuse-r1b", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--partition", choices=PARTS)
    p.add_argument("--smoke", action="store_true", help="Verify data plumbing on one pinned sequence per partition")
    p.add_argument("--background-review", type=Path)
    run(p.parse_args())


if __name__ == "__main__":
    main()
