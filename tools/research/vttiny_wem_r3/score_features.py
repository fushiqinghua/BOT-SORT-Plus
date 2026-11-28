"""Extract S0 RGB features and score geometry, history and local identity competition."""

import argparse
from collections import Counter, defaultdict, deque
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


METHOD = "B_early_moderate"
RGB_SHA = "27f110f2fa8ae3c619941c126699486771a42d252b4e675d1c077bb91e0bd2d1"
SPLIT_SHA = "43329496f50105b87a9175d312f70b65b1dd3e050307697b20a0656709398c04"
ASSET_SHA = "b69d05eea9c39e09e941d5a890addf0e87c78a2028eacab0accbb9ecb6ba81b1"
REFERENCE_SHA = "ddb7f476ad97d870aae3acf8aa1c071ef190846cd62fcb267f8105014b0aa69a"
PARTITIONS = ("fit", "calibration", "holdout")


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def overlap(first, second):
    if np.any(first[2:] <= first[:2]) or np.any(second[2:] <= second[:2]):
        return 0.0
    delta = np.maximum(0.0, np.minimum(first[2:], second[2:]) - np.maximum(first[:2], second[:2]))
    inter = float(np.prod(delta))
    return inter / (float(np.prod(first[2:]-first[:2]) + np.prod(second[2:]-second[:2])) - inter)


def center(box):
    return (box[:2] + box[2:]) / 2.


def row_vector(cache, item):
    offsets, valid, vectors = (cache[k] for k in ("offsets", "rgb_valid", "rgb"))
    time, row = item["frame"], item["det_index"]
    start, end = (int(x) for x in offsets[time-1:time+1])
    if not 0 <= row < end - start:
        raise ValueError("History is outside a frozen detector row")
    idx = start + row
    if not valid[idx]:
        return None
    vector = np.asarray(vectors[idx], dtype=np.float64)
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Valid history RGB embedding is corrupt")
    return vector / norm


def trajectory_context(history, frame, warps, features):
    observations = [item for item in history if 1 <= frame - item["frame"] <= 5]
    if len({item["frame"] for item in observations}) != len(observations):
        raise ValueError("Repeated history frame")
    vectors = [vector for item in observations if (vector := row_vector(features, item)) is not None]
    trend = None
    if len(observations) >= 2:
        points, times = [], []
        for item in observations:
            t = item["frame"]
            pos = center(np.asarray(item["raw_xyxy"], dtype=np.float64))
            for f in range(t + 1, frame + 1):
                warp = warps[f]
                pos = warp[:, :2] @ pos + warp[:, 2]
            points.append(pos)
            times.append(t - frame)
        trend = np.linalg.lstsq(np.column_stack((np.ones(len(times)), times)), np.stack(points), rcond=None)[0][0]
    return list(vectors), trend, len(observations)


def evidence(candidate, s0_vec, geometry_iou, local_box, historic_vectors, trend):
    cosine = None
    motion = None
    if s0_vec is not None and historic_vectors:
        cosine = float(np.clip(np.median(np.asarray(historic_vectors) @ s0_vec), -1.0, 1.0))
    if trend is not None:
        scale = max(1.0, float(np.linalg.norm(local_box[2:] - local_box[:2])))
        delta = float(np.linalg.norm(center(np.asarray(candidate["raw_xyxy"])) - trend))
        motion = float(np.clip(1.0 - delta / scale, 0.0, 1.0))
    score = float(np.clip((candidate["raw_score"] - .05) / .05, 0.0, 1.0))
    utility = (geometry_iou + score + (cosine + 1.0) / 2. + motion) / 4. if (
        cosine is not None and motion is not None) else None
    return dict(native_iou=geometry_iou, normalized_score=score, cosine=cosine, motion=motion, history_score=utility)


def groups_and_margins(proposals):
    by_track = defaultdict(list)
    for row in proposals:
        by_track[row["track_id"]].append(row)
    for candidates in by_track.values():
        order = sorted(candidates, key=lambda r: (-r["native_iou"],-r["raw_score"],r["det_index"]))
        boxes = list(dict.fromkeys(tuple(r["raw_box_xyxy"]) for r in order))
        for row in candidates:
            alternatives = [r for r in candidates if r["raw_box_xyxy"] != row["raw_box_xyxy"]]
            values = [r["history_score"] for r in alternatives if r["history_score"] is not None]
            score = row["history_score"]
            row["row_unknown"] = any(r["history_score"] is None for r in alternatives)
            row["row_margin"] = score - max(values) if score is not None and values else None
            row["geometry_group_rank"] = boxes.index(tuple(row["raw_box_xyxy"])) + 1


def frame_features(shot, cache_frame, features, encoder, warps, image):
    from botsort_plus.feature_export import _encode_batches, _valid_rgb_crop

    frame = shot["frame"]
    if shot["image"] != cache_frame["image"] or frame != cache_frame["frame"]:
        raise ValueError("Snapshot RGB image differs from frozen training cache")
    raw = np.asarray(cache_frame["detections"], dtype=np.float32).reshape(-1, 6).astype(np.float64)
    dets = shot["detections"]
    if len(dets) != len(raw):
        raise ValueError("Frozen detector row length changed")
    candidates = [edge for edge in shot["s0_edges"] if (
        edge["native_low_feasible"] and edge["gmc_guard_feasible"])]
    owned = set(shot["consumed"]) | set(shot["reserved"])
    useful = {}
    for row in sorted({edge["det_index"] for edge in candidates}):
        item = dets[row]
        if item["band"] != "S0" or item["raw_xyxy"] != raw[row, :4].tolist() or item["raw_score"] != raw[row, 4]:
            raise ValueError("S0 candidate is not the corresponding frozen raw detector row")
        box = raw[row, :4]
        if row not in owned and all(overlap(box, raw[idx,:4]) < .90 for idx in owned):
            useful[row] = item
    if not useful:
        return [], [], None
    indices = list(useful)
    boxes = raw[indices, :4].astype(np.float32)
    can_crop = _valid_rgb_crop(boxes, image)
    vectors = np.zeros((len(indices), 1280), np.float32)
    valid = np.zeros(len(indices), bool)
    if np.any(can_crop):
        vectors[can_crop], valid[can_crop], _ = _encode_batches(encoder, boxes[can_crop], image, "rgb", 64)
    weak = {idx: (np.asarray(vec, dtype=np.float64) if ok else None) for idx, vec, ok in zip(indices,vectors,valid)}
    embeddings = [(frame, idx, vec, bool(ok)) for idx, vec, ok in zip(indices, vectors, valid)]
    start = {item["track_id"]:item for item in shot["start_tracks"] if item["collection"] != "removed_stracks"}
    prior = {item["track_id"]: np.asarray(item["selected_xyxy"]) for item in shot["post_gmc"]}
    accepted = {item["track_id"]:item for item in shot["accepted"]}
    current = {k: np.asarray(v["raw_xyxy"]) for k, v in accepted.items()}
    identities = {tid for tid, item in start.items() if item["confirmed"] and
                  (item["state"] == 1 or frame - item["last_frame"] <= 5)} | accepted.keys()
    history = {tid: trajectory_context(start.get(tid, {}).get("raw_history", ()), frame, warps, features)
               for tid in identities}
    contenders = {}
    for idx, item in useful.items():
        box = raw[idx,:4]
        rivals = []
        for tid in sorted(identities):
            geom = overlap(box, prior[tid]) if tid in prior else 0.
            curr = overlap(box, current[tid]) if tid in current else 0.
            if max(geom, curr) < .5:
                continue
            vectors_hist, trend, past = history[tid]
            # A current accepted observation may defend its own identity, including a new birth.
            used_obs = accepted.get(tid)
            if used_obs is not None:
                seen_vector = row_vector(features, used_obs)
                vectors_hist = [*vectors_hist, seen_vector] if seen_vector is not None else vectors_hist
                if trend is None:
                    trend = center(current[tid])
            local = current[tid] if curr > geom else prior[tid]
            status = "start_tracked_or_recent" if used_obs is None else used_obs["stage"]
            rival = evidence(item, weak[idx], max(geom,curr), local, vectors_hist, trend)
            rivals.append(dict(track_id=tid, source=status, history_observations=past, **rival))
        contenders[idx] = rivals
    proposals = []
    for edge in candidates:
        tid, idx = edge["track_id"], edge["det_index"]
        if idx not in useful:
            continue
        entry = useful[idx]
        vecs, trend, count = history[tid]
        parts = evidence(entry, weak[idx], edge["selected_iou"], prior[tid], vecs if len(vecs) >= 2 else [], trend)
        score = parts["history_score"]
        others = [r for r in contenders[idx] if r["track_id"] != tid]
        compared = [r["history_score"] for r in others if r["history_score"] is not None]
        x1,y1,x2,y2 = np.rint(raw[idx, :4]).astype(int)
        h,w = image.shape[:2]
        side = [int(max(0,min(w,x2)-max(0,x1))),int(max(0,min(h,y2)-max(0,y1)))]
        proposals.append(dict(reference=METHOD, sequence=shot["sequence"], frame=frame, track_id=tid,
                              det_index=idx, raw_box_xyxy=entry["raw_xyxy"], raw_score=entry["raw_score"],
                              crop_wh=side, rgb_weak_valid=weak[idx] is not None,
                              valid_history_rgb=len(vecs), previous_regular_observations=count,
                              column_unknown=any(r["history_score"] is None for r in others),
                              column_margin=score-max(compared) if score is not None and compared else None,
                              challengers=others, **parts))
    groups_and_margins(proposals)
    return proposals, embeddings, sha(Path(shot["image"]))


def sequence_proof(proof, seq):
    return proof["references"]["B"]["files"][seq]["observation_proof"]["snapshots_sha256"]


def extract(args):
    sys.path.insert(0, str(args.repo.resolve(strict=True)))
    from botsort_plus.feature_export import load_rgb_encoder
    from botsort_plus.io import FeatureCache, forbid_ground_truth, read_detection_cache
    import cv2
    import torch

    sys.addaudithook(forbid_ground_truth)
    if (sha(args.split)!=SPLIT_SHA or sha(args.assets/"verified.json")!=ASSET_SHA
            or sha(args.rgb_weights)!=RGB_SHA or sha(args.reference_protocol)!=REFERENCE_SHA):
        raise ValueError("Split, frozen assets, or encoder fingerprint changed")
    split, assets, proof = read(args.split), read(args.assets/"verified.json"), read(args.parity)
    reference_pixels = read(args.reference_protocol)["decoded_image_hashes"]
    seqs = {seq: (part, n) for part in PARTITIONS for seq,n in split["split"][part]["sequences"].items()}
    names=set(proof["sequences"])
    if (proof["status"]!="PASS" or proof["stage"]!=args.scope
            or (names!=set(seqs) if args.scope=="full" else len(names)!=3 or not names<=set(seqs))):
        raise ValueError("Training reference parity differs from the intended scope")
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    encoder, metadata = load_rgb_encoder(args.rgb_weights, args.device, args.runtime_root)
    args.output.mkdir(exist_ok=True, parents=True)
    ledger = dict(status="COMPLETE", scope=args.scope, reference=METHOD, encoder=metadata,
                  reference_protocol_sha256=REFERENCE_SHA, feature_implementation_sha256=sha(Path(__file__)),
                  input_split_sha256=sha(args.split),
                  runtime_device=args.device, parity_sha256=sha(args.parity), sequences={})
    for seq,(part,frames) in sorted(seqs.items()):
        if seq not in names:
            continue
        snapshot = args.snapshots / f"{seq}.jsonl.gz"
        source = args.assets / "detections" / f"{seq}.json.gz"
        feature_path = args.assets / "features" / f"{seq}.npz"
        expected = assets["per_sequence"][seq]
        if (sha(snapshot)!=sequence_proof(proof,seq) or sha(source)!=expected["detection_sha256"]
                or sha(feature_path)!=expected["rgb_features_sha256"]
                or sha(feature_path.with_suffix(".json"))!=expected["rgb_metadata_sha256"]):
            raise ValueError(f"Frozen features/capture inputs changed: {seq}")
        rgb_path = args.output/f"{seq}.npz"
        scores = args.output/f"{seq}.jsonl.gz"
        meta = args.output/f"{seq}.json"
        if meta.is_file():
            prior = read(meta)
            if (prior["scores_sha256"]!=sha(scores) or prior["weak_rgb_sha256"]!=sha(rgb_path)
                    or prior["snapshot"]!=sha(snapshot) or prior.get("decoded_rgb_sha256")!=reference_pixels[seq]):
                raise ValueError(f"Invalid existing feature outputs: {seq}")
            ledger["sequences"][seq] = prior
            continue
        if rgb_path.exists() or scores.exists():
            raise ValueError(f"Interrupted feature output: {seq}")
        cache = read_detection_cache(source)
        history = FeatureCache(feature_path, cache, expected_rgb_sha=RGB_SHA).arrays
        warps, images, weak_rgb = {}, {}, []
        counts = Counter(dict.fromkeys(("frames", "edges", "weak_crops", "rgb_valid"), 0))
        pixel_digest = hashlib.sha256()
        with gzip.open(snapshot, "rt", encoding="utf-8") as frames_in, gzip.open(
                scores.with_suffix(".partial"), "wt", encoding="utf-8", compresslevel=2) as out:
            for line, det_frame in zip(frames_in,cache["frames"],strict=True):
                shot=json.loads(line)
                frame=shot["frame"]
                if frame!=counts["frames"]+1 or shot["sequence"]!=seq or shot["reference"]!=METHOD:
                    raise ValueError("Snapshot time or reference mismatch")
                if shot["image"] != det_frame["image"] or frame != det_frame["frame"]:
                    raise ValueError("Snapshot RGB path or frame differs from the frozen detection cache")
                image = cv2.imread(str(shot["image"]), cv2.IMREAD_COLOR)
                if image is None or list(image.shape) != shot["image_shape"]:
                    raise ValueError("RGB crop source image cannot be reproduced")
                pixels = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
                pixel_record = {"frame":frame,"shape":list(image.shape),"dtype":str(image.dtype),"sha256":pixels}
                pixel_digest.update((json.dumps(pixel_record,sort_keys=True,separators=(",", ":"))+"\n").encode("utf-8"))
                counts["frames"] += 1
                warps[frame]=np.asarray(shot["gmc"]["selected"],dtype=np.float64)
                if not shot["s0_edges"]:
                    continue
                rows, vectors, image_sha = frame_features(shot,det_frame,history,encoder,warps,image)
                if image_sha is not None:
                    images[str(frame)]=image_sha
                weak_rgb.extend(vectors)
                for row in rows:
                    out.write(json.dumps(row,allow_nan=False,separators=(",",":"))+"\n")
                counts.update(edges=len(rows), weak_crops=len(vectors), rgb_valid=sum(1 for v in vectors if v[-1]))
        if counts["frames"]!=frames or pixel_digest.hexdigest()!=reference_pixels[seq]:
            raise ValueError(f"Incomplete feature coverage or decoded RGB differs from frozen replay: {seq}")
        np.savez_compressed(rgb_path, keys=np.asarray([(f,d) for f,d,_,_ in weak_rgb],np.int64).reshape(-1,2),
                            embeddings=np.asarray([v for _,_,v,_ in weak_rgb],np.float32).reshape(-1,1280),
                            valid=np.asarray([ok for *_,ok in weak_rgb],bool))
        scores.with_suffix(".partial").replace(scores)
        prior=dict(sequence=seq, partition=part, counts=counts, snapshot=sha(snapshot), images=images,
                   decoded_rgb_sha256=pixel_digest.hexdigest(),
                   scores_sha256=sha(scores), weak_rgb_sha256=sha(rgb_path))
        meta.write_text(json.dumps(prior,indent=2)+"\n",encoding="utf-8")
        ledger["sequences"][seq]=prior
        print(seq, part, dict(counts), flush=True)
    (args.output/"manifest.json").write_text(json.dumps(ledger,indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("repo","split","assets","parity","snapshots","reference-protocol","rgb-weights","runtime-root","output"):
        parser.add_argument("--"+name,required=True,type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--scope", choices=("smoke","full"), default="full")
    extract(parser.parse_args())
