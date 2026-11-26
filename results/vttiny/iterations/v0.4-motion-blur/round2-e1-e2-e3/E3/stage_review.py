"""Sample E3-B visual evidence with no automatic background labels."""
from __future__ import annotations

from collections import defaultdict
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

from audit_native_fp import BASE, csv_output, fingerprint, read_json


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def by_key(row):
    return row["sequence"], int(row["frame"]), int(row["track_id"])


def select_cases(detail, tracks, features):
    grouped = defaultdict(list)
    for row in detail:
        grouped[row["sequence"], int(row["track_id"])].append(row)
    candidates = sorted((x for x in tracks if x["lifetime_unannotated_candidate_010"] == "True"),
                        key=lambda x: (-int(x["clear_fp"]), x["sequence"], int(x["track_id"])))
    lookup = {(p["sequence"], int(p["track_id"])): p for p in candidates}
    must = [key for key in (("DJI_0339_2", 2), ("DJI_0339_2", 6)) if key in lookup]
    selected = [lookup[key] for key in must]
    groups = {row["original_video"] for row in selected}
    for row in candidates:
        if row["original_video"] not in groups and len(selected) < 10:
            selected.append(row)
            groups.add(row["original_video"])
    if len(selected) < 10:
        keys = {(r["sequence"], int(r["track_id"])) for r in selected}
        selected.extend([r for r in candidates if (r["sequence"], int(r["track_id"])) not in keys][:10-len(selected)])

    samples = []
    for track in selected:
        key = track["sequence"], int(track["track_id"])
        eligible = [r for r in grouped[key] if r["offline_category"] == "persistent_no_gt_support"]
        if not eligible:
            continue
        start = len(eligible) // 4
        stop = max(start + 1, len(eligible) - start)
        mid = max(eligible[start:stop], key=lambda r: float(r["output_score"]))
        samples.append(("persistent_no_gt_support", mid))

    controls = defaultdict(list)
    for key, rows in grouped.items():
        for row in rows:
            evidence = features[by_key(row)]
            speed = evidence["gmc_speed_diag_per_frame"]
            if (row["matched_gt"] and speed and float(speed) <= .12
                    and float(evidence["output_short_side"]) <= 12):
                controls[key[0]].append(row)
    taken = set()
    for _, row in list(samples):
        f = features[by_key(row)]
        allowed = [c for c in controls[row["sequence"]] if (c["sequence"], c["track_id"]) not in taken]
        if not allowed:
            continue
        def closeness(control):
            g = features[by_key(control)]
            return abs(math.log(float(f["output_short_side"])/float(g["output_short_side"]))) + abs(
                float(f["output_score"]) - float(g["output_score"]))
        control = min(allowed, key=closeness)
        taken.add((control["sequence"], control["track_id"]))
        samples.append(("CLEAR_matched_small_near_static", control))
        if len(taken) >= 8:
            break

    soft = sorted((r for r in detail if r["offline_category"] == "same_target_localization_below_threshold"),
                  key=lambda r: (abs(float(r["output_iou_to_prior_gt"])-.5), by_key(r)))
    later = set()
    for row in soft:
        if row["sequence"] not in later:
            later.add(row["sequence"])
            samples.append(("same_target_near_IoU_boundary", row))
        if len(later) == 5:
            break
    return grouped, samples


def bounded_box(box, shape):
    height, width = shape[:2]
    x1 = max(0, min(width - 1, math.floor(box[0])))
    y1 = max(0, min(height - 1, math.floor(box[1])))
    x2 = min(width, max(x1 + 1, math.ceil(box[2])))
    y2 = min(height, max(y1 + 1, math.ceil(box[3])))
    return x1, y1, x2, y2


def recent_frames(frame, length):
    return sorted({max(1, frame - 1), frame, min(length, frame + 1)})


def write_view(source_image, box, official, frame, frame_roles, output, Image, ImageDraw):
    x1, y1, x2, y2 = bounded_box(box, (source_image.height, source_image.width))
    native = source_image.crop((x1, y1, x2, y2))
    native.save(output / f"{frame:04}_crop_native.png")
    origin = source_image.copy()
    origin.save(output / f"{frame:04}_context_native.png")
    draw = ImageDraw.Draw(origin)
    line = (255, 95, 27) if frame_roles != "adjacent_position_only" else (255, 215, 64)
    draw.rectangle((x1-1, y1-1, x2+1, y2+1), outline=line, width=1)
    for truth in official:
        a, b, c, d = bounded_box(truth["xyxy"], (origin.height, origin.width))
        if abs((a+c-x1-x2)/2) <= 60 and abs((b+d-y1-y2)/2) <= 60:
            draw.rectangle((a, b, c, d), outline=(32, 231, 117), width=1)
    origin.save(output / f"{frame:04}_context_annotated.png")

    risk_margin = 23
    focus = (max(0, x1-risk_margin), max(0, y1-risk_margin),
             min(origin.width, x2+risk_margin), min(origin.height, y2+risk_margin))
    local = source_image.crop(focus)
    zoom = max(1, min(5, 230 // max(local.size)))
    enlarged = local.resize((local.width*zoom, local.height*zoom), Image.Resampling.NEAREST)
    context = origin.crop(focus).resize(enlarged.size, Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (640, 256), (14, 24, 36))
    canvas.paste(enlarged, (8, 20))
    canvas.paste(context, (250, 20))
    miniature = origin.resize((152, 122), Image.Resampling.BOX)
    canvas.paste(miniature, (486, 42))
    out = ImageDraw.Draw(canvas)
    out.text((8, 4), f"{frame}: local RGB", fill="white")
    out.text((250, 4), f"{frame}: orange track | green GT", fill="white")
    out.text((488, 173), frame_roles, fill="white")
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("repo", "frozen_root", "input", "output"):
        parser.add_argument("--" + field.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo))
    from botsort_plus.io import detection_files, detection_fingerprint, read_detection_cache, resolve_image, sequence_name
    from botsort_plus.diagnostics.common import iter_trace, load_gt, sha256
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw

    a = read_json(args.input / "E3_A_evaluation_audit.json")
    assert a["status"] == "PASS"
    for filename, expected in a["outputs_sha256"].items():
        assert fingerprint(args.input / filename) == expected, filename
    protocol = args.frozen_root / "observed/protocols" / f"{BASE}.json"
    assert fingerprint(protocol) == a["run_protocol_sha256"]
    p = read_json(protocol)
    analysis = read_json(args.frozen_root / "analysis/summary.json")
    gt_root = Path(analysis["ground_truth"]["root"])
    baseline = Path(p["input_roots"]["baseline"])
    images = Path(p["input_roots"]["images"])
    all_boxes = read_csv(args.input / "E3_all_rows_CLEAR.csv")
    profiles = read_csv(args.input / "E3_trajectory_profiles.csv")
    evidence = {by_key(r): r for r in read_csv(args.input / "E3_causal_signal_rows.csv")}
    tracks, selected = select_cases(all_boxes, profiles, evidence)
    paths = {sequence_name(i): i for i in detection_files(baseline / "detections")}
    frames_required = defaultdict(set)
    for _, row in selected:
        sequence, frame, _ = by_key(row)
        frames_required[sequence].update(recent_frames(frame, p["sequences"][sequence]))

    args.output.mkdir(parents=True)
    seq_data = {}
    for seq, frames in sorted(frames_required.items()):
        trace = args.frozen_root / "observed" / p["diagnostics"]["traces"][seq]["path"]
        image_sha = {}
        for state in iter_trace(trace, seq, p["sequences"][seq]):
            if state["frame"] in frames:
                image_sha[state["frame"]] = state["image_sha256"]
        assert image_sha.keys() == frames, seq
        cache = read_detection_cache(paths[seq])
        assert detection_fingerprint(cache) == p["detection_fingerprints"][seq], seq
        gt_path = gt_root / seq / "gt" / "gt.txt"
        assert sha256(gt_path) == a["input_sha256"][seq]["gt"], seq
        gt, _ = load_gt(gt_path, p["sequences"][seq])
        seq_data[seq] = (image_sha, cache, gt)
    rows = []
    for number, (tier, row) in enumerate(selected, 1):
        seq, center_frame, track = by_key(row)
        hash_rows, cache, gt = seq_data[seq]
        case_id = f"E3_{number:03}"
        target = args.output / case_id
        target.mkdir()
        with_images = []
        frame_roles = {}
        same_track = {int(r["frame"]): r for r in tracks[seq, track]}
        image_info = {}
        saved = {}
        for f in recent_frames(center_frame, p["sequences"][seq]):
            det = cache["frames"][f - 1]
            path = resolve_image(det["image"], images)
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            assert bgr.shape == (512, 640, 3), path
            source_hash = hashlib.sha256(np.ascontiguousarray(bgr).tobytes()).hexdigest()
            assert source_hash == hash_rows[f], (seq, f, "decoded image mismatch")
            image_info[f] = {"input_image": det["image"], "decoded_bgr_sha256": source_hash}
            frow = same_track.get(f, row)
            box = json.loads(frow["output_box_xyxy"])
            role = "center" if f == center_frame else ("adjacent_same_track" if f in same_track else "adjacent_position_only")
            frame_roles[f] = role
            image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            panel = write_view(image, box, gt[f], f, role, target, Image, ImageDraw)
            with_images.append(panel)
        title = f"{case_id}  {seq}  track {track}  frame {center_frame}  {tier}"
        card = Image.new("RGB", (640, 26+256*len(with_images)), (14, 24, 36))
        for panel_index, panel in enumerate(with_images):
            card.paste(panel, (0, 26+256*panel_index))
        ImageDraw.Draw(card).text((8, 7), title, fill="white")
        card.save(target / "review.png")
        saved = {x.name: fingerprint(x) for x in sorted(target.glob("*.png"))}
        with (target / "image_provenance.json").open("x", encoding="utf-8") as file:
            json.dump({"input": image_info, "outputs_sha256": saved, "frame_roles": frame_roles}, file,
                      ensure_ascii=False, indent=2)
        item = evidence[seq, center_frame, track]
        rows.append({
            "case_id": case_id, "audit_stratum": tier, "sequence": seq, "original_video": row["original_video"],
            "frame": center_frame, "track_id": track, "det_index": row["det_index"],
            "center_matched_gt": row["matched_gt"], "center_iou_max": row["output_max_iou"],
            "center_short_side": item["output_short_side"], "center_score": item["output_score"],
            "gmc_speed": item["gmc_speed_diag_per_frame"],
            "card_png": f"{case_id}/review.png", "human_label": "pending", "human_reviewer": "", "human_reason": "",
        })
    csv_output(args.output, "E3_manual_review.csv", rows)
    with (args.output / "review_contract.json").open("x", encoding="utf-8") as stream:
        json.dump({"status":"WAITING_FOR_HUMAN_REVIEW", "samples":len(rows), "source_videos":len({r["original_video"] for r in rows}),
                   "sampling":"Highest FP burden with source diversity, same-video small near-static CLEAR matches and localization controls",
                   "labels":["confirmed_background","possible_unlabelled_object","true_object","uncertain"],
                   "a_digest": fingerprint(args.input / "E3_A_evaluation_audit.json"),
                   "manual_review_sha256": fingerprint(args.output / "E3_manual_review.csv"),
                   "note":"Sampling is diagnostic. Traces and GT are used for overlay and evidence selection only."},
                  stream, ensure_ascii=False, indent=2)
    print(f"E3_B review package ready: {len(rows)} cases", flush=True)


if __name__ == "__main__":
    main()
