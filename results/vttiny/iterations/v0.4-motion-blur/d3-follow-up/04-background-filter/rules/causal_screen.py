"""Screen predeclared read-only gates on sealed output-only BoT-SORT replay."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from math import hypot
from pathlib import Path
from zipfile import ZipFile

METHOD = "B_postlow_output_v1"
BASE = "B_early_moderate"
MATCH_EVENTS = frozenset(("match", "recovered", "native_birth"))


def mot_rows(raw):
    rows = defaultdict(list)
    for row in csv.reader(io.StringIO(raw.decode("utf-8"))):
        frame, tid = map(int, row[:2])
        x, y, w, h, score = map(float, row[2:7])
        rows[tid].append((frame, x + w / 2, y + h / 2, w, h, score))
    return rows


def hash_member(archive, hashes, name):
    data = archive.read(name)
    assert hashlib.sha256(data).hexdigest() == hashes[name], name
    return data


def csv_data(raw):
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))


def row_key(row):
    return row["sequence"], int(row["frame"]), int(row["track_id"])


def build_features(archive, hashes, sequences):
    features = []
    for sequence in sorted(sequences):
        native = mot_rows(hash_member(archive, hashes, f"full/trajectories/{BASE}/{sequence}.txt"))
        lines = hash_member(archive, hashes, f"full/events/{METHOD}/{sequence}.jsonl").splitlines()
        events = [json.loads(line) for line in lines]
        supplemental_keys = set()
        restores = defaultdict(list)
        observed = defaultdict(dict)
        for event in events:
            kind = event["event"]
            if kind in MATCH_EVENTS:
                observed[event["track_id"]][event["frame"]] = kind if kind != "match" else event["stage"]
            if kind == "recovered":
                restores[event["track_id"]].append(event)
            if kind != "output_only_low":
                continue
            frame, tid = event["frame"], event["track_id"]
            real = observed[tid]
            assert event["last_native_observation_frame"] == frame - 1
            assert frame - 1 in real
            recovery = restores[tid][-1]
            assert recovery["frame"] == event["b_recovery_frame"] < frame
            history = [row for row in native[tid] if row[0] < frame]
            assert history and history[-1][0] == frame - 1
            previous = history[-1]
            recent = [row for row in history if frame - row[0] <= 15]
            span = max(hypot(a[1] - b[1], a[2] - b[2]) for a in recent for b in recent)
            prior_diagonal = hypot(previous[3], previous[4])
            x1, y1, x2, y2 = event["xyxy"]
            features.append(
                dict(sequence=sequence, frame=frame, track_id=tid,
                     detector_score=event["score"],
                     prediction_iou=1 - event["cost"],
                     min_side_px=min(x2 - x1, y2 - y1),
                     prior_score=previous[5],
                     high_10=any(frame - row[0] <= 10 and row[5] >= .5 for row in recent),
                     high_30=any(frame - row[0] <= 30 and row[5] >= .5 for row in history),
                     native_rows_15=len(recent), raw_center_span_15=span / prior_diagonal,
                     restores_15=sum(frame - e["frame"] <= 15 for e in restores[tid]),
                     prior_observation_stage=real[frame-1],
                     B_age=frame - recovery["frame"],
                     B_gap=recovery["age"], B_hits=recovery["hits"],
                     B_app=recovery["appearance_cost"], B_geo=recovery["geometry_cost"]))
            assert (frame, tid) not in supplemental_keys
            supplemental_keys.add((frame, tid))
    return features


def choose(rows):
    rules = {
        "score_midband": lambda row: row["detector_score"] >= .075,
        "recent_native_high_30": lambda row: row["high_30"],
        "recent_native_high_10": lambda row: row["high_10"],
        "min_side_8": lambda row: row["min_side_px"] >= 8,
        "raw_static_veto": lambda row: row["native_rows_15"] < 5 or row["raw_center_span_15"] > .50,
        "single_restore_15": lambda row: row["restores_15"] < 2,
    }
    return [
        {**row_key_dict(row), "rule": rule, "keep": int(decision(row))}
        for rule, decision in rules.items() for row in rows
    ]


def row_key_dict(row):
    return {k: row[k] for k in ("sequence", "frame", "track_id")}


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(masks, labels, d3):
    groups = defaultdict(Counter)
    for mask in masks:
        key = row_key(mask)
        label = labels[key]
        for scope in ("all", key[0]):
            counts = groups[mask["rule"], scope]
            if label["clear_gt"]:
                outcome = "CLEAR_TP"
            else:
                outcome = "CLEAR_FP"
            counts["available_" + outcome] += 1
            if key in d3:
                counts["available_D3_same_original_ID"] += 1
            if not mask["keep"]:
                continue
            counts["kept_" + outcome] += 1
            if key in d3:
                counts["kept_D3_same_original_ID"] += 1
            if label["attribution"] == "CLEAR_same_gt_and_prior_detection_agree":
                counts["kept_verified_prev_same_GT"] += 1
    return [
        {"rule": rule, "scope": scope, **{key: value for key, value in c.items()}}
        for (rule, scope), c in groups.items()
    ]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("archive", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    args.output.mkdir(exist_ok=False)
    with ZipFile(args.archive) as archive:
        hashes = json.loads(archive.read("EVIDENCE_SHA256.json"))
        protocol = json.loads(hash_member(archive, hashes, f"full/protocols/{BASE}.json"))
        features = build_features(archive, hashes, protocol["sequences"])
        assert len(features) == len({row_key(row) for row in features}) == 191
        masks = choose(features)
        write_csv(args.output / "causal_features.csv", features)
        write_csv(args.output / "masks_all.csv", masks)

        labels = csv_data(hash_member(archive, hashes, "parity/full/supplemental_rows.csv"))
        assert (sum(bool(x["clear_gt"]) for x in labels), len(labels)) == (72, 191)
        labels = {row_key(row): row for row in labels}
        events = csv_data(hash_member(archive, hashes, "identity/full/paired_events.csv"))
        d3 = {(row["sequence"], int(row["original_failure_frame"]),
               int(row["run_verified_anchor_id"]))
              for row in events if row["run"] == METHOD
              and row["original_failure_true_output_only_rescue"] == "True"}
        assert len(d3) == 49 and d3 <= set(labels)
        analysis = evaluate(masks, labels, d3)
    (args.output / "screen.json").write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    for row in analysis:
        if row["scope"] in ("all", "DJI_0327_3", "DJI_0339_2"):
            print(json.dumps(row))


if __name__ == "__main__":
    main()
