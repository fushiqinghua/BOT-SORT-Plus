"""Audit the replay's intervention footprint, without opening ground truth."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

FROZEN = "B_early_moderate"
VARIANT = "B_postlow_v1"


def read_tracks(root, name, seq):
    path = root / "trackers" / name / "data" / (seq + ".txt")
    with path.open(newline="", encoding="utf-8") as stream:
        return {(int(row[0]), int(row[1])): row[2:] for row in csv.reader(stream)}


def read_events(root, name, seq):
    path = root / "events" / name / (seq + ".jsonl")
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def compare(frozen, new, seq):
    base = read_tracks(frozen, FROZEN, seq)
    variant = read_tracks(new, VARIANT, seq)
    before = read_events(frozen, FROZEN, seq)
    after = read_events(new, VARIANT, seq)
    probes = [event for event in after if event["event"] == "post_B_weak_check"]
    accepted = [event for event in after if event.get("stage") == "post_B_weak"]
    first = min((event["frame"] for event in accepted), default=float("inf"))
    for key in base.keys() | variant.keys():
        if key[0] < first:
            assert base.get(key) == variant.get(key), (seq, key, "pre-intervention prediction")
    assert [e for e in before if e.get("frame", first) < first] == [
        e for e in after if e.get("frame", first) < first and e["event"] != "post_B_weak_check"
    ], (seq, "pre-intervention trace")
    if first == float("inf"):
        assert before == [e for e in after if e["event"] != "post_B_weak_check"], seq

    frame_observations = defaultdict(list)
    for event in after:
        if event["event"] in {"match", "recovered", "native_birth"}:
            frame_observations[event["frame"]].append(event)
    footprint = []
    for event in accepted:
        frame, tid, index = (int(event[k]) for k in ("frame", "track_id", "det_index"))
        same_frame = frame_observations[frame]
        assert sum(e.get("track_id") == tid for e in same_frame) == 1
        assert sum(e.get("det_index") == index for e in same_frame) == 1
        probe = next(e for e in probes if e["frame"] == frame and e["track_id"] == tid)
        assert probe["reason"] == "accepted" and probe["best_det_index"] == index
        assert probe["best_pred_iou"] - max(probe["row_rival_iou"], probe["track_rival_iou"]) >= .10
        key = (frame, tid)
        footprint.append({
            **event, "check": probe, "frozen_mot_output": base.get(key),
            "variant_mot_output": variant.get(key),
            "variant_previous_three_output_frames": [
                f for f in range(frame - 3, frame) if (f, tid) in variant
            ],
            "variant_following_three_output_frames": [
                f for f in range(frame + 1, frame + 4) if (f, tid) in variant
            ],
        })
    diff_keys = {key for key in base.keys() | variant.keys() if base.get(key) != variant.get(key)}
    return {
        "check_reasons": dict(Counter(event["reason"] for event in probes)),
        "weak_match_count": len(accepted),
        "weak_match_emitted": sum(e["variant_mot_output"] is not None for e in footprint),
        "changed_mot_frames": sorted({f for f, _ in diff_keys}),
        "frozen_only_row_count": len(base.keys() - variant.keys()),
        "variant_only_row_count": len(variant.keys() - base.keys()),
        "same_id_changed_row_count": len(
            {key for key in base.keys() & variant.keys() if base[key] != variant[key]}
        ),
        "direct_weak_matches": footprint,
    }


def main():
    parser = ArgumentParser(description=__doc__)
    for arg in ("frozen", "new", "destination"):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    protocol = json.loads((args.new / "protocols" / (VARIANT + ".json")).read_bytes())
    assert protocol["status"] == "COMPLETE"
    document = {seq: compare(args.frozen, args.new, seq) for seq in protocol["sequences"]}
    with args.destination.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    for seq, details in document.items():
        if details["weak_match_count"]:
            print(seq, details["weak_match_count"], details["weak_match_emitted"],
                  len(details["changed_mot_frames"]), details["check_reasons"])


if __name__ == "__main__":
    main()
