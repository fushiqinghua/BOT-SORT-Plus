"""Join committed W and changed B events to the official CLEAR ledger after prediction."""

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path


def events(path, event):
    with path.open(encoding="utf-8") as f:
        yield from (record for line in f if (record := json.loads(line)).get("event") == event)


def save(path, rows, columns):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def diagnose(batch, arms, names, inputs, methods):
    from experiments.vttiny_gmc.round3_joint.analysis import evaluate_sequence, load_trackeval
    from experiments.vttiny_gmc.round3_joint.common import read_json, write_json

    trackeval, scorer = load_trackeval(Path(inputs["trackeval"]["root"]))
    gt = Path(inputs["ground_truth"]["root"])
    view = batch / "eval_input"
    output = batch / "attribution"
    output.mkdir(exist_ok=False)
    committed, lost_b, per_sequence = [], [], []
    variants = [(role, mode) for role, mode in arms if mode != "off"]
    lengths = read_json(view / "protocols/B_off.json")["sequences"]
    for seq in names:
        n = lengths[seq]
        before_by_role = {role: evaluate_sequence(trackeval, gt, view / "trackers", f"{role}_off", seq, n, output)
                          for role in methods}
        for role, mode in variants:
            label, reference, method = f"{role}_{mode}", f"{role}_off", methods[role]
            old = before_by_role[role]
            new = evaluate_sequence(trackeval, gt, view / "trackers", label, seq, n, output)
            identity = {"arm": label, "sequence": seq}
            gains = losses = 0
            for before, after in zip(old["frames"], new["frames"], strict=True):
                gains += len(after["coverage"].keys() - before["coverage"].keys())
                losses += len(before["coverage"].keys() - after["coverage"].keys())
            old_pairs = list(events(batch / reference / "events" / method / (seq+".jsonl"), "recovered"))
            new_pairs = list(events(batch / label / "events" / method / (seq+".jsonl"), "recovered"))
            w_events = list(events(batch / label / "wem_events" / method / (seq+".jsonl"), "committed"))
            w_by_gt = defaultdict(list)
            outcomes = Counter()
            for event in w_events:
                frame, tid = event["frame"], event["track_id"]
                before, after = old["frames"][frame-1], new["frames"][frame-1]
                row = after["outputs"][tid]
                gid = row["matched_gt"]
                history = [h["outputs"].get(tid,{}).get("matched_gt")
                           for h in new["frames"][max(0,frame-6):frame-1]]
                confident = {g for g in history if g is not None}
                anchor = next(iter(confident)) if sum(g is not None for g in history)>=2 and len(confident)==1 else None
                if gid is None:
                    outcome = "unmatched_FP"
                elif anchor is None:
                    outcome = "TP_past_identity_unconfirmed"
                elif gid == anchor:
                    outcome = "TP_same_old_identity"
                else:
                    outcome = "TP_other_identity"
                newly_covered = gid is not None and gid not in before["coverage"]
                if gid is not None:
                    w_by_gt[gid,tid].append(frame)
                outcomes[outcome] += 1
                outcomes["newly_covered"] += int(newly_covered)
                committed.append(dict(**identity, frame=frame, track_id=tid, det_index=event["det_index"],
                                      output_kind="prediction" if event["prediction"] else "measurement",
                                      matched_gt=gid, preceding_gt=anchor, newly_covered=int(newly_covered),
                                      outcome=outcome, max_iou=row["max_iou"]))
            current_b = {(e["frame"],e["track_id"],e["det_index"]) for e in new_pairs}
            before_recovery_lost = Counter()
            for event in old_pairs:
                frame, tid = event["frame"], event["track_id"]
                if (frame,tid,event["det_index"]) in current_b:
                    continue
                gid = old["frames"][frame-1]["outputs"].get(tid,{}).get("matched_gt")
                retained = new["frames"][frame-1]["coverage"].get(gid)
                recent = [t for t in w_by_gt[gid,tid] if 0 <= frame-t <= 10]
                if gid is None:
                    disposition = "reference_B_not_matched"
                elif retained is None:
                    disposition = "reference_B_coverage_lost"
                elif retained==tid and recent:
                    disposition = "same_id_covered_with_recent_W"
                else:
                    disposition = "coverage_retained"
                before_recovery_lost[disposition] += 1
                lost_b.append(dict(**identity,frame=frame,track_id=tid,det_index=event["det_index"],gt=gid,
                                   observed_track=retained,previous_w=max(recent,default=None),disposition=disposition))
            per_sequence.append(dict(**identity, gained=gains, lost=losses,
                fp_delta=new["counts"]["FP"]-old["counts"]["FP"], idsw_delta=new["counts"]["IDSW"]-old["counts"]["IDSW"],
                B_reference=len(old_pairs), B_actual=len(new_pairs),
                **{k:outcomes[k] for k in ("unmatched_FP","TP_same_old_identity","TP_other_identity",
                                           "TP_past_identity_unconfirmed","newly_covered")},
                **{k:before_recovery_lost[k] for k in ("reference_B_not_matched", "reference_B_coverage_lost",
                                                       "same_id_covered_with_recent_W","coverage_retained")}))
    save(output/"committed_clear.csv",committed,["arm","sequence","frame","track_id","det_index","output_kind",
                                                  "matched_gt","preceding_gt","newly_covered","outcome","max_iou"])
    save(output/"changed_B_events.csv",lost_b,["arm","sequence","frame","track_id","det_index","gt",
                                               "observed_track","previous_w","disposition"])
    save(output/"per_sequence.csv",per_sequence,per_sequence[0].keys())
    totals = defaultdict(Counter)
    for r in per_sequence:
        totals[r["arm"]].update({k:v for k,v in r.items() if isinstance(v,int)})
    write_json(output/"summary.json",dict(status="COMPLETE", evaluator=scorer, variants=totals,
               scope="offline CLEAR and existing W/B event keys only"))
