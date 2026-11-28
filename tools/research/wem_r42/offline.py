"""Offline CLEAR attribution for the fixed BG-only W R4.2 experiment.

Prediction has already finished before this module receives evaluation paths.
The ledger retains old W event keys while explicitly identifying output-only
measurements whose state was never committed to the online tracker.
"""
from collections import Counter, defaultdict
from pathlib import Path

from tools.research.vttiny_wem_r4.offline import events, save


def diagnose(batch, arms, names, inputs, methods):
    from experiments.vttiny_gmc.round3_joint.analysis import evaluate_sequence, load_trackeval
    from experiments.vttiny_gmc.round3_joint.common import read_json, write_json

    batch = Path(batch)
    arms, names = tuple(arms), list(names)
    if not methods or len(set(arms)) != len(arms) or len(set(names)) != len(names):
        raise ValueError("Attribution needs unique arms, sequences, and reference roles")
    if {role for role, mode in arms} != set(methods) or any((role, "off") not in arms for role in methods):
        raise ValueError("Every attribution role requires its actual off reference")
    view = batch / "eval_input"
    reference_role = next(iter(methods))
    lengths = read_json(view / "protocols" / f"{reference_role}_off.json")["sequences"]
    if set(names) != set(lengths) or inputs["sequences"] != lengths:
        raise ValueError("Attribution scope differs from the completed prediction/evaluation scope")
    for role in methods:
        if read_json(view / "protocols" / f"{role}_off.json")["sequences"] != lengths:
            raise ValueError("Reference sequence lengths differ")
    trackeval, scorer = load_trackeval(Path(inputs["trackeval"]["root"]))
    gt = Path(inputs["ground_truth"]["root"])
    output = batch / "attribution"
    output.mkdir(exist_ok=False)
    committed, lost_b, per_sequence = [], [], []
    variants = [(role, mode) for role, mode in arms if mode != "off"]
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
            old_pairs = list(events(batch / reference / "events" / method / (seq + ".jsonl"), "recovered"))
            new_pairs = list(events(batch / label / "events" / method / (seq + ".jsonl"), "recovered"))
            w_events = list(events(batch / label / "wem_events" / method / (seq + ".jsonl"), "committed"))
            w_by_gt, outcomes = defaultdict(list), Counter()
            for event in w_events:
                frame, tid = event["frame"], event["track_id"]
                before, after = old["frames"][frame - 1], new["frames"][frame - 1]
                row = after["outputs"][tid]
                gid = row["matched_gt"]
                history = [item["outputs"].get(tid, {}).get("matched_gt")
                           for item in new["frames"][max(0, frame - 6):frame - 1]]
                confident = {value for value in history if value is not None}
                anchor = next(iter(confident)) if sum(value is not None for value in history) >= 2 and len(confident) == 1 else None
                outcome = ("unmatched_FP" if gid is None else "TP_past_identity_unconfirmed" if anchor is None
                           else "TP_same_old_identity" if gid == anchor else "TP_other_identity")
                newly_covered = gid is not None and gid not in before["coverage"]
                if gid is not None:
                    w_by_gt[gid, tid].append(frame)
                outcomes[outcome] += 1
                outcomes["newly_covered"] += int(newly_covered)
                is_output_only = mode == "W0_emit_only"
                if is_output_only and (event.get("state_committed") is not False or event.get("output_only") is not True):
                    raise ValueError("Output-only event lacks explicit no-state-feedback evidence")
                state_committed = event.get("state_committed", True)
                if type(state_committed) is not bool or (not is_output_only and not state_committed):
                    raise ValueError("Weak event state commitment disagrees with its experiment arm")
                kind = ("emitted_measurement_no_state_feedback" if is_output_only else
                        "prediction" if event["prediction"] else "measurement")
                committed.append(dict(**identity, frame=frame, track_id=tid, det_index=event["det_index"],
                                      output_kind=kind, state_committed=int(state_committed), matched_gt=gid,
                                      preceding_gt=anchor, newly_covered=int(newly_covered), outcome=outcome,
                                      max_iou=row["max_iou"]))
            current_b = {(event["frame"], event["track_id"], event["det_index"]) for event in new_pairs}
            before_recovery_lost = Counter()
            for event in old_pairs:
                frame, tid = event["frame"], event["track_id"]
                if (frame, tid, event["det_index"]) in current_b:
                    continue
                gid = old["frames"][frame - 1]["outputs"].get(tid, {}).get("matched_gt")
                retained = new["frames"][frame - 1]["coverage"].get(gid)
                recent = [time for time in w_by_gt[gid, tid] if 0 <= frame - time <= 10]
                disposition = ("reference_B_not_matched" if gid is None else "reference_B_coverage_lost" if retained is None
                               else "same_id_covered_with_recent_W" if retained == tid and recent else "coverage_retained")
                before_recovery_lost[disposition] += 1
                lost_b.append(dict(**identity, frame=frame, track_id=tid, det_index=event["det_index"], gt=gid,
                                   observed_track=retained, previous_w=max(recent, default=None), disposition=disposition))
            per_sequence.append(dict(**identity, gained=gains, lost=losses,
                                     fp_delta=new["counts"]["FP"] - old["counts"]["FP"],
                                     idsw_delta=new["counts"]["IDSW"] - old["counts"]["IDSW"],
                                     B_reference=len(old_pairs), B_actual=len(new_pairs),
                                     **{key: outcomes[key] for key in ("unmatched_FP", "TP_same_old_identity",
                                        "TP_other_identity", "TP_past_identity_unconfirmed", "newly_covered")},
                                     **{key: before_recovery_lost[key] for key in ("reference_B_not_matched",
                                        "reference_B_coverage_lost", "same_id_covered_with_recent_W", "coverage_retained")}))
    save(output / "committed_clear.csv", committed, ["arm", "sequence", "frame", "track_id", "det_index", "output_kind",
                                                     "state_committed", "matched_gt", "preceding_gt", "newly_covered", "outcome", "max_iou"])
    save(output / "changed_B_events.csv", lost_b, ["arm", "sequence", "frame", "track_id", "det_index", "gt",
                                                 "observed_track", "previous_w", "disposition"])
    save(output / "per_sequence.csv", per_sequence, per_sequence[0].keys())
    totals = defaultdict(Counter)
    for row in per_sequence:
        totals[row["arm"]].update({key: value for key, value in row.items() if isinstance(value, int)})
    write_json(output / "summary.json", dict(status="COMPLETE", evaluator=scorer, variants=totals,
                                             scope="offline CLEAR and existing W/B event keys only",
                                             state_commitment="committed event may be output-only; consult state_committed"))
