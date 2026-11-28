"""Compare fixed W admission scores and lock a training-side operating point."""

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata

from score_features import METHOD, PARTITIONS, REFERENCE_SHA, RGB_SHA, SPLIT_SHA, read, sha


PROTOCOL_SHA = "6aff764a2ef8209e96079e1cf52f65491df87bf68937ce956ec99e54f1a41ffb"
METHODS = ("geometry", "geometry_score", "history", "history_competition")
COSINES = (.75, .85, .92)
MARGINS = (.02, .05, .08)
LABELS = ("same_identity", "other_identity", "confirmed_background", "unknown", "ignored")


def grid():
    yield from (dict(method="geometry", floor=x) for x in (.5, .6, .7))
    yield from (dict(method="geometry_score", floor=x) for x in (.4, .55, .7))
    yield from (dict(method="history", cosine_floor=x) for x in COSINES)
    yield from (dict(method="history_competition", cosine_floor=x, margin=m)
                for x, m in itertools.product(COSINES, MARGINS))


FIXED_DIAGNOSTICS = (
    dict(method="geometry", floor=.5), dict(method="geometry_score", floor=.55),
    dict(method="history", cosine_floor=.85), dict(method="history_competition", cosine_floor=.85, margin=.05),
)


def line_rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def key(row):
    return row["sequence"], row["frame"], row["track_id"], row["det_index"]


def score(row, setting):
    method = setting["method"]
    if method == "reject_all":
        return None
    if method in ("geometry", "geometry_score"):
        value = row["native_iou"] if method == "geometry" else (
            row["native_iou"] + row["normalized_score"])/2.
        return value if value >= setting["floor"] else None
    if row["history_score"] is None or row["cosine"] < setting["cosine_floor"] or row["motion"] < .25:
        return None
    value = row["history_score"]
    if value < .55:
        return None
    if method == "history_competition":
        if row["row_unknown"] or row["column_unknown"]:
            return None
        if any(row[k] is not None and row[k] < setting["margin"] for k in ("row_margin","column_margin")):
            return None
    return value


def assign(frame, setting):
    viable = [(i, value) for i,row in enumerate(frame) if (value := score(row, setting)) is not None]
    if not viable:
        return []
    tracks = sorted({frame[i]["track_id"] for i, _ in viable})
    groups = sorted({tuple(frame[i]["raw_box_xyxy"]) for i, _ in viable})
    rr = {tid:i for i,tid in enumerate(tracks)}
    cc = {group:i for i,group in enumerate(groups)}
    matrix = np.zeros((len(tracks), len(groups)+len(tracks)))
    chosen = {}
    for index, value in viable:
        row = frame[index]
        slot = rr[row["track_id"]], cc[tuple(row["raw_box_xyxy"])]
        # A tiny deterministic score/detection key resolves otherwise equal utility.
        order = value, row["raw_score"], -row["det_index"]
        if slot not in chosen or order > chosen[slot][0]:
            chosen[slot] = order, index
            matrix[slot] = value + row["normalized_score"] * 1e-10 + 1e-12/(row["det_index"]+1)
    return [frame[chosen[i,j][1]] for i,j in zip(*linear_sum_assignment(matrix,maximize=True))
            if (i,j) in chosen]


def compare_raw(left, right):
    if left["raw_box_xyxy"]!=right["raw_box_xyxy"] or left["raw_score"]!=right["raw_score"]:
        raise ValueError("Candidate key does not resolve to original score and coordinates")
    return right


def evaluate(rows, setting, budget=5):
    per_frame = defaultdict(list)
    group_label = {}
    for row in rows:
        if budget is not None and row["geometry_group_rank"] > budget:
            continue
        per_frame[row["sequence"],row["frame"]].append(row)
        identity_key = (row["sequence"],row["frame"],row["track_id"],*row["raw_box_xyxy"])
        existing = group_label.setdefault(identity_key,row)
        if existing["label"]["identity_label"] != row["label"]["identity_label"]:
            raise ValueError("Exact duplicate rows have inconsistent offline identity labels")
    opportunities = Counter()
    group_sources = defaultdict(set)
    possible_fn = set()
    for row in group_label.values():
        label=row["label"]
        name=label["identity_label"]
        opportunities[name]+=1
        group_sources[name].add(label["source_video"])
        if label["opportunity"]:
            possible_fn.add((row["sequence"],row["frame"],label["gt_id"]))
    adopted = [row for group in per_frame.values() for row in assign(group,setting)]
    outcome = Counter(row["label"]["identity_label"] for row in adopted)
    targets, episodes, useful_sources = set(), set(), set()
    for row in adopted:
        label = row["label"]
        if not label["opportunity"]:
            continue
        seq, gt = row["sequence"], label["gt_id"]
        targets.add((seq,row["frame"],gt))
        episodes.add((seq,gt,label["fn_episode_start_frame"]))
        useful_sources.add(label["source_video"])
    counts = dict(available_groups=len(group_label), accepted=len(adopted),
                  useful_target_frames=len(targets), useful_fn_episodes=len(episodes), possible_target_frames=len(possible_fn),
                  useful_proxy_groups=len(useful_sources))
    for label in LABELS:
        counts["available_"+label]=opportunities[label]
        counts["available_proxy_groups_"+label]=len(group_sources[label])
        counts["accepted_"+label]=outcome[label]
    return counts,adopted


def constraint_failures(result):
    reasons = []
    for label in ("other_identity","confirmed_background"):
        n, errors = result["available_"+label],result["accepted_"+label]
        if n and errors*100 > n:
            reasons.append(label+"_budget_exceeded")
    if result["accepted_unknown"] or result["accepted_ignored"]:
        reasons.append("unknown_or_ignored_accepted")
    return reasons


def pilot_failures(result):
    reasons = constraint_failures(result)
    for label in ("other_identity","confirmed_background"):
        n = result["available_"+label]
        if n < 20 or result["available_proxy_groups_"+label] < 3:
            reasons.append(label+"_insufficient_support")
    if result["useful_fn_episodes"] < 10 or result["useful_proxy_groups"] < 3:
        reasons.append("insufficient_correct_coverage")
    return reasons


def rank_choice(setting, counts):
    return (counts["useful_fn_episodes"],
            -(counts["accepted_other_identity"]+counts["accepted_confirmed_background"]+
              counts["accepted_unknown"]+counts["accepted_ignored"]),
            counts["useful_target_frames"],
            setting.get("floor",setting.get("cosine_floor",0)), setting.get("margin",0))


def auc(values0, values1):
    if not values0 or not values1:
        return None
    ranks=rankdata(values0+values1)
    n=len(values1)
    return float((ranks[len(values0):].sum() - n*(n+1)/2)/(n*len(values0)))


def feature_stats(rows):
    common = [r for r in rows if (r["history_score"] is not None
              and r["label"]["identity_label"] in ("same_identity","other_identity"))]
    labels = defaultdict(list)
    for row in common:
        labels[row["label"]["identity_label"]].append(row)
    values = {}
    for name, fun in (
        ("native_iou",lambda r:r["native_iou"]),
        ("geometry_score",lambda r:(r["native_iou"]+r["normalized_score"])/2),
        ("rgb_cosine",lambda r:r["cosine"]),
        ("motion",lambda r:r["motion"]),
        ("history_score",lambda r:r["history_score"]),
    ):
        values[name]=auc([fun(x) for x in labels["other_identity"]],[fun(x) for x in labels["same_identity"]])
    return dict(rows=len(rows),current_rgb_valid=sum(r["rgb_weak_valid"] for r in rows),
                usable_history=sum(r["history_score"] is not None for r in rows),
                row_unknown=sum(r["row_unknown"] for r in rows),
                column_unknown=sum(r["column_unknown"] for r in rows),
                competition_present=sum(bool(r["challengers"]) for r in rows),
                common_scored_same=len(labels["same_identity"]),common_scored_other=len(labels["other_identity"]),
                known_pair_auc=values)


def write_csv(path, content):
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        fields=sorted({k for row in content for k in row})
        out=csv.DictWriter(f,fieldnames=fields)
        out.writeheader()
        out.writerows(content)


def known_other_identity_cases(grouped):
    cases = []
    for part, rows in grouped.items():
        for row in rows:
            if row["label"]["identity_label"] != "other_identity":
                continue
            frame = [r for r in rows if key(r)[:2] == key(row)[:2] and r["geometry_group_rank"] <= 5]
            decisions = []
            for setting in FIXED_DIAGNOSTICS:
                assigned = {key(r) for r in assign(frame, setting)}
                decisions.append(dict(**setting, score_pass=score(row,setting) is not None, assigned=key(row) in assigned))
            cases.append(dict(partition=part, edge=row, fixed_diagnostics=decisions))
    return cases


def run(args):
    if sha(args.split)!=SPLIT_SHA or sha(args.protocol)!=PROTOCOL_SHA:
        raise ValueError("Preregistered split or protocol changed")
    split=read(args.split)
    manifest=read(args.features/"manifest.json")
    if (manifest["status"]!="COMPLETE" or manifest["scope"]!=args.scope or manifest["encoder"]["sha256"]!=RGB_SHA
            or manifest["input_split_sha256"]!=SPLIT_SHA):
        raise ValueError("Feature manifest is inconsistent")
    if args.scope=="full" and (manifest["reference"]!=METHOD or manifest["reference_protocol_sha256"]!=REFERENCE_SHA):
        raise ValueError("Full feature extraction lacks frozen decoded RGB proof")
    provenance=read(args.labels/"summary.json")
    if (provenance["status"]!="TRAINING_DIAGNOSTIC_LABELS_ONLY" or provenance["stage"]!=args.scope
            or provenance["split_sha256"]!=SPLIT_SHA or provenance["official_test_gt_used"]
            or provenance["reference"]!=METHOD
            or provenance["source"]["capture"]!=manifest["parity_sha256"]):
        raise ValueError("Unexpected training-only label provenance")
    sequences={s:part for part in PARTITIONS for s in split["split"][part]["sequences"]}
    if args.scope=="smoke":
        sequences={seq:part for seq,part in sequences.items() if seq in manifest["sequences"]}
    labels={}
    label_files={part:args.labels/part/"candidate_edges.jsonl.gz" for part in PARTITIONS}
    for part,path in label_files.items():
        if sha(path)!=provenance["partitions"][part]["labels_sha256"]:
            raise ValueError("Offline label hash changed")
    for part,row in ((part,row) for part,path in label_files.items() for row in line_rows(path)):
        seq=row["sequence"]
        if sequences.get(seq)!=part or row["source_video"]!=split["training_sequence_to_source_video"][seq]:
            raise ValueError("Label source split drift")
        qualified=(row["formal_geometry_allowed"] and row["gmc_guard_allowed"] and not row["hard_ownership_conflict"]
                   and not row["near_duplicate_ownership_risk"])
        if not qualified:
            continue
        k=key(row)
        if k in labels:
            raise ValueError("Repeated labeled raw edge")
        labels[k]=row
    grouped=defaultdict(list)
    if set(manifest["sequences"])!=set(sequences):
        raise ValueError("Incomplete training-side feature coverage")
    label_snapshots = {seq["sequence"]: seq["provenance"]["snapshot_sha256"]
                       for part in PARTITIONS for seq in provenance["partitions"][part]["sequences"]}
    for seq,part in sorted(sequences.items()):
        path=args.features/f"{seq}.jsonl.gz"
        proof=manifest["sequences"][seq]
        if (proof["scores_sha256"]!=sha(path) or proof["partition"]!=part
                or proof["snapshot"]!=label_snapshots[seq]
                or (args.scope=="full" and not proof.get("decoded_rgb_sha256"))):
            raise ValueError("Feature snapshot source mismatch")
        count=0
        for row in line_rows(path):
            label=compare_raw(row,labels.pop(key(row)))
            grouped[part].append(dict(row,label=label))
            count+=1
        if count!=proof["counts"].get("edges",0):
            raise ValueError("Feature edge count differs from its receipt")
    if labels:
        raise ValueError(f"Unscored geometry and ownership eligible label edges: {len(labels)}")
    if args.scope=="smoke":
        args.output.mkdir(parents=True,exist_ok=False)
        stats={part:feature_stats(grouped[part]) for part in PARTITIONS}
        result=dict(status="SMOKE_INPUT_AND_FEATURE_PARITY", feature_stats=stats)
        (args.output/"smoke.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
        print(result["status"], {part:v["rows"] for part,v in stats.items()},flush=True)
        return
    chosen_by_method={}
    calibrations=[]
    for setting in grid():
        counts,_=evaluate(grouped["calibration"],setting)
        failures=constraint_failures(counts)
        calibrations.append(dict(**setting,**counts,failures=" ".join(failures)))
        method=setting["method"]
        if failures:
            continue
        old=chosen_by_method.get(method)
        if old is None or rank_choice(setting,counts)>rank_choice(*old):
            chosen_by_method[method]=(setting,counts)
    empty,_=evaluate(grouped["calibration"],dict(method="reject_all"))
    calibrations.append(dict(method="reject_all",**empty,failures=""))
    threshold_chosen={}
    tables=[]
    admissions=[]
    for method in METHODS:
        selected=chosen_by_method.get(method)
        if selected is None:
            threshold_chosen[method]=dict(status="CALIBRATION_REJECT_ALL",calibration=None,holdout=None,
                                          failures=["no_grid_point_meets_known_error_and_unknown_budgets"])
            continue
        setting,counts=selected
        held,kept=evaluate(grouped["holdout"],setting)
        reasons={"calibration":pilot_failures(counts),"holdout":pilot_failures(held)}
        threshold_chosen[method]=dict(status="LOCKED_DIAGNOSTIC" if any(reasons.values()) else "PILOT_ELIGIBLE",
                                      setting=setting,calibration=counts,holdout=held,failures=reasons)
        for row in kept:
            label=row["label"]
            admissions.append(dict(method=method,partition="holdout",sequence=row["sequence"],
                                   frame=row["frame"],track_id=row["track_id"],det_index=row["det_index"],
                                   identity_label=label["identity_label"],useful=label["opportunity"]))
    for part in PARTITIONS:
        for setting in FIXED_DIAGNOSTICS:
            value,_=evaluate(grouped[part],setting)
            tables.append(dict(partition=part,**setting,budget=5,**value))
        for budget in (1,10,None):
            value,_=evaluate(grouped[part],FIXED_DIAGNOSTICS[-1],budget)
            tables.append(dict(partition=part,**FIXED_DIAGNOSTICS[-1],
                               budget="all" if budget is None else budget,**value))
        unavailable=defaultdict(list)
        for row in grouped[part]:
            unavailable[(row["sequence"],row["frame"],row["track_id"])].append(row)
        subset=[max(rows,key=lambda r:(r["raw_score"],-r["det_index"])) for rows in unavailable.values()
                if rows[0]["history_score"] is None and len({tuple(r["raw_box_xyxy"]) for r in rows})==1
                and not rows[0]["challengers"]]
        value,_=evaluate(subset,FIXED_DIAGNOSTICS[0])
        tables.append(dict(partition=part,method="geometry_unique_missing_history",floor=.5,budget=5,**value))
    eligible=[name for name,result in threshold_chosen.items() if result["status"]=="PILOT_ELIGIBLE"]
    background_budget_state = {}
    for part in PARTITIONS:
        counts, _ = evaluate(grouped[part], dict(method="reject_all"))
        n = counts["available_confirmed_background"]
        background_budget_state[part] = ("NOT_AVAILABLE" if not n else "COUNTED_IN_OFFLINE_DIAGNOSTICS")
    result=dict(status="PILOT_ELIGIBLE" if eligible else "REJECT_ALL_FALLBACK",
                decision="retain_fallback" if not eligible else "pilot_evidence_available",
                operating_policy="reject_all" if not eligible else "requires_separate_online_pilot",
                enabled_online=False, eligible_methods=eligible,preregistration_sha256=sha(args.protocol),
                background_budget_state=background_budget_state,
                admission_implementation_sha256=sha(Path(__file__)),
                frozen_split_sha256=sha(args.split),frozen_rgb_encoder_sha256=RGB_SHA,
                feature_manifest_sha256=sha(args.features/"manifest.json"),
                offline_label_receipt_sha256=sha(args.labels/"summary.json"),
                offline_label_sha256={part:sha(path) for part,path in label_files.items()},
                feature_stats={part:feature_stats(grouped[part]) for part in PARTITIONS},
                selected_by_method=threshold_chosen,
                fixed_diagnostics=tables,
                calibration_grid=calibrations,
                known_other_identity_cases=known_other_identity_cases(grouped),
                scope="training-side diagnostic only; no untouched sensor holdout and no W online replay")
    args.output.mkdir(parents=True,exist_ok=False)
    write_csv(args.output/"calibration_grid.csv",calibrations)
    write_csv(args.output/"fixed_diagnostics.csv",tables)
    if admissions:
        write_csv(args.output/"heldout_diagnostic_adoptions.csv",admissions)
    (args.output/"calibration_lock.json").write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)
                                                   +"\n",encoding="utf-8")
    print(result["status"],result["eligible_methods"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("features","labels","split","protocol","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--scope",choices=("smoke","full"),default="full")
    run(parser.parse_args())
