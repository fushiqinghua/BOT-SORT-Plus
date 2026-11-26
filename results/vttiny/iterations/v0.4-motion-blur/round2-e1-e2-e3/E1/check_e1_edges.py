"""Verify feasibility changes and identify direct-center versus assignment spillover."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from audit_e1 import (ROOT, STAGES, TRACKER, FrameEvents, Replay, base_identity, csv_groups,
                      high_result, read, seq_inputs, support, table, write)
from audit_e1 import PlusConfig, MotionTransport, freeze_verify_scope


def load_deltas(path):
    indexed=defaultdict(list)
    with path.open(encoding="utf-8-sig",newline="") as stream:
        for r in csv.DictReader(stream):
            indexed[r["sequence"],int(r["frame"])].append(r)
    return indexed


def role_delta(snapshot, stage, tid, di, history):
    frame=snapshot["frame"]
    if stage=="native_high":
        track=next(x for x in snapshot["high"]["tracks"] if x[0]==tid)
        center=history.get((frame,tid))
        box=np.asarray(track[3:])
        changed=center is not None and not np.allclose(center,(box[:2]+box[2:])*.5,atol=1e-12,rtol=0.)
        return "direct_current_center" if changed else "assignment_spillover"
    track=next(x for x in snapshot["b"]["tracks"] if x["id"]==tid)
    cand=next(x for x in snapshot["b"]["candidates"] if x["obs"][-1][1]==di)
    obsframes={x[0] for x in cand["obs"]}
    affected=[]
    for f,box,_ in track["predictions"]:
        center=history.get((f,tid))
        geom=np.asarray(box)
        if f in obsframes and center is not None and not np.allclose(
                center,(geom[:2]+geom[2:])*.5,atol=1e-12,rtol=0.):
            affected.append(f)
    if frame in affected:
        return "direct_current_center"
    return "direct_history_only" if affected else "assignment_spillover"


def transitions(seq,frame,stage,variant,rows,cols,before,after,track_rows,det_rows):
    pending=np.argwhere(before!=after)
    for i,j in pending:
        tid,det=rows[i],cols[j]
        evidence,gid,_=support(track_rows,det_rows,frame,tid,det)
        yield dict(variant=variant,stage=stage,sequence=seq,original_video=seq.rsplit("_",1)[0],
                   frame=frame,track_id=tid,det_index=det,gt_support=evidence,gt_id=gid,
                   direction="added" if after[i,j] else "removed")


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--repo",required=True,type=Path)
    args=cli.parse_args()
    out=ROOT/"edge_analysis"
    out.mkdir(exist_ok=False)
    old=args.repo.parent/"BOT-SORT-Plus"
    settings=read(old/"configs/diagnostics/motion_blur_v0.4.local.json")
    frozen,protocol,baseline=freeze_verify_scope(settings,old)
    cfg=PlusConfig.load(args.repo/"configs/botsort_plus/B_early_moderate.json")
    parity=read(ROOT/"full_parity.json")
    assert parity["status"]=="PASS"
    assert read(ROOT/"compare_analysis/E1_prediction_summary.json")["status"]=="COMPLETE"
    original=load_deltas(ROOT/"compare_analysis/E1_assignment_deltas.csv")
    targets=csv_groups(ROOT/"cohort/lost_cohort.csv",("frame","track_id"))
    all_tracks=csv_groups(frozen/"analysis/track_error.csv",("frame","track_id"))
    all_dets=csv_groups(frozen/"analysis/detection_audit.csv",("frame","det_index"))
    edges,scopes=[],[]
    for seq,proof in parity["sequences"].items():
        source,packets=seq_inputs(frozen,ROOT,seq,proof["frames"],"full",protocol,
                                  Path(settings["features_dir"]),baseline)
        replay=Replay(cfg,packets)
        motion=MotionTransport()
        historical=({}, {})
        for state,snapshot in source:
            frame=state["frame"]
            assert snapshot["frame"]==frame
            base_identity(snapshot,state)
            replay.frame(snapshot)
            h0,b0=high_result(frame,snapshot["high"]),replay.recovery(frame)
            centers=motion.advance(state)
            actual_delta=original.get((seq,frame),[])
            verification=set()
            for i,variant in enumerate(("P1","P2")):
                lookup=historical[i]
                for key in [key for key in lookup if key[0]<=frame-cfg.recovery.window]:
                    del lookup[key]
                curr={tid:value[i] for tid,value in centers.items() if value[i] is not None}
                lookup.update({(frame,tid):v for tid,v in curr.items()})
                h,b=high_result(frame,snapshot["high"],curr),replay.recovery(frame,lookup)
                native_ids=[r[0] for r in snapshot["high"]["tracks"]]
                native_dets=[r[0] for r in snapshot["high"]["detections"]]
                b_ids=[r["id"] for r in snapshot["b"]["tracks"]]
                b_dets=[c["obs"][-1][1] for c in snapshot["b"]["candidates"]]
                threshold=snapshot["high"]["threshold"]
                for stage,rows,cols,zero,result in (
                        ("native_high",native_ids,native_dets,h0["cost"]<=threshold,h["cost"]<=threshold),
                        ("B_raw",b_ids,b_dets,np.isfinite(b0["raw"]),np.isfinite(b["raw"])),
                        ("B_eligible",b_ids,b_dets,np.isfinite(b0["eligible"]),np.isfinite(b["eligible"]))):
                    edges.extend(transitions(seq,frame,stage,variant,rows,cols,zero,result,all_tracks[seq],all_dets[seq]))
                for stage,former,new in (("native_high",h0["pairs"],h["pairs"]),("B",b0["pairs"],b["pairs"])):
                    for direction, pairs in (("added",new.keys()-former.keys()),("removed",former.keys()-new.keys())):
                        verification.update((variant,stage,direction,tid,di) for tid,di in pairs)
                for row in actual_delta:
                    if row["variant"]!=variant:
                        continue
                    tid,di=int(row["track_id"]),int(row["det_index"])
                    scopes.append({**row,"scope":"primary_cohort" if (frame,tid) in targets[seq] else "outside_primary_cohort",
                        "center_change_role":role_delta(snapshot,row["stage"],tid,di,lookup)})
            expected={(r["variant"],r["stage"],r["direction"],int(r["track_id"]),int(r["det_index"])) for r in actual_delta}
            if expected!=verification:
                raise ValueError((seq,frame,"assignment deltas do not reproduce"))
        print("E1_EDGE_SEQUENCE",seq,flush=True)
    edge_stats=Counter((x["variant"],x["stage"],x["direction"],x["gt_support"]) for x in edges)
    scope_stats=Counter((x["variant"],x["stage"],x["direction"],x["gt_support"],x["scope"],x["center_change_role"])
                        for x in scopes)
    fields=("variant","stage","direction","gt_support")
    table(out/"E1_feasible_edge_deltas.csv",edges)
    table(out/"E1_feasible_edge_summary.csv",
          [{**dict(zip(fields,k)),"edges":n} for k,n in sorted(edge_stats.items())])
    table(out/"E1_assignment_scope.csv",scopes)
    table(out/"E1_assignment_scope_summary.csv",
          [{**dict(zip((*fields,"scope","center_change_role"),k)),"pairs":n} for k,n in sorted(scope_stats.items())])
    write(out/"E1_edge_proof.json",dict(status="COMPLETE",recomputed_assignment_changes_exact=True,
                                       sequences=len(parity["sequences"]),edges=len(edges),assignment_rows=len(scopes)))
    print("E1_FEASIBLE_EDGE_AUDIT_COMPLETE",len(edges),len(scopes),flush=True)


if __name__=="__main__":
    main()
