"""Audit frozen Lost assignments and fixed-state prediction counterfactuals."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from itertools import zip_longest
import math
from pathlib import Path
import sys

import numpy as np

from run_shadow import ROOT, TRACKER, digest, read, write
sys.path.insert(0, str(ROOT/"shadow_checkout"))

from botsort_plus.config import PlusConfig
from botsort_plus.diagnostics.common import iter_trace, load_gt
from botsort_plus.geometry import GeometryConfig, geometry_cost
from botsort_plus.io import FeatureCache, read_detection_cache
from reconstruct_e1 import Replay, high_result, moved_box, verify_value
from motion_transport import MotionTransport

VARIANTS=("P0","P1","P2")
STAGES=("native_high","B")
MISSING={"missing_with_usable_detection","missing_without_usable_detection","missing_with_ambiguous_detection"}
ACCEPTED={"match","native_birth","recovered"}


def table(path, rows):
    with path.open("x", encoding="utf-8-sig",newline="") as stream:
        if rows:
            cols=list(dict.fromkeys(k for row in rows for k in row))
            editor=csv.DictWriter(stream,fieldnames=cols)
            editor.writeheader()
            editor.writerows(rows)


def csv_groups(path, fields):
    rows=defaultdict(dict)
    with path.open(encoding="utf-8-sig",newline="") as stream:
        for row in csv.DictReader(stream):
            seq=row["sequence"]
            key=tuple(int(row[k]) for k in fields)
            if key in rows[seq]:
                raise ValueError((path,key,"duplicate row"))
            rows[seq][key]=row
    return rows


def shadow_frames(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        yield from map(json.loads,stream)


def finite(value):
    return float(value) if np.isfinite(value) else None


def margins(matrix,i,j):
    if not np.isfinite(matrix[i,j]):
        return None,None
    value=matrix[i,j]
    other_row=np.delete(matrix[i,:],j)
    other_col=np.delete(matrix[:,j],i)
    return finite(np.min(other_row,initial=np.inf)-value),finite(np.min(other_col,initial=np.inf)-value)


def native_evidence(high,score,tier,tid,di):
    if tier=="low":
        return {"native_reason":"lost_no_native_low_eligibility"}
    i=high["row"].get(tid)
    j=high["column"].get(di)
    if i is None or j is None:
        return {"native_reason":"unknown_no_native_high_row_or_column"}
    cost=float(high["cost"][i,j])
    owner=next((other for other,index in high["pairs"] if index==di),None)
    iou=1.-float(high["iou_distance"][i,j])
    if (tid,di) in high["pairs"]:
        reason="native_high_accepted"
    elif cost>score:
        reason="native_high_cost_above_gate"
    elif owner is not None:
        reason="native_high_other_track_owns_detection"
    elif any(other==tid for other,_ in high["pairs"]):
        reason="native_high_track_selected_other_detection"
    else:
        reason="native_high_feasible_not_assigned"
    row_margin,col_margin=margins(high["cost"],i,j)
    return {"native_reason":reason,"native_high_cost":cost,"native_iou":iou,
            "native_raw_reid_cost":finite(high["appearance_raw"][i,j]),
            "native_reid_within_proximity":bool(high["near"][i,j]),
            "native_owner":owner,"native_row_gap":row_margin,"native_column_gap":col_margin}


def match_stage(event):
    return event.get("stage",event["event"])


def recovery_evidence(b,capture,events,cfg,tid,di):
    candidate=b["columns"].get(di)
    own=events.get(di)
    if candidate is None:
        if own is not None and (own["event"]!="recovered" or own["track_id"]!=tid):
            return {"B_reason":"not_residual_consumed_"+match_stage(own)}
        residual=capture["candidate"]
        if di not in residual["residual"]:
            return {"B_reason":"not_in_original_residual"}
        suspect=[e for e in events.candidate_notes.get(di,())
                 if e["event"] in ("candidate_observation_suppressed","candidate_seed_deferred")]
        if len(suspect)==1:
            return {"B_reason":suspect[0]["event"]+":"+suspect[0]["reason"]}
        cid=events.candidate_lookup.get(di)
        if cid is not None and events.removed.get(cid)=="capacity":
            return {"B_reason":"candidate_capacity_removed","candidate_id":cid}
        return {"B_reason":"unknown_candidate_absent"}
    j,cid=candidate
    i=b["rows"].get(tid)
    if i is None:
        return {"B_reason":"outside_original_B_eligible_pool","candidate_id":cid}
    diag=b["diagnostics"][(tid,cid)]
    reason=diag["reason"]
    accepted=(tid,di) in b["pairs"]
    if reason=="plausible":
        if not diag.get("ready") and not diag.get("two_hit_evidence",False):
            reason="insufficient_B_history"
        elif not accepted and np.isfinite(b["eligible"][i,j]):
            reason="B_assignment_competition"
        elif not accepted:
            reason="unknown_B_not_eligible"
    row_margin,col_margin=margins(b["raw"],i,j)
    return {"B_reason":reason,"candidate_id":cid,"B_candidate_hits":diag["hits"],
            "B_two_hit_evidence":diag.get("two_hit_evidence",False),
            "B_affinity":diag.get("appearance_cost"),"B_geometry":diag.get("geometry_cost"),
            "B_cost":diag.get("cost"),"B_raw_feasible":bool(np.isfinite(b["raw"][i,j])),
            "B_final_eligible":bool(np.isfinite(b["eligible"][i,j])),
            "B_row_gap":row_margin,"B_column_gap":col_margin}


class FrameEvents(dict):
    def __init__(self,rows):
        super().__init__()
        self.candidate_notes=defaultdict(list)
        self.candidate_lookup={}
        self.removed={}
        for e in rows:
            kind=e["event"]
            if kind in ACCEPTED:
                index=int(e["det_index"])
                if index in self:
                    raise ValueError((index,"duplicate original accepted detection"))
                self[index]=e
            elif kind in ("candidate_observation_suppressed","candidate_seed_deferred"):
                self.candidate_notes[int(e["det_index"])].append(e)
            elif kind in ("candidate_seed","candidate_hit"):
                self.candidate_lookup[int(e["det_index"])]=int(e["candidate_id"])
            elif kind=="candidate_removed":
                self.removed[int(e["candidate_id"])]=e["reason"]


def support(track_rows, det_rows, f, tid, di):
    a,d=track_rows.get((f,tid)),det_rows.get((f,di))
    if not a or not d or a["gt_status"]!="present" or not a["anchor_gt_id"] or d["gt_match_status"]!="matched":
        return "unknown",None,"unverified"
    before=a["output_status"]
    if a["anchor_gt_id"] != d["gt_id"]:
        return "wrong",int(a["anchor_gt_id"]),before
    return "correct",int(a["anchor_gt_id"]),before


def changes(seq,f,stage,version,p0,pnew,track_rows,det_rows):
    for direction, delta in (("added",pnew.keys()-p0.keys()),("removed",p0.keys()-pnew.keys())):
        for tid,di in sorted(delta):
            gt,anchor,baseline_status=support(track_rows,det_rows,f,tid,di)
            previous=p0.get((tid,di))
            current=pnew.get((tid,di))
            data=track_rows.get((f,tid),{})
            meaning=(
                "candidate_without_baseline_coverage" if gt=="correct" and baseline_status in MISSING
                else "identity_continuation_only" if gt=="correct" and baseline_status=="different_id"
                else "existing_identity_output" if gt=="correct" and baseline_status=="same_id_correct"
                else "coverage_unknown" if gt=="correct" else gt
            )
            yield dict(sequence=seq,original_video=seq.rsplit("_",1)[0],frame=f,stage=stage,
                       variant=version,direction=direction,track_id=tid,det_index=di,
                       gt_support=gt,anchor_gt_id=anchor,baseline_output_status=baseline_status,
                       interpretation=meaning,state_before=data.get("state_before"),
                       last_seen=data.get("last_seen"),old_cost=previous,new_cost=current)


def base_identity(seqrow,trace):
    original=trace["after_gmc"]
    observed=seqrow["high"]["tracks"]
    if len(observed) != len(original):
        raise ValueError("High pool length differs from original trace")
    for hi,snap in zip(observed, original):
        verify_value(hi[:3],[snap["track_id"],snap["state"],snap["last_seen"]],
                     (seqrow["sequence"],seqrow["frame"],"native pool order"))
        verify_value(hi[3:],snap["xyxy"],(seqrow["sequence"],seqrow["frame"],"native box"))
    dets=trace["detections"]
    for det in seqrow["high"]["detections"]:
        verify_value(det[2:],dets[det[0]]["xyxy"],(seqrow["sequence"],seqrow["frame"],"native detection box"))
        verify_value(det[1],dets[det[0]]["score"],(seqrow["sequence"],seqrow["frame"],"native detection score"))
    frame=trace["frame"]
    checks=[{k:v for k,v in e.items() if k not in ("event","track_origin")}
            for e in trace["events"] if e["event"]=="recovery_check"]
    verify_value(seqrow["b"]["diagnostics"],checks,(seqrow["sequence"],frame,"original_B_reasons"))
    accepted=[{k:v for k,v in e.items() if k not in ("event","track_origin")}
              for e in trace["events"] if e["event"]=="recovered"]
    verify_value(seqrow["b"]["matches"],accepted,(seqrow["sequence"],frame,"original_B_acceptance"))


def seq_inputs(frozen,root,seq,n,phase,protocol,features,baseline):
    orig_seq=frozen/"observed/diagnostics"/TRACKER/(seq+".jsonl.gz")
    shadow_seq=root/(phase+"_replay")/"shadow"/TRACKER/(seq+".jsonl.gz")
    expected=read(root/(phase+"_parity.json"))["sequences"][seq]["shadow_sha256"]
    if digest(shadow_seq)!=expected:
        raise ValueError("Shadow content changed after parity")
    cache=read_detection_cache(baseline/"detections"/(seq+".json.gz"))
    feature_path=features/(seq+".npz")
    hashes=protocol["feature_hashes"][seq]
    if digest(feature_path)!=hashes["features"] or digest(feature_path.with_suffix(".json"))!=hashes["metadata"]:
        raise ValueError("Frozen embedding cache differs")
    fc=FeatureCache(feature_path,cache,expected_rgb_sha=protocol["rgb_encoder_sha256"],require_thermal=False)
    memo={}
    def packets(frame):
        if frame not in memo:
            memo[frame]=fc.frame(frame-1)
        return memo[frame]
    reader=zip_longest(iter_trace(orig_seq,seq,n),shadow_frames(shadow_seq))
    return reader,packets


def freeze_verify_scope(settings, original):
    frozen=Path(settings["output_root"])
    protocol=read(frozen/"observed/protocols"/(TRACKER+".json"))
    baseline=Path(settings["baseline_root"])
    if not baseline.is_absolute():
        baseline=original/baseline
    return frozen,protocol,baseline


def counter_stats(rows, mode):
    vals=Counter((r["cohort"],r["native_reason"],r["B_reason"]) for r in rows)
    first=Counter(r["B_reason"] for r in rows if r["cohort"]=="unconsumed_low")
    first_hi=Counter(r["native_reason"] for r in rows if r["cohort"]=="unconsumed_high")
    hi_b=Counter(r["B_reason"] for r in rows if r["cohort"]=="unconsumed_high")
    return dict(status="COMPLETE",phase=mode,cohort_rows=len(rows),P0_full_stage_recomputed=True,
                groups=[dict(cohort=co,native=n,B=b,rows=v) for (co,n,b),v in sorted(vals.items())],
                unconsumed_low_B_reasons=dict(first),unconsumed_high_native_reasons=dict(first_hi),
                unconsumed_high_B_reasons=dict(hi_b))


def episode_table(rows):
    groups=defaultdict(list)
    for r in rows:
        groups[r["sequence"],r["track_id"],r["last_seen"]].append(r)
    return [dict(sequence=s,original_video=s.rsplit("_",1)[0],track_id=t,last_seen=last,
                 gt_id=v[0]["anchor_gt_id"],n=len(v),
                 distinct_GT_frames=len({(x["frame"],x["anchor_gt_id"]) for x in v}),
                 cohort_count=json.dumps(dict(Counter(x["cohort"] for x in v)),ensure_ascii=False),
                 native_reason_count=json.dumps(dict(Counter(x["native_reason"] for x in v)),ensure_ascii=False),
                 B_reason_count=json.dumps(dict(Counter(x["B_reason"] for x in v)),ensure_ascii=False))
            for (s,t,last),v in sorted(groups.items())]


def position_row(seqrow,trace,cohort,centers,unavailable,gt,high,bs,cfg):
    frame,tid=seqrow["frame"],int(cohort["track_id"])
    prior_track=next(t for t in trace["after_gmc"] if t["track_id"]==tid)
    original_box=np.asarray(prior_track["xyxy"],dtype=float)
    observed_truth=np.asarray(gt,dtype=float)
    gt_center=(observed_truth[:2]+observed_truth[2:])*.5
    index=cohort["gt_paired_det_index"]
    di=int(index) if index else None
    row={"sequence":seqrow["sequence"],"original_video":seqrow["sequence"].rsplit("_",1)[0],
         "frame":frame,"track_id":tid,"last_seen":cohort["last_seen"],
         "gt_id":cohort["anchor_gt_id"],"det_index":di,"cohort":cohort["cohort"]}
    for k,name in enumerate(VARIANTS):
        use=None if k==0 else centers.get(tid,(None,None))[k-1]
        box=moved_box(original_box,use)
        center=(box[:2]+box[2:])*.5
        row[name+"_center_error_px"]=float(np.linalg.norm(center-gt_center))
        if k:
            row[name+"_available"]=use is not None
            if use is None:
                row[name+"_fallback_reason"]=unavailable.get(tid,"missing_accepted_anchor")
        if di is not None:
            prior_det=trace["detections"][di]
            center_difference=center-(np.asarray(prior_det["xyxy"][:2])+np.asarray(prior_det["xyxy"][2:]))*.5
            row[name+"_det_center_distance_px"]=float(np.linalg.norm(center_difference))
            det_geometry=geometry_cost(box,prior_det["xyxy"],cfg.geometry,
                                       center_cov=np.asarray(prior_track["projected_covariance"])[:2,:2])
            row[name+"_current_geometry_pass"]=math.isfinite(det_geometry)
            row[name+"_current_geometry_cost"]=finite(det_geometry)
            nd=native_evidence(high[k],seqrow["high"]["threshold"],cohort["detection_tier"],tid,di)
            if nd.get("native_high_cost") is not None:
                row[name+"_native_high_feasible"]=nd["native_high_cost"]<=seqrow["high"]["threshold"]
                row[name+"_native_row_gap"]=nd["native_row_gap"]
                row[name+"_native_column_gap"]=nd["native_column_gap"]
            b=bs[k]
            col=b["columns"].get(di)
            rr=b["rows"].get(tid)
            if col is not None and rr is not None:
                j,cid=col
                row[name+"_B_raw_feasible"]=bool(np.isfinite(b["raw"][rr,j]))
                row[name+"_B_final_eligible"]=bool(np.isfinite(b["eligible"][rr,j]))
                row[name+"_B_reason"]=b["diagnostics"][(tid,cid)]["reason"]
                row[name+"_B_row_gap"],row[name+"_B_column_gap"]=margins(b["raw"],rr,j)
    return row


def comparison_summary(positions,all_changes):
    output=[]
    for version in VARIANTS[1:]:
        for stage in STAGES:
            subset=[r for r in all_changes if r["variant"]==version and r["stage"]==stage]
            scopes={"all":subset}
            for key in ("original_video", "sequence"):
                for value in sorted({r[key] for r in subset}):
                    scopes[key+":"+value]=[r for r in subset if r[key]==value]
            positives=[r for r in subset if r["direction"]=="added" and r["gt_support"]=="correct"
                       and r["interpretation"]=="candidate_without_baseline_coverage"]
            contributors=Counter(r["sequence"] for r in positives)
            if contributors:
                most=contributors.most_common(1)[0][0]
                scopes["excluding_largest_positive_sequence:"+most]=[r for r in subset if r["sequence"]!=most]
            for scope, selected in scopes.items():
                for direction in ("added","removed"):
                    for label in ("correct","wrong","unknown"):
                        rows=[r for r in selected if r["direction"]==direction and r["gt_support"]==label]
                        output.append(dict(variant=version,stage=stage,scope=scope,direction=direction,
                          gt_support=label,assignment_pairs=len(rows),sequence_count=len({r["sequence"] for r in rows}),
                          independent_gt_frames=len({(r["sequence"],r["frame"],r["anchor_gt_id"]) for r in rows if r["anchor_gt_id"] is not None}),
                          gt_tracks=len({(r["sequence"],r["anchor_gt_id"]) for r in rows if r["anchor_gt_id"] is not None}),
                          old_id_lost_episodes=len({(r["sequence"],r["track_id"],r["last_seen"]) for r in rows if r["state_before"]=="2"}),
                          candidate_without_baseline_coverage=sum(r["interpretation"]=="candidate_without_baseline_coverage" for r in rows),
                          identity_continuation_only=sum(r["interpretation"]=="identity_continuation_only" for r in rows)))
    global_stats={}
    for label in VARIANTS:
        vals=np.array([r[label+"_center_error_px"] for r in positions])
        valid="all" if label=="P0" else sum(r[label+"_available"] for r in positions)
        geom=[r[label+"_current_geometry_pass"] for r in positions if label+"_current_geometry_pass" in r]
        native=[r[label+"_native_high_feasible"] for r in positions if label+"_native_high_feasible" in r]
        valid_B=[r[label+"_B_raw_feasible"] for r in positions if label+"_B_raw_feasible" in r]
        global_stats[label]={"center_MED":float(np.median(vals)),"center_MEAN":float(np.mean(vals)),
            "available":valid,"current_geometry_pass":sum(geom),"current_geometry_denominator":len(geom),
            "native_high_pass":sum(native),"native_high_denominator":len(native),
            "B_raw_pass":sum(valid_B),"B_raw_denominator":len(valid_B)}
    return output,global_stats


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase",choices=("pilot","audit","compare"))
    cli.add_argument("--repo",required=True,type=Path)
    cli.add_argument("--attempt")
    args=cli.parse_args()
    name=args.attempt or args.phase+"_analysis"
    out=ROOT/name
    out.mkdir(exist_ok=False)
    phase="pilot" if args.phase=="pilot" else "full"
    if args.phase=="compare":
        assert read(ROOT/"audit_analysis/E1_association_summary.json")["status"]=="COMPLETE"
    original=args.repo.parent/"BOT-SORT-Plus"
    settings=read(original/"configs/diagnostics/motion_blur_v0.4.local.json")
    frozen,protocol,baseline=freeze_verify_scope(settings,original)
    cfg=PlusConfig.load(args.repo/"configs/botsort_plus/B_early_moderate.json")
    cohort=csv_groups(ROOT/"cohort/lost_cohort.csv",("frame","track_id"))
    detection=csv_groups(frozen/"analysis/detection_audit.csv",("frame","det_index"))
    track=csv_groups(frozen/"analysis/track_error.csv",("frame","track_id")) if args.phase=="compare" else {}
    byseq=read(ROOT/(phase+"_parity.json"))["sequences"]
    stages,positions,deltas=[],[],[]
    for seq,proof in byseq.items():
        n=proof["frames"]
        pairs_reader, packets=seq_inputs(frozen,ROOT,seq,n,phase,protocol,Path(settings["features_dir"]),baseline)
        rb=Replay(cfg,packets)
        pending_cohort=set(cohort[seq])
        by_frame=defaultdict(dict)
        for (frame,tid),row in cohort[seq].items():
            by_frame[frame][tid]=row
        if args.phase=="compare":
            gtgt,_=load_gt(Path(settings["gt_root"])/seq/"gt/gt.txt",n)
            gts={f:{r["gt_id"]:r["xyxy"] for r in rows} for f,rows in gtgt.items()}
            motion=MotionTransport()
            alt_hist=({}, {})
        for number,(trace,snapshot) in enumerate(pairs_reader,1):
            if trace is None or snapshot is None or trace["frame"]!=number or snapshot["frame"]!=number or snapshot["sequence"]!=seq:
                raise ValueError((seq,number,"native and shadow trace length/order differ"))
            base_identity(snapshot,trace)
            rb.frame(snapshot)
            h0=high_result(number,snapshot["high"])
            b0=rb.recovery(number)
            if args.phase!="compare":
                ev=FrameEvents(trace["events"])
                for tid,row in by_frame[number].items():
                    pending_cohort.remove((number,tid))
                    if row["gt_paired_det_index"]:
                        di=int(row["gt_paired_det_index"])
                        native=native_evidence(h0,snapshot["high"]["threshold"],row["detection_tier"],tid,di)
                        b=recovery_evidence(b0,snapshot,ev,cfg,tid,di)
                    else:
                        native={"native_reason":"no_unique_GT_candidate"}
                        b={"B_reason":"no_unique_GT_candidate"}
                    stages.append({**row,**native,**b})
                continue
            centers=motion.advance(trace)
            first={tid:pair[0] for tid,pair in centers.items() if pair[0] is not None}
            second={tid:pair[1] for tid,pair in centers.items() if pair[1] is not None}
            for history,cur in zip(alt_hist,(first,second)):
                for key in [k for k in history if k[0]<=number-cfg.recovery.window]:
                    del history[key]
                history.update({(number,tid):value for tid,value in cur.items()})
            high=[h0,high_result(number,snapshot["high"],first),high_result(number,snapshot["high"],second)]
            bs=[b0,rb.recovery(number,alt_hist[0]),rb.recovery(number,alt_hist[1])]
            for i,version in enumerate(VARIANTS[1:],1):
                for stage, p0, pn in ((STAGES[0],h0,high[i]),(STAGES[1],b0,bs[i])):
                    deltas.extend(changes(seq,number,stage,version,p0["pairs"],pn["pairs"],track[seq],detection[seq]))
            for tid,row in by_frame[number].items():
                pending_cohort.remove((number,tid))
                gt=gts[number][int(row["anchor_gt_id"])]
                positions.append(position_row(snapshot,trace,row,centers,motion.last_unavailable,gt,high,bs,cfg))
        if pending_cohort:
            raise ValueError((seq,len(pending_cohort),"uncovered E1-A rows"))
        print("E1_AUDIT_SEQUENCE",args.phase,seq,n,flush=True)
    if args.phase=="compare":
        grouped,metrics=comparison_summary(positions,deltas)
        table(out/"E1_prediction_pairs.csv",positions)
        table(out/"E1_assignment_deltas.csv",deltas)
        table(out/"E1_group_summary.csv",grouped)
        write(out/"E1_prediction_summary.json",dict(status="COMPLETE",sequences=len(byseq),positions=len(positions),assignment_deltas=len(deltas),metrics=metrics))
        print(json.dumps(metrics,ensure_ascii=False),flush=True)
    else:
        table(out/"E1_failure_stages.csv",stages)
        table(out/"E1_episode_summary.csv",episode_table(stages))
        write(out/"E1_association_summary.json",counter_stats(stages,args.phase))
    print("E1_PHASE_COMPLETE",args.phase,len(byseq),flush=True)


if __name__=="__main__":
    main()
