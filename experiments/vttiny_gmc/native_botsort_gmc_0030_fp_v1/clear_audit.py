"""Capture the pinned TrackEval CLEAR assignments and pair outputs by frozen detection row."""
from collections import Counter
import json
import gzip
from pathlib import Path
import numpy as np
from common_audit import ROOT,FACTOR,SETTINGS,TRACK_EVAL,SKIP,FULL,SEQ,read,write,sha,csvout,motions


def source():
    from botsort_plus.io import read_detection_cache
    return read_detection_cache(Path(read(SETTINGS)["baseline_root"])/"detections"/(SEQ+".json.gz"))["frames"]


def evaluate_rows(folder,method,detections):
    from evaluate_vttiny import VTTinyObjects
    from trackeval.metrics import CLEAR
    import trackeval.metrics.clear as clear_module
    n=len(detections)
    gtroot=Path(read(SETTINGS)["gt_root"])
    dataset=VTTinyObjects(dict(GT_FOLDER=str(gtroot),TRACKERS_FOLDER=str(folder),
        OUTPUT_FOLDER=str(ROOT/"clear"),TRACKERS_TO_EVAL=[method],BENCHMARK="VT-Tiny-MOT",
        SPLIT_TO_EVAL="test",SKIP_SPLIT_FOL=True,SEQ_INFO={SEQ:n},DO_PREPROC=False,PRINT_CONFIG=False))
    raw=dataset.get_raw_seq_data(method,SEQ)
    data=dataset.get_preprocessed_seq_data(raw,"object")
    clear=CLEAR({"PRINT_CONFIG":False})
    captured={}
    frames=iter([t for t in range(n) if len(data["gt_ids"][t]) and len(data["tracker_ids"][t])])
    solve=clear_module.linear_sum_assignment
    def observe(cost):
        t=next(frames)
        rows,cols=solve(cost)
        valid=-cost[rows,cols]>np.finfo(float).eps
        captured[t]={int(j):int(i) for i,j in zip(rows[valid],cols[valid])}
        return rows,cols
    clear_module.linear_sum_assignment=observe
    try:
        official=clear.eval_sequence(data)
    finally:
        clear_module.linear_sum_assignment=solve
    assert next(frames,None) is None
    all_rows=[]; switches=[]; stats=[]; last={}; counts=Counter()
    for t,cache in enumerate(detections):
        mask=np.asarray(raw["gt_extras"][t]["zero_marked"])!=0
        gids=raw["gt_ids"][t][mask]
        ids=raw["tracker_ids"][t]
        boxes=data["tracker_dets"][t]; conf=raw["tracker_confidences"][t]
        truth=data["gt_dets"][t]
        assert np.array_equal(boxes,raw["tracker_dets"][t])
        assert np.array_equal(truth,raw["gt_dets"][t][mask])
        ds=np.array(cache["detections"],np.float32).reshape(-1,6)
        raw_boxes=ds[:,:4].astype(float).copy()
        raw_boxes[:,2:]-=raw_boxes[:,:2]
        sim_raw=dataset._calculate_similarities(truth,raw_boxes)
        sim=data["similarity_scores"][t]
        matched=captured.get(t,{})
        frame=t+1; outputs={}; nsw=0
        for j,(tid,box,score) in enumerate(zip(ids,boxes,conf,strict=True)):
            indices=np.flatnonzero(ds[:,4].astype(float)==score)
            assert len(indices)==1,(method,frame,j,score,indices)
            det=int(indices[0]); tid=int(tid)
            assert det not in outputs
            i=matched.get(j)
            gid=int(gids[i]) if i is not None else None
            if i is not None:
                previous=last.get(gid)
                if previous and previous[0]!=tid:
                    switches.append(dict(method=method,frame=frame,gt=gid,previous_frame=previous[1],
                                         previous_tid=previous[0],current_tid=tid,det=det,iou=float(sim[i,j])))
                    nsw+=1
                last[gid]=(tid,frame)
            near=int(np.argmax(sim[:,j])) if len(gids) and sim[:,j].max()>0 else None
            near_det=int(np.argmax(sim_raw[:,det])) if len(gids) and sim_raw[:,det].max()>0 else None
            outputs[det]=dict(tid=tid,det=det,score=float(score),status="TP" if i is not None else "FP",gt=gid,
                box=list(map(float,box)),near_gt=int(gids[near]) if near is not None else None,
                max_iou=float(sim[:,j].max()) if len(gids) else 0.,
                raw_near_gt=int(gids[near_det]) if near_det is not None else None,
                raw_iou=float(sim_raw[:,det].max()) if len(gids) else 0.,raw_xywh=raw_boxes[det].tolist(),
                truth_xywh=truth[near].tolist() if near is not None else None)
        tp=len(matched)
        row=dict(frame=frame,TP=tp,FP=len(ids)-tp,FN=len(gids)-tp,IDSW=nsw)
        stats.append(row)
        counts.update({k:v for k,v in row.items() if k!="frame"})
        all_rows.append(outputs)
    assert all(counts[k]==int(official[f]) for k,f in
               {"TP":"CLR_TP","FP":"CLR_FP","FN":"CLR_FN","IDSW":"IDSW"}.items()),(method,counts,official)
    return all_rows,switches,stats,dict(counts)


def compare(left,right,left_name,right_name,seqdir):
    a,sa,fa,ca=left;b,sb,fb,cb=right
    changed=[];allrows=[];kinds=Counter();deltas=[]
    for i,(p,q) in enumerate(zip(a,b,strict=True)):
        for det in sorted(p.keys()|q.keys()):
            before,after=p.get(det,{}),q.get(det,{})
            r=dict(frame=i+1,det=det,change=(before.get("status","absent")+"->"+after.get("status","absent")))
            for prefix,x in (("before",before),("after",after)):
                r.update({prefix+"_"+k:x.get(k) for k in ("tid","gt","status","max_iou","near_gt","box")})
            r.update({k:(after or before)[k] for k in ("score","raw_iou","raw_near_gt","raw_xywh")})
            allrows.append(r)
            if (before.get("status")=="FP") != (after.get("status")=="FP"):
                changed.append(r);kinds[r["change"]]+=1
        deltas.append(dict(frame=i+1,**{f"{m}_{k}":x[k] for m,x in ((left_name,fa[i]),(right_name,fb[i])) for k in ca},
                           **{"delta_"+k:fb[i][k]-fa[i][k] for k in ca}))
    csvout(seqdir/"detection_pairs.csv",allrows)
    csvout(seqdir/"changed_fp.csv",changed,allrows[0].keys())
    csvout(seqdir/"frames.csv",deltas)
    csvout(seqdir/"switches.csv",sa+sb,["method","frame","gt","previous_frame","previous_tid","current_tid","det","iou"])
    gross_in=sum(v for k,v in kinds.items() if k.endswith("->FP"))
    gross_out=sum(v for k,v in kinds.items() if k.startswith("FP->"))
    assert gross_in-gross_out==cb["FP"]-ca["FP"]
    summary=dict(before_name=left_name,after_name=right_name,before=ca,after=cb,gross_in=gross_in,
                 gross_out=gross_out,delta_fp=gross_in-gross_out,kinds=dict(kinds),switches=sa+sb)
    write(seqdir/"summary.json",summary)
    return summary


def main():
    assert sha(TRACK_EVAL/"trackeval/metrics/clear.py")=="da78468866f61ecf84e197446e630382e0c976ca1f47b70e1dd4904ab7ba598e"
    detections=source()
    seqdir=ROOT/"diagnosis"
    results={m:evaluate_rows(FACTOR/"full/trackers",m,detections) for m in (SKIP,FULL)}
    frozen={r["tracker"]:r for r in read(FACTOR/"full/metrics/summary.json")["results"] if r["sequence"]==SEQ}
    for m,(_,_,_,totals) in results.items():
        assert all(totals[k]==frozen[m][k] for k in ("FP","FN","IDSW"))
    summary=compare(results[SKIP],results[FULL],SKIP,FULL,seqdir)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
