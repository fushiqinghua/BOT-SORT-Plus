"""Offline official CLEAR ownership, detection support, and attribution of delayed releases."""
import csv,json,sys
from collections import Counter
import numpy as np
import clear_audit
from evaluate_vttiny import VTTinyObjects
from trackeval.metrics import CLEAR
import trackeval.metrics.clear as clear_module
from driver import ROOT,ART,PREV,REF,FOCUS,j,save,settings
from common_audit import csvout


def owners(trackers,method,seq,n):
    dataset=VTTinyObjects(dict(GT_FOLDER=settings["gt_root"],TRACKERS_FOLDER=str(trackers),OUTPUT_FOLDER=str(ROOT/"offline"),
        TRACKERS_TO_EVAL=[method],BENCHMARK="VT-Tiny-MOT",SPLIT_TO_EVAL="test",SKIP_SPLIT_FOL=True,SEQ_INFO={seq:n},
        DO_PREPROC=False,PRINT_CONFIG=False))
    raw=dataset.get_raw_seq_data(method,seq); data=dataset.get_preprocessed_seq_data(raw,"object")
    times=iter([t for t in range(n) if len(data["gt_ids"][t]) and len(data["tracker_ids"][t])])
    ids=[{} for _ in range(n)];solve=clear_module.linear_sum_assignment
    def hook(cost):
        t=next(times)
        rows,cols=solve(cost);ok=-cost[rows,cols]>np.finfo(float).eps
        mask=np.asarray(raw["gt_extras"][t]["zero_marked"])!=0
        gt=raw["gt_ids"][t][mask]
        for a,b in zip(rows[ok],cols[ok]):
            ids[t][int(raw["tracker_ids"][t][b])]=int(gt[a])
        return rows,cols
    clear_module.linear_sum_assignment=hook
    try: result=CLEAR({"PRINT_CONFIG":False}).eval_sequence(data)
    finally: clear_module.linear_sum_assignment=solve
    assert next(times,None) is None and sum(map(len,ids))==result["CLR_TP"]
    return ids,[set(map(int,a)) for a in raw["tracker_ids"]],result

def main(stage,methods):
    focus=stage.startswith("focus")
    runs=ROOT/"runs"/("focus" if focus else "full")
    out=ART/stage;trackers=ROOT/"scores"/stage/"trackers"
    summary=j(out/"metrics/summary.json")["results"]
    reference=j(PREV/"full_runs"/REF/"protocols"/(REF+".json"))
    seqs={FOCUS:reference["sequences"][FOCUS]} if focus else reference["sequences"]
    flags=[r for r in csv.DictReader((PREV/"results"/FOCUS/"evidence/evidence_both_affected.csv").open(encoding="utf-8-sig"))
           if r["before"]=="FP" and r["after"]=="absent"]
    assert len(flags)==67
    clear_audit.SEQ=FOCUS; dets=clear_audit.source()
    by={m:clear_audit.evaluate_rows(trackers,m,dets) for m in methods}
    base=by[REF]
    target={}; gtrows=[]
    for m,actual in by.items():
        counts=actual[-1]; row=next(r for r in summary if r["sequence"]==FOCUS and r["tracker"]==m)
        assert all(counts[k]==row[k] for k in ("FP","FN","IDSW"))
        late=[]; false_rows=Counter()
        for s in flags:
            entry=actual[0][int(s["frame"])-1].get(int(s["det"]))
            false_rows[entry["status"] if entry else "absent"]+=1
        for t,rows in enumerate(actual[0],1):
            targetowner=next((v for v in rows.values() if v["gt"]==4),{})
            id7=next((v for v in rows.values() if v["tid"]==7),{})
            if t>391 and targetowner.get("tid")==7: late.append(t)
            if t>=371:
                gtrows.append(dict(method=m,frame=t,GT4_tid=targetowner.get("tid"),GT4_det=targetowner.get("det"),
                                   ID7_status=id7.get("status","absent"),ID7_gt=id7.get("gt"),ID7_det=id7.get("det")))
        target[m]=dict(late_GT4_with_ID7=late,old_67=dict(false_rows),FP=counts["FP"],FN=counts["FN"],IDSW=counts["IDSW"])
        if m!=REF:
            clear_audit.compare(base,actual,REF,m,out/"focus_diffs"/m)
    csvout(out/"GT4_ID7_frames.csv",gtrows)
    save(out/"target.json",target)
    decisions=[]; seqcounts=[]; case_events={}
    for m in methods:
        if m==REF:continue
        p=j(runs/m/"protocols"/(m+".json"))
        for seq,n in seqs.items():
            ledger=j(runs/m/"pending"/m/(seq+".json"))
            release=[e for e in ledger["events"] if e["event"]=="release"]
            if seq==FOCUS: case_events[m]=ledger
            seqcounts.append(dict(method=m,seq=seq,**{k:ledger["counts"].get(k,0) for k in ("defer","close","probe","release","probation_pass","probation_low_reject")}))
            if not release:continue
            frame_ids,emitted,counts=owners(trackers,m,seq,n)
            rr=next(r for r in summary if r["sequence"]==seq and r["tracker"]==m)
            assert all(int(counts[v])==rr[k] for v,k in (("CLR_FP","FP"),("CLR_FN","FN"),("IDSW","IDSW")))
            holds={e["tid"]:None for e in release}
            for e in ledger["events"]:
                tid=e["tid"]
                if e["event"]=="defer":holds[tid]=e
                if e["event"]!="release":continue
                t=e["frame"];hold=holds[tid];old=next((r[tid] for r in reversed(frame_ids[:hold["last_seen"]]) if tid in r),None)
                gt=frame_ids[t-1].get(tid)
                status=("not_emitted" if tid not in emitted[t-1] else "FP" if gt is None else
                        "prior_unknown" if old is None else "same_gt" if gt==old else "other_gt")
                later=frame_ids[t-1:min(t+4,n)]
                decisions.append(dict(method=m,seq=seq,**e,deferred=hold["frame"],last_seen=hold["last_seen"],
                                      prior_gt=old,current_gt=gt,outcome=status,next5_frames_same_prior=sum(old is not None and r.get(tid)==old for r in later)))
    csvout(out/"release_outcomes.csv",decisions,["method","seq","frame","tid","event","cid","det","age","waiting","reason","score",
                                                "deferred","last_seen","prior_gt","current_gt","outcome","next5_frames_same_prior"])
    csvout(out/"pending_by_sequence.csv",seqcounts);save(out/"focus_events.json",case_events)
    save(out/"audit_summary.json",{m:dict(Counter(d["outcome"] for d in decisions if d["method"]==m)) for m in methods if m!=REF})
    print(json.dumps(target),flush=True)


if __name__=="__main__":main(sys.argv[1],sys.argv[2:])
