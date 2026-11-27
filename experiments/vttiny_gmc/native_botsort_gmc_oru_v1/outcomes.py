"""Offline diagnostics. Use only after online outputs are closed."""
from collections import Counter,defaultdict
import csv
import gzip
import json
import sys
from pathlib import Path
import numpy as np
from botsort_plus.diagnostics.common import diagnostic_assignment,load_gt
from paths import ROOT,SETTINGS,METHODS,FOCUS,read,save

METRICS=("HOTA","DetA","AssA","IDF1","MOTA","FP","FN","IDSW")
HORIZONS=(3,5,10)


def csvout(file,rows):
    if not rows:
        return
    with file.open("w",newline="",encoding="utf-8-sig") as s:
        w=csv.DictWriter(s,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows)


def logrows(root,method,kind,seq):
    with gzip.open(root/"audit"/method/kind/(seq+".jsonl.gz"),"rt") as s:
        return [json.loads(line) for line in s]


def labels(path,truth):
    boxes=defaultdict(list)
    with path.open() as s:
        for r in csv.reader(s):
            f,tid=int(r[0]),int(r[1])
            x,y,w,h=map(float,r[2:6]);boxes[f].append((tid,[x,y,x+w,y+h]))
    byid,bygt,amb={},{},{}
    for f,gts in truth.items():
        det=boxes[f]
        confirmed,status,iou,assigned=diagnostic_assignment([d[1] for d in det],gts)
        byid[f]={det[i][0]:gid for i,gid in confirmed.items()}
        bygt[f]={gid:tid for tid,gid in byid[f].items()}
        amb[f]=set()
        for i,s in enumerate(status):
            if s=="ambiguous":
                amb[f].update(gts[j]["gt_id"] for j in np.flatnonzero(iou[i]>=.5))
    return byid,bygt,amb


def follow(lab,gid,tid,frame,n):
    byid,bygt,amb=lab
    if tid is None:
        return "anchor_unresolved"
    for f in range(frame,frame+n+1):
        if byid[f].get(tid)==gid:
            continue
        if gid in bygt[f] or tid in byid[f]:
            return "different_id"
        if gid in amb[f]:
            return "ambiguous"
        return "no_clear_output"
    return "held"


def analyze_sequence(root,seq,nframes):
    gt,_=load_gt(Path(read(SETTINGS)["gt_root"])/seq/"gt/gt.txt",nframes)
    centers={f:{r["gt_id"]:(np.array(r["xyxy"][:2])+r["xyxy"][2:])/2 for r in rows}
             for f,rows in gt.items()}
    labs={m:labels(root/"trackers"/m/"data"/(seq+".txt"),gt) for m in ("native","oru")}
    windows,events,matched=[],[],[]
    for method in ("native","oru"):
        byid=labs[method][0]
        for r in logrows(root,method,"recoveries",seq):
            t,f,tid=r["frame"],r["old_frame"],r["tid"]
            source=byid[f].get(tid);end=byid[t].get(tid)
            state=("same_gt" if source is not None and source==end else
                   "different_gt" if source is not None and end is not None else "unresolved")
            events.append(dict(sequence=seq,method=method,**r,anchor_gt=source,recovery_gt=end,status=state))
            if state!="same_gt":
                continue
            for n in HORIZONS:
                if t+n>nframes or any(source not in centers[k] for k in range(t,t+n+1)):
                    continue
                outcome=follow(labs[method],source,tid,t,n)
                windows.append(dict(sequence=seq,method=method,anchor=f,frame=t,gt=source,
                                    track=tid,gap=r["gap"],horizon=n,outcome=outcome))
                if method=="native":
                    otid=labs["oru"][1][f].get(source)
                    other=follow(labs["oru"],source,otid,t,n)
                    matched.append(dict(sequence=seq,anchor=f,frame=t,gt=source,gap=r["gap"],
                           horizon=n,native_id=tid,oru_id=otid,native=outcome,oru=other))
    warps={m:logrows(root,m,"warps",seq) for m in METHODS}
    assert all(len(v)==nframes for v in warps.values())
    summary=Counter(frames=nframes)
    for i,(base,adapt,oru) in enumerate(zip(*(warps[m] for m in METHODS)),1):
        assert base["frame"]==adapt["frame"]==oru["frame"]==i
        assert base["used"]==base["native"]==oru["native"]==oru["used"]==adapt["native"],(seq,i)
        summary[adapt["choice"]]+=1
        summary["trigger"]+=int(adapt["trigger"])
        summary["with_validation"]+=int("native_error" in adapt)
        summary["lg_valid_fit"]+=int("lightglue" in adapt.get("candidate_errors",{}))
    positions=[]
    if root.name=="smoke":
        pre=logrows(root,"native","prestate",seq)
        assert len(pre)==nframes
        for pool,a in zip(pre,warps["adaptive_gmc"]):
            t=pool["frame"]
            if a["choice"]=="sof":
                continue
            base,new=np.array(a["native"]),np.array(a["used"])
            for tid,f,x,y in pool["tracks"]:
                if f!=t-1:
                    continue
                gid=labs["native"][0][f].get(tid)
                if gid not in centers[t]:
                    continue
                q=centers[t][gid]; point=np.array([x,y,1.])
                olderr,newerr=(float(np.linalg.norm(mat@point-q)) for mat in (base,new))
                positions.append(dict(sequence=seq,frame=t,gt=gid,track=tid,choice=a["choice"],
                                      native_px=olderr,adaptive_px=newerr,delta_px=newerr-olderr))
    return windows,events,matched,dict(sequence=seq,**summary),positions


def tally(rows):
    answer={}
    for n in HORIZONS:
        selected=[r for r in rows if r["horizon"]==n]
        a=sum(r["native"]=="held" for r in selected)
        b=sum(r["oru"]=="held" for r in selected)
        gain=sum(r["native"]!="held" and r["oru"]=="held" for r in selected)
        loss=sum(r["native"]=="held" and r["oru"]!="held" for r in selected)
        assert b-a==gain-loss
        answer[n]=dict(eligible=len(selected),native=a,oru=b,gained=gain,lost=loss,
                       oru_outcomes=dict(Counter(r["oru"] for r in selected)))
    return answer


def main(stage):
    root=ROOT/stage
    dest=root/"diagnostic_results";dest.mkdir(exist_ok=True)
    windows,events,matched,gmc,positions=[],[],[],[],[]
    raw=read(root/"metrics/summary.json")["results"]
    seqs=read(root/"protocols/native.json")["sequences"]
    for seq,nframes in seqs.items():
        w,e,p,g,points=analyze_sequence(root,seq,nframes)
        windows+=w;events+=e;matched+=p;gmc.append(g);positions+=points
    for name,rows in [("recovery_events",events),("own_windows",windows),("paired_windows",matched),
                      ("gmc_frames",gmc),("gmc_paired_centers",positions)]:
        csvout(dest/(name+".csv"),rows)
    own={m:{n:dict(Counter(r["outcome"] for r in windows if r["method"]==m and r["horizon"]==n))
            for n in HORIZONS} for m in ("native","oru")}
    counts={m:dict(total=sum(r["method"]==m for r in events),
        categories=dict(Counter(r["status"] for r in events if r["method"]==m)),
        replayed=sum(r["method"]==m and r["virtual"]>0 for r in events),
        numerical_fallback=sum(r["method"]==m and r.get("fallback",False) for r in events))
        for m in ("native","oru")}
    joint=read(root/"metrics/comparison.json")["results"]
    focus={s:[r for r in raw if r["sequence"]==s] for s in FOCUS}
    ctr={}
    for s in FOCUS:
        d=[r["delta_px"] for r in positions if r["sequence"]==s]
        ctr[s]=dict(n=len(d),improved_1px=sum(x < -1 for x in d),worse_1px=sum(x>1 for x in d),
                    mean_delta=float(np.mean(d)) if d else None,median_delta=float(np.median(d)) if d else None)
    deltas=[]
    for s in [*seqs,"COMBINED_SEQ"]:
        result={r["tracker"]:r for r in raw if r["sequence"]==s}
        for m in METHODS:
            row=dict(sequence=s,method=m)
            row.update({k:result[m][k] for k in METRICS})
            row.update({"delta_"+k:result[m][k]-result["native"][k] for k in METRICS})
            deltas.append(row)
    csvout(dest/"sequence_metrics.csv",deltas)
    gtotal=Counter()
    for g in gmc:
        gtotal.update({k:v for k,v in g.items() if k!="sequence"})
    counts_by_seq={s:tally([r for r in matched if r["sequence"]==s]) for s in seqs}
    save(dest/"paired_by_sequence.json",counts_by_seq)
    summary=dict(stage=stage,parity=read(root/"parity.json"),metrics=joint,focus_metrics=focus,
                 fixed_windows=tally(matched),own_windows=own,recoveries=counts,gmc_total=dict(gtotal),
                 gmc_focus=[g for g in gmc if g["sequence"] in FOCUS],
                 gmc_paired_centers=ctr if stage=="smoke" else {},focus_windows={s:counts_by_seq[s] for s in FOCUS})
    save(ROOT/(stage+"_summary.json"),summary)
    print("offline complete:",stage,"events",counts,"gmc",dict(gtotal),flush=True)


if __name__=="__main__":
    main(sys.argv[1])
