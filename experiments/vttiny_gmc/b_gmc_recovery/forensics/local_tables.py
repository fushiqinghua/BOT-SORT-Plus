"""Consolidate true per-detection changes, accepted updates and intervention scores."""
import csv,gzip,json
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parent/"results"
def get(path):
    return json.loads(path.read_bytes())
def rows(path):
    with path.open(encoding="utf-8-sig",newline="") as f: return list(csv.DictReader(f))
def write(path,rr):
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rr for k in r)))
        w.writeheader();w.writerows(rr)
all_scores=[]
for p in sorted(ROOT.glob("DJI_*")):
    by={}
    for m in ("B","BG"):
        with gzip.open(p/(m+"_events.jsonl.gz"),"rt",encoding="utf8") as f:
            events=[json.loads(l) for l in f]
        lookup={};seen={};last_b={};episode={}
        for e in events:
            if e["event"] not in ("match","recovered","native_birth"): continue
            t=e["frame"];tid=e["track_id"];det=e["det_index"];stage=e.get("stage",e["event"])
            gap=t-seen[tid] if tid in seen else None
            if stage=="recovered": last_b[tid]=t
            if stage in ("recovered","native_birth") or gap is not None and gap>1:
                episode[tid]=(t,stage)
            lookup[t,tid,det]=dict(stage=stage,previous_update=seen.get(tid),latest_B=last_b.get(tid),
                                   episode_start=episode.get(tid,(None,None))[0],
                                   episode_stage=episode.get(tid,(None,None))[1],cost=e.get("cost"))
            seen[tid]=t
        by[m]=lookup
    ledger=[]
    for r in rows(p/"with_B/changed_fp.csv"):
        x=dict(frame=r["frame"],det=r["det"],change=r["change"],score=r["score"],raw_near_gt=r["raw_near_gt"],raw_iou=r["raw_iou"])
        for key,side in (("B","before"),("BG","after")):
            tid=r[side+"_tid"];x[key+"_tid"]=tid;x[key+"_status"]=r[side+"_status"];x[key+"_gt"]=r[side+"_gt"]
            if not tid:continue
            orig=by[key][int(r["frame"]),int(tid),int(r["det"])]
            x.update({key+"_"+k:v for k,v in orig.items()})
        ledger.append(x)
    write(p/"FP_event_ledger.csv",ledger)
    base=get(p/"official.json")["B_early_moderate_dual_low"]
    for stage in ("initial","cause","trial","evidence","path"):
        sc=get(p/(stage+"_scores.json"))
        for r in sc["metrics"]:
            all_scores.append(dict(stage=stage,**r,**{"delta_vs_BG_"+k:r[k]-base[k] for k in ("FP","FN","IDSW","HOTA","IDF1","MOTA")}))
write(ROOT/"interventions.csv",all_scores)
if (ROOT/"full/summary.json").exists():
    allrows=get(ROOT/"full/summary.json")["results"]
    rr={(r["tracker"],r["sequence"]):r for r in allrows}
    out=[]
    for r in allrows:
        if r["tracker"] not in ("margin_both","evidence_both"):continue
        for m in ("native","dual_low","B_early_moderate","B_early_moderate_dual_low"):
            base=rr[m,r["sequence"]]
            out.append(dict(**r,ref=m,**{"delta_"+k:r[k]-base[k] for k in ("FP","FN","IDSW","HOTA","DetA","AssA","IDF1","MOTA")}))
    write(ROOT/"full/per_sequence_deltas.csv",out)
