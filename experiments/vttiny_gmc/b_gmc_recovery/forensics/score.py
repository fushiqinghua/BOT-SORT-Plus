import sys
import clear_audit as audit
from common import OUT,ROOT,read,save
from common_audit import csvout
from evaluate_vttiny import evaluate

def main(seq,stage):
    audit.SEQ=seq
    dets=audit.source()
    plan=read(OUT/seq/(stage+"_replays.json"))
    directory=ROOT/"runs"/stage
    names=[r["name"] for r in plan["runs"]]
    if not names: return
    info=evaluate("rgb","test",directory/"trackers",names,directory/"metrics"/seq,seqs={seq:len(dets)},
                  gt_folder=audit.Path(audit.read(audit.SETTINGS)["gt_root"]))
    rows=[];switches=[]
    comparison=audit.evaluate_rows(ROOT/"runs/initial/trackers","BG",dets) if stage!="initial" else None
    for name in names:
        r=next(r for r in info["results"] if r["tracker"]==name and r["sequence"]==seq)
        res=audit.evaluate_rows(directory/"trackers",name,dets)
        assert all(r[k]==res[-1][k] for k in ("FP","FN","IDSW"))
        rows.append(r);switches.extend(res[1])
        if stage!="initial":
            affected=[]
            for t,(a,b) in enumerate(zip(comparison[0],res[0]),1):
                for det in a.keys()|b.keys():
                    before=a.get(det,{});after=b.get(det,{})
                    if (before.get("status"),before.get("gt"))!=(after.get("status"),after.get("gt")):
                        affected.append(dict(frame=t,det=det,before=before.get("status","absent"),after=after.get("status","absent"),
                                             old_tid=before.get("tid"),new_tid=after.get("tid"),old_gt=before.get("gt"),new_gt=after.get("gt")))
            csvout(OUT/seq/stage/(name+"_affected.csv"),affected,["frame","det","before","after","old_tid","new_tid","old_gt","new_gt"])
    save(OUT/seq/(stage+"_scores.json"),dict(metrics=rows,switches=switches))


if __name__=="__main__": main(*sys.argv[1:])
