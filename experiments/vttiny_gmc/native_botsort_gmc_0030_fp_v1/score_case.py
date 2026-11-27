import sys
from pathlib import Path
from common_audit import ROOT,SEQ,SETTINGS,FACTOR,SKIP,FULL,read,write,csvout
from clear_audit import evaluate_rows,source,compare
from evaluate_vttiny import evaluate


def main(stage):
    root=ROOT/stage;plan=read(root/"plan.json")
    official=evaluate("rgb","test",root/"trackers",plan["methods"],root/"metrics",
                      seqs={SEQ:plan["n"]},gt_folder=Path(read(SETTINGS)["gt_root"]))
    metrics={r["tracker"]:r for r in official["results"] if r["sequence"]==SEQ}
    detections=source()
    reference=evaluate_rows(FACTOR/"full/trackers",FULL,detections)
    rows=[]
    for m in plan["methods"]:
        case=evaluate_rows(root/"trackers",m,detections)
        if stage=="trace":
            compare(reference,case,FULL,m,root/"details"/m)
        assert all(case[3][k]==metrics[m][k] for k in ("FP","FN","IDSW"))
        rows.append(dict(**metrics[m],gt2_switch_frames=",".join(str(s["frame"]) for s in case[1] if s["gt"]==2),
                         gt1_switch_frames=",".join(str(s["frame"]) for s in case[1] if s["gt"]==1),
                         fp_398_406=sum(s["FP"] for s in case[2][397:406]),
                         fp_410=case[2][409]["FP"],fp_474=case[2][473]["FP"]))
    csvout(root/"metrics_rows.csv",rows)
    write(root/"summary.json",rows)
    print(stage,"COMPLETE",len(rows),flush=True)


if __name__=="__main__":
    main(sys.argv[1])
