"""Offline verification of every accepted additional continuation match."""
from collections import Counter, defaultdict
from pathlib import Path
import csv
import json
import sys

import numpy as np

JOB=Path(__file__).resolve().parent
sys.path.insert(0,str(JOB.parent))
from run_round import FULL, GT, REPO, read, write
from compare_round2 import BASE, table
sys.path.insert(0,str(REPO))
from botsort_plus.diagnostics.common import box_iou_matrix, diagnostic_assignment, load_gt
from botsort_plus.io import read_detection_cache

BASE_PATH=FULL/"observed/trackers/B_early_moderate/data"
CACHE=REPO/"results/vttiny/yolo11m_rgb_rgbt/rgbt/detections"
STAGES={"B_postlow_v1":"post_B_weak","B_adjacent_gate":"low_adjacent_fallback"}


def events(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def boxes_by_frame(path):
    result=defaultdict(dict)
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            frame,tid=int(row[0]),int(row[1])
            x,y,w,h=map(float,row[2:6])
            result[frame][tid]=[x,y,x+w,y+h]
    return result


def diag_gt(boxes, truth):
    return diagnostic_assignment(boxes, truth, threshold=.5, ambiguity_margin=.05)[0]


def main():
    runs=read(JOB/"same_input_verification.json")["run_evidence"]
    report={}
    for method,stage in STAGES.items():
        root=Path(runs[method]["root"])
        proto=read(root/"protocols"/(method+".json"))
        summaries=Counter()
        output=[]
        for seq, length in proto["sequences"].items():
            evt=events(root/"events"/method/(seq+".jsonl"))
            attached=Counter(e["frame"] for e in evt if e["event"]=="match" and e.get("stage")==stage)
            if not attached:continue
            obs=read_detection_cache(CACHE/(seq+".json.gz"))["frames"]
            truth,_=load_gt(GT/seq/"gt/gt.txt",length)
            assigned={}
            wanted={f for f in attached}|{f-1 for f in attached}
            for f in sorted(wanted):
                det=np.asarray(obs[f-1]["detections"],dtype=np.float32).reshape(-1,6)
                assigned[f]=diag_gt(det[:,:4],truth[f])
            variant=boxes_by_frame(root/"trackers"/method/"data"/(seq+".txt"))
            original=boxes_by_frame(BASE_PATH/(seq+".txt"))
            prev={}
            latest_restore={}
            seen=set()
            for e in evt:
                if e["event"] not in ("match","recovered","native_birth"):continue
                tid,di,frame=int(e["track_id"]),int(e["det_index"]),int(e["frame"])
                if e["event"]=="recovered":
                    latest_restore[tid]=frame
                if e.get("stage")==stage and e["event"]=="match":
                    assert (tid,frame) not in seen
                    seen.add((tid,frame))
                    assert tid in latest_restore and frame-latest_restore[tid] in (1,2,3)
                    assert tid in prev and prev[tid][0]==frame-1
                    before,observed=assigned[frame-1].get(prev[tid][1]),assigned[frame].get(di)
                    if before is None or observed is None:
                        label="unknown_identity"
                    elif before==observed:
                        label="same_gt"
                    else:
                        label="different_gt"
                    gt=[g["xyxy"] for g in truth[frame] if g["gt_id"]==before]
                    after=variant[frame].get(tid)
                    score=float(obs[frame-1]["detections"][di][4])
                    orig_ids=list(original[frame])
                    base_labels=diag_gt([original[frame][i] for i in orig_ids],truth[frame])
                    post_iou=float(box_iou_matrix([after],gt)[0,0]) if gt and after is not None else None
                    summaries[label]+=1
                    summaries["emits_output"]+=after is not None
                    summaries["anchor_iou_ge_half"]+=post_iou is not None and post_iou>=.5
                    summaries["anchor_available"]+=before is not None
                    summaries["same_gt_missing_in_B"]+=label=="same_gt" and before not in base_labels.values()
                    output.append(dict(sequence=seq,frame=frame,track_id=tid,det_index=di,restore_frame=latest_restore[tid],
                       detector_score=score,previous_det_index=prev[tid][1],anchor_gt=before,current_det_gt=observed,
                       identity_support=label,emits_output=after is not None,output_iou_to_anchor=post_iou,
                       baseline_has_anchor=before in base_labels.values() if before is not None else None))
                prev[tid]=(frame,di)
        summaries["triggers"]=len(output)
        summaries["sequences"]=len({x["sequence"] for x in output})
        report[method]=dict(summaries)
        if output:table(JOB/(method+"_fallback_audit.csv"),output)
    write(JOB/"fallback_audit.json",report)
    print(json.dumps(report,ensure_ascii=False))


if __name__=="__main__":
    main()
