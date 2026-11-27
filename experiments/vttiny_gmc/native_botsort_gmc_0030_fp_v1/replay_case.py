"""Frozen-camera interventions at one sequence. GT is scored in a separate process."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import cv2
import numpy as np
import torch
from common_audit import ROOT,FACTOR,REPO,SETTINGS,SEQ,SKIP,FULL,read,write,sha,csvout,motions,env
from botsort_plus.io import (FeatureCache,forbid_ground_truth,read_detection_cache,
                             detection_fingerprint,resolve_image)
from botsort_plus.config import PlusConfig
from botsort_plus.native_adapter import BoTSORTPlus
from guard import POLICIES,LowMatchGuard,capture_priors,decide,iou


class CaseInputs:
    def __init__(self):
        cfg=read(SETTINGS)
        self.ref=read(FACTOR/"full/protocols/native.json")
        detfile=Path(cfg["baseline_root"])/"detections"/(SEQ+".json.gz")
        cache=read_detection_cache(detfile)
        assert detection_fingerprint(cache)==self.ref["detection_fingerprints"][SEQ]
        feat=Path(cfg["features_dir"])/(SEQ+".npz")
        assert sha(feat)==self.ref["feature_hashes"][SEQ]["features"]
        assert sha(feat.with_suffix(".json"))==self.ref["feature_hashes"][SEQ]["metadata"]
        fc=FeatureCache(feat,cache,expected_rgb_sha=self.ref["rgb_encoder_sha256"])
        self.frames=[]; digest=hashlib.sha256()
        for i,row in enumerate(cache["frames"]):
            image=cv2.imread(str(resolve_image(row["image"],Path(cfg["images_root"]))))
            assert image.shape==(512,640,3)
            d=np.asarray(row["detections"],np.float32).reshape(-1,6);d[:,5]=0
            pixel=hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
            digest.update((json.dumps(dict(frame=i+1,shape=list(image.shape),dtype=str(image.dtype),sha256=pixel),
                                      sort_keys=True,separators=(",",":"))+"\n").encode())
            self.frames.append((image,d,fc.frame(i)))
        assert digest.hexdigest()==self.ref["decoded_image_hashes"][SEQ]
        self.full=motions(FULL,SEQ);self.backup=motions(SKIP,SEQ)
        self.changed=[f["frame"] for f,b in zip(self.full,self.backup,strict=True) if f["used"]!=b["used"]]
        assert all(self.full[t-1]["choice"]=="lightglue" for t in self.changed)
        self.config=PlusConfig.load(REPO/"configs/botsort_plus/12_baseline_ref.json")
        assert not self.config.enabled


def run_arm(inputs,name,mode,base_dir,trace=False,frame=None):
    cv2.setRNGSeed(0)
    class CaseTracker(LowMatchGuard,BoTSORTPlus):
        pass
    tracker=CaseTracker(plus_config=inputs.config,reid_weights=Path("cache-only.pt"),
                        device=torch.device("cpu"),half=False,**inputs.ref["native_config"])
    tracker.case_policy=mode;tracker.capture_trace=trace;tracker.edges=[]
    def pick(image,dets):
        t=tracker.frame_count
        full=inputs.full[t-1]; skipped=inputs.backup[t-1]
        a=np.asarray(full["used"],np.float64);b=np.asarray(skipped["used"],np.float64)
        if mode==SKIP or (mode=="drop_one" and t==frame) or (mode=="only_one" and t!=frame):
            used=b
        else:
            used=decide(mode,a,b,full)
        tracker.matched_camera=full["choice"]=="lightglue" and not np.array_equal(used,b)
        if trace or (mode=="dual_low" and tracker.matched_camera):
            capture_priors(tracker,used,b)
        return used
    tracker.cmc.apply=pick
    path=base_dir/"trackers"/name/"data"/(SEQ+".txt")
    path.parent.mkdir(parents=True,exist_ok=True)
    details=[]
    with path.open("w",newline="") as f:
        w=csv.writer(f,lineterminator="\n")
        for t,(image,dets,feats) in enumerate(inputs.frames,1):
            pred=tracker.update(dets,image,features=feats)
            for row in pred:
                x,y,u,v,tid,score,_,det=map(float,row);tid=int(tid);det=int(det)
                w.writerow([t,tid,x,y,u-x,v-y,score,-1,-1,-1])
                if trace:
                    before=tracker.priors.get(tid)
                    details.append(dict(frame=t,tid=tid,det=det,score=score,previous_frame=before[1] if before else None,
                        previous_state=before[2] if before else None,was_confirmed=before[3] if before else None,
                        used_prior_iou=iou(before[0],dets[det,:4]) if before else None,
                        backup_prior_iou=iou(tracker.backup_boxes[tid],dets[det,:4]) if before else None,
                        gmc_lightglue=int(tracker.matched_camera)))
    if trace:
        csvout(base_dir/"trace"/(name+"_low_edges.csv"),tracker.edges)
        csvout(base_dir/"trace"/(name+"_outputs.csv"),details)
    if mode in (SKIP,FULL):
        old=read(FACTOR/"full/protocols"/(mode+".json"))
        assert sha(path)==old["output_prediction_hashes"][SEQ],(name,"reference replay mismatch",sha(path))
    print(name,sha(path)[:12],flush=True)


def main(stage):
    sys.addaudithook(forbid_ground_truth)
    cv2.setNumThreads(1);torch.set_num_threads(1)
    inputs=CaseInputs()
    path=ROOT/stage
    if stage=="trace":
        arms=[(SKIP,SKIP,True,None),(FULL,FULL,True,None)]+[(m,m,True,None) for m in POLICIES]
    else:
        arms=[(f"{mode}_{t}",mode,False,t) for t in inputs.changed for mode in ("drop_one","only_one")]
    for name,mode,trace,t in arms:
        run_arm(inputs,name,mode,path,trace,t)
    write(path/"plan.json",dict(methods=[a[0] for a in arms],n=len(inputs.frames),changed_frames=inputs.changed))
    subprocess.run([sys.executable,"-Bu",str(ROOT/"score_case.py"),stage],env=env(),check=True)


if __name__=="__main__":
    main(sys.argv[1])
