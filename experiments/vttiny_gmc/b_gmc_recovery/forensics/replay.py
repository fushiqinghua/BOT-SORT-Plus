"""Causal tracker replays with cached, verified camera matrices. GT is only opened by score.py."""
import csv,gzip,json,sys
from collections import Counter
import cv2
import numpy as np
import torch
import replay_case as previous
from botsort_plus.io import forbid_ground_truth
from botsort_plus.config import PlusConfig
from botsort_plus.native_adapter import BoTSORTPlus
from boxmot.trackers.botsort.botsort_track import STrack
from common_audit import motions, sha
from guard import capture_priors,projected_xyxy,iou
from lost_camera_margin import LostCameraMargin
from common import ROOT, REPO, JOINT, OUT, B, COMBO, read, save


def run(seq, inputs, native, plan, directory):
    cv2.setRNGSeed(0)
    trace=plan.get("trace",False)
    name=plan["name"]
    use_full=plan.get("camera","full")
    keep=plan.get("only")
    drop=plan.get("drop",[])
    counters=Counter()
    events=[]
    class Tracker(BoTSORTPlus):
        def _cost(self,tracks,detections,unconfirmed=False):
            cost=super()._cost(tracks,detections,unconfirmed)
            if not unconfirmed and margin is not None:
                margin.high_cost(cost,tracks)
            for frame,tid,det in plan.get("high_veto",[]):
                if not unconfirmed and self.frame_count==frame:
                    for i,st in enumerate(tracks):
                        for j,d in enumerate(detections):
                            if int(st.id)==tid and int(d.det_ind)==det:
                                cost[i,j]=1.
                                counters["high_veto"]+=1
            return cost
        def guard_low_cost(self,cost,tracks,detections):
            if self.dual and not plan.get("no_guard"):
                for i,track in enumerate(tracks):
                    backup=self.backup_boxes[int(track.id)]
                    for j,det in enumerate(detections):
                        if cost[i,j]<=.5 and iou(backup,det[:4])<.5:
                            if trace:
                                events.append(dict(frame=self.frame_count,event="guard",track_id=int(track.id),det_index=int(det[6]),
                                                   full_iou=1-float(cost[i,j]),backup_iou=iou(backup,det[:4])))
                            counters["masked_edges"]+=1
                            cost[i,j]=1.
    tracker=Tracker(plus_config=PlusConfig.load(REPO/"configs/botsort_plus/B_early_moderate.json"),
                    reid_weights=ROOT/"cache-only.pt",device=torch.device("cpu"),half=False,**inputs.ref["native_config"])
    tracker.capture_trace=False
    mode=plan.get("margin")
    margin=LostCameraMargin(tracker,recovery=mode in ("B","both","evidence"),high=mode in ("high","both","evidence"),
                            require_observation=mode=="evidence") if mode else None
    if margin:
        tracker.recovery_matcher._pair=margin.recovery_pair(tracker.recovery_matcher._pair)
    def camera(img,dets):
        idx=tracker.frame_count-1;t=idx+1
        full=inputs.full[idx]; backup=inputs.backup[idx]
        sof=np.asarray(native[idx]["used"],np.float64)
        fallback=np.asarray(backup["used"],np.float64)
        use=use_full!="native" and t not in drop and (keep is None or t in keep)
        used=np.asarray(full["used"],np.float64) if use else sof
        if use_full=="skip": used=fallback
        tracker.dual=use and use_full!="skip" and full["choice"]=="lightglue" and not np.array_equal(used,fallback)
        tracker.changed=not np.array_equal(used,sof)
        if margin: margin.observe(t,sof,used)
        if tracker.dual: capture_priors(tracker,used,fallback)
        if trace:
            tracker.counterboxes={int(st.id):[projected_xyxy(st.mean,m) for m in (sof,used)]
                                  for st in [*tracker.active_tracks,*tracker.lost_stracks] if st.mean is not None}
        return used
    tracker.cmc.apply=camera
    orig=tracker.recovery_matcher.match
    ban=plan.get("veto",[])
    def match(frame,views,candidates,appearance):
        decision=orig(frame,views,candidates,appearance)
        keep_matches=[]
        for m in decision.matches:
            candidate=next(c for c in candidates if c.candidate_id==m.candidate_id)
            hist=[(o.frame,o.det_index,round(float(o.score),4)) for o in candidate.observations if o.frame>frame-5]
            if trace:
                events.append(dict(frame=frame,event="accepted_history",track_id=m.track_id,det_index=m.det_index,
                                   candidate_id=m.candidate_id,observations=hist))
            if any((frame==f or f==-1) and (m.track_id==tid or tid==-1) for f,tid in ban):
                counters["vetoed"]+=1
            else:
                keep_matches.append(m)
        decision.matches=keep_matches
        return decision
    tracker.recovery_matcher.match=match
    dest=directory/"trackers"/name/"data"/(seq+".txt")
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open("w",newline="") as f:
        writer=csv.writer(f,lineterminator="\n")
        for t,(img,dets,features) in enumerate(inputs.frames,1):
            override = plan.get("track_camera",{})
            if t == override.get("frame"):
                parent_gmc=STrack.multi_gmc
                old_descriptor=STrack.__dict__.get("multi_gmc")
                record = native[t-1] if override["use"]=="native" else inputs.full[t-1]
                alt=np.asarray(record["used"],np.float64)
                def group_gmc(pool,warp):
                    changed=[st for st in pool if int(st.id)==override["tid"]]
                    rest=[st for st in pool if int(st.id)!=override["tid"]]
                    parent_gmc(rest,warp)
                    parent_gmc(changed,alt)
                    counters["individual_pose_updates"]+=len(changed)
                STrack.multi_gmc=staticmethod(group_gmc)
                try:
                    out=tracker.update(dets,img,features=features)
                finally:
                    if old_descriptor is None: del STrack.multi_gmc
                    else: STrack.multi_gmc=old_descriptor
            else:
                out=tracker.update(dets,img,features=features)
            for row in out:
                x,y,u,v,tid,score,_,det=map(float,row)
                writer.writerow([t,int(tid),x,y,u-x,v-y,score,-1,-1,-1])
            counters["recovered"]+=sum(e["event"]=="recovered" for e in tracker.last_events)
            if trace:
                for event in tracker.last_events:
                    if event["event"] not in ("match","recovered","native_birth","recovery_check","deferred_birth","suppressed_native_birth"):
                        continue
                    info=dict(frame=t,**event)
                    tid=info.get("track_id"); det=info.get("det_index")
                    if tid in tracker.counterboxes and det is not None:
                        info["sof_iou"],info["used_iou"]=[iou(b,dets[det,:4]) for b in tracker.counterboxes[tid]]
                    events.append(info)
    if trace:
        with gzip.open(OUT/seq/(name+"_events.jsonl.gz"),"wt") as f:
            for e in events: f.write(json.dumps(e,allow_nan=False)+"\n")
    if name in ("B","BG"):
        hist=read(JOINT/"full/protocols"/((B if name=="B" else COMBO)+".json"))
        assert sha(dest)==hist["output_prediction_hashes"][seq],(seq,name,sha(dest))
    if margin:
        counters.update(recovery_pair_rejections=margin.recovery_rejections,high_pair_rejections=margin.high_rejections,
                        unsupported_rejections=margin.unsupported_rejections)
    return dict(name=name,hash=sha(dest),**counters)


def main(seq,stage):
    cv2.setNumThreads(1);torch.set_num_threads(1);torch.manual_seed(0)
    previous.SEQ=seq
    sys.addaudithook(forbid_ground_truth)
    inputs=previous.CaseInputs()
    native=motions("native",seq)
    changed=[r["frame"] for r,n in zip(inputs.full,native) if r["used"]!=n["used"]]
    if stage=="initial":
        plan=[dict(name="B",camera="native",trace=True),dict(name="BG",trace=True),
              dict(name="BG_no_guard",no_guard=True),dict(name="B_skip",camera="skip")]
    else: plan=read(ROOT/("plan_"+stage+".json")).get(seq,[])
    directory=ROOT/"runs"/stage
    result=[run(seq,inputs,native,arm,directory) for arm in plan]
    save(OUT/seq/(stage+"_replays.json"),dict(changed=changed,runs=result))


if __name__=="__main__": main(*sys.argv[1:])
