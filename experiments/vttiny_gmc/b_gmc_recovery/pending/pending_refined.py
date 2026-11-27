"""Bounded dormant identity, verified residual tracklet and guarded return to tracking."""
from copy import deepcopy
import math
import numpy as np

from pending import PendingIdentity, Hypothesis
from botsort_plus.geometry import geometry_cost, box_iou, warp_box
from botsort_plus.recovery import RecoveryMatch


class DeferredEvidence(PendingIdentity):
    def __init__(self, tracker, shadow=False, **kwargs):
        super().__init__(tracker, temporal=True, probation=True)
        self.shadow=shadow
        self.warps={}
        pair=tracker.recovery_matcher._pair
        def probation_pair(frame,view,candidate,history,appearance):
            info=pair(frame,view,candidate,history,appearance)
            floor=self.provisional.get(int(view.track_id))
            if floor is None or info["reason"]!="plausible": return info
            app=info.get("appearance_cost")
            supported=(candidate.latest.score>tracker.track_high_thresh or
                       (app is not None and app<=self.cfg.appearance_max_cost-self.cfg.ambiguity_margin))
            if not supported and (candidate.latest.score<floor or not self.stable(history,frame,floor)):
                return {**info,"reason":"probation_weak_recovery"}
            return info
        tracker.recovery_matcher._pair=probation_pair

    def stable(self, observations, frame, floor):
        g=list(observations)[-self.cfg.min_hits:]
        if floor is None or len(g)!=self.cfg.min_hits or g[0].frame!=frame-len(g)+1 or g[-1].frame!=frame: return False
        if g[-1].score<floor or np.median([o.score for o in g])<floor: return False
        return all(b.frame==a.frame+1 and b.frame in self.warps and
                   box_iou(warp_box(a.box,self.warps[b.frame]),b.box)>=.5 for a,b in zip(g,g[1:]))

    def bridge_cost(self, predicted, observed, gap):
        geometry=self.tracker.plus.geometry
        p=predicted[2:]-predicted[:2];q=observed[2:]-observed[:2]
        if min(*p,*q)<=0:return math.inf
        ratios=np.maximum(p/q,q/p)
        if ratios.max()>geometry.max_scale_ratio:return math.inf
        base=max(geometry.pixel_floor,geometry.center_gate_scale*min(np.sqrt(p.prod()),np.sqrt(q.prod())))
        radius=min(geometry.max_center_pixels,base*math.sqrt(gap))
        err=np.linalg.norm((predicted[:2]+predicted[2:]-observed[:2]-observed[2:])*.5)
        size=float(np.log(ratios).mean()/np.log(geometry.max_scale_ratio))
        return float((1-geometry.size_weight)*err/radius+geometry.size_weight*size)

    def step(self,frame,warp,eligible,candidates,decision,activated,refound,appearance):
        cfg=self.cfg; safe=cfg.max_cost-cfg.ambiguity_margin
        self.warps[frame]=warp.copy()
        self.warps.pop(frame-cfg.window,None)
        self.private_detections={}
        busy={int(t.id) for t in [*activated,*refound]}|{int(m.track_id) for m in decision.matches}
        occupied={int(m.candidate_id) for m in decision.matches}
        available={int(c.candidate_id):c for c in candidates if c.candidate_id not in occupied}
        cells=[i for i in decision.diagnostics if i.get("cost") is not None and i["cost"]<cfg.max_cost]
        self.opened={k:v for k,v in self.opened.items() if frame-v<=2*cfg.max_age}
        for tid,slot in list(self.slots.items()):
            if tid in busy or slot.cid in occupied:self.cancel(frame,tid,"ordinary_assignment")
            elif frame-slot.created>cfg.max_age:self.cancel(frame,tid,"deadline")
            else:slot.box=warp_box(slot.box,warp)
        # A larger displacement after misses permits a private proposal only after observing a short tracklet.
        edges=[]
        for tid,s in self.slots.items():
            for cid,c in available.items():
                gap=frame-s.last_hit
                history=list(c.observations)[-cfg.min_hits:]
                app=appearance(s.memory,c.latest.features)
                if app is not None and app>cfg.appearance_max_cost:continue
                strong=c.latest.score>self.tracker.track_high_thresh or (app is not None and app<=cfg.appearance_max_cost-cfg.ambiguity_margin)
                temporal=s.seed_cost<=safe and self.stable(history,frame,s.score_floor)
                normal=geometry_cost(s.box,c.latest.box,self.tracker.plus.geometry)
                bridge=(normal>safe and gap>self.tracker.plus.birth.max_misses and len(history)==cfg.min_hits and
                        history[0].frame>s.last_hit and history[0].frame==frame-cfg.min_hits+1 and (strong or temporal))
                cost=self.bridge_cost(s.box,c.latest.box,gap) if bridge else normal
                if cost<=cfg.max_cost:edges.append((cost,tid,cid,bridge,strong,temporal,app))
        probed_ids=set()
        for cost,tid,cid,bridge,strong,temporal,app in sorted(edges,key=lambda e:e[:3]):
            if tid in probed_ids or tid not in self.slots or cid not in available or cost>safe:continue
            if any((j==tid)!=(k==cid) and d<=cost+cfg.ambiguity_margin for d,j,k,*_ in edges):continue
            probed_ids.add(tid)
            s=self.slots[tid];c=available.pop(cid);obs=c.latest
            overlap=box_iou(s.box,obs.box);gap=frame-s.last_hit
            s.hits=[h for h in s.hits if h[0]>frame-cfg.window]
            s.hits.append((frame,obs.score,overlap))
            enough=len(s.hits)>=cfg.min_hits or bridge
            contested=any(y["candidate_id"]==cid and y["track_id"]!=tid and y["reason"] in ("plausible","accepted","ambiguous") for y in cells)
            recent=list(c.observations)[-cfg.min_hits:]
            # Temporal evidence must postdate deferral. A single reidentified high-quality frame may confirm an already observed probe.
            ready=enough and not contested and (strong or (temporal and recent[0].frame>s.created))
            self.log(frame,tid,"probe",cid=cid,det=int(obs.det_index),cost=cost,iou=overlap,score=obs.score,
                     strong=bool(strong),stable=bool(temporal),bridge=bool(bridge),contested=contested,appearance=app,gap=gap)
            self.private_detections[obs.det_index]=tid
            if ready:
                if not self.shadow:
                    track=eligible.get(tid,s.track)
                    if tid not in eligible:self.tracker.lost_stracks=[x for x in self.tracker.lost_stracks if int(x.id)!=tid]
                    predicted=warp_box(recent[-2].box,warp) if bridge else s.box
                    prior=np.r_[(predicted[:2]+predicted[2:])*.5,predicted[2:]-predicted[:2]]
                    track.mean,track.covariance=track.kalman_filter.initiate(prior)
                    eligible[tid]=track
                    self.tracker.removed_stracks=[x for x in self.tracker.removed_stracks if int(x.id)!=tid]
                    if tid not in self.tracker.memories and s.memory is not None:self.tracker.memories[tid]=s.memory
                    decision.matches.append(RecoveryMatch(tid,cid,obs.det_index,cost,cost,app,frame-s.last_seen,len(recent)))
                    decision.reserved_candidate_ids.discard(cid)
                    if not strong:self.provisional[tid]=s.score_floor
                self.log(frame,tid,"would_release" if self.shadow else "release",cid=cid,det=obs.det_index,
                         waiting=frame-s.created,age=frame-s.last_seen,reason="strong" if strong else "temporal",score=obs.score)
                del self.slots[tid];busy.add(tid)
            else:
                s.last_hit=frame;s.cid=cid;s.box=obs.box.copy()
        for item in sorted(cells,key=lambda x:x["cost"]):
            if item["reason"] not in ("weak_observations","uncertain_camera_recovery") or item["hits"]<cfg.min_hits:continue
            tid=int(item["track_id"]);cid=int(item["candidate_id"])
            if tid in self.slots or tid in busy or tid not in eligible or cid not in available:continue
            if any(s.cid==cid for s in self.slots.values()) or self._rival(cells,tid,cid)<=item["cost"]+cfg.ambiguity_margin:continue
            tr=eligible[tid];seen=int(tr.end_frame)
            if self.opened.get(tid)==seen:continue
            c=available.pop(cid);hist=self.history.get(tid,())
            floor=self.provisional.get(tid)
            if floor is None and len(hist)==cfg.window:floor=float(np.median([score for _,score in hist]))
            s=Hypothesis(tid,seen,frame,frame,cid,c.latest.box.copy(),deepcopy(tr),deepcopy(self.tracker.memories.get(tid)),floor)
            s.seed_cost=float(item["cost"])
            self.slots[tid]=s;self.opened[tid]=seen;self.private_detections[c.latest.det_index]=tid
            self.log(frame,tid,"defer",cid=cid,det=c.latest.det_index,last_seen=seen,score_floor=floor,reason=item["reason"],cost=s.seed_cost)
