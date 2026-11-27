"""Read-only private hypotheses for a deferred B assignment. No sequence or GT inputs."""
from collections import Counter, deque
from copy import deepcopy
from dataclasses import dataclass, field
import numpy as np

from botsort_plus.geometry import box_iou, geometry_cost, warp_box
from botsort_plus.recovery import RecoveryMatch


@dataclass
class Hypothesis:
    tid: int
    last_seen: int
    created: int
    last_hit: int
    cid: int
    box: np.ndarray
    track: object
    memory: object
    score_floor: float | None
    hits: list = field(default_factory=list)


class PendingIdentity:
    def __init__(self, tracker, temporal=False, probation=False):
        self.tracker = tracker
        self.temporal = temporal
        self.probation = probation
        self.cfg = tracker.plus.recovery
        self.slots = {}
        self.opened = {}
        self.history = {}
        self.provisional = {}
        self.private_detections = {}
        self.events = []
        self.counts = Counter()

    def log(self, frame, tid, kind, **values):
        self.counts[kind] += 1
        self.events.append(dict(frame=int(frame),tid=int(tid),event=kind,**values))

    def cancel(self, frame, tid, reason):
        slot = self.slots.pop(tid)
        self.log(frame,tid,"close",reason=reason,start=slot.created)

    def record_outputs(self, rows, detections):
        t=self.tracker.frame_count
        for row in rows:
            tid=int(row[4]); idx=int(row[7])
            holder=self.private_detections.get(idx)
            if holder in self.slots and tid!=holder:
                self.cancel(t,holder,"observation_claimed")
            self.history.setdefault(tid,deque(maxlen=self.cfg.window)).append((t,float(detections[idx,4])))
            if tid in self.provisional and detections[idx,4]>self.tracker.track_high_thresh:
                self.provisional.pop(tid)
                self.log(t,tid,"probation_pass")
        for tid in list(self.history):
            if t-self.history[tid][-1][0]>self.cfg.max_age and tid not in self.slots:
                del self.history[tid]
        for tid in list(self.provisional):
            if tid not in self.history and tid not in self.slots: self.provisional.pop(tid)

    def guard_low(self, cost, tracks, detections):
        if not self.probation: return
        for i,track in enumerate(tracks):
            floor=self.provisional.get(int(track.id))
            if floor is None: continue
            for j,det in enumerate(detections):
                if cost[i,j]<=.5 and float(det[4])<floor:
                    cost[i,j]=1.
                    self.log(self.tracker.frame_count,track.id,"probation_low_reject",det=int(det[6]))

    def _rival(self, cells, tid, cid):
        rivals=[y for y in cells if (y["track_id"]==tid)!=(y["candidate_id"]==cid)]
        return min((y["cost"] for y in rivals),default=float("inf"))

    def step(self, frame, warp, eligible, candidates, decision, activated, refound, appearance):
        cfg=self.cfg
        self.private_detections={}
        busy_ids={int(t.id) for t in [*activated,*refound]} | {int(m.track_id) for m in decision.matches}
        occupied={int(m.candidate_id) for m in decision.matches}
        available={int(c.candidate_id):c for c in candidates if c.candidate_id not in occupied}
        cells=[item for item in decision.diagnostics if item.get("cost") is not None and item["cost"] < cfg.max_cost]
        self.opened={k:v for k,v in self.opened.items() if frame-v<=2*cfg.max_age}
        for tid,slot in list(self.slots.items()):
            if tid in busy_ids or slot.cid in occupied:
                self.cancel(frame,tid,"ordinary_assignment")
            elif frame-slot.created>cfg.max_age or frame-slot.last_hit>cfg.window+1:
                self.cancel(frame,tid,"expired")
            else:
                slot.box=warp_box(slot.box,warp)

        # Match private probes only to residual observations, protecting both axes against ambiguity.
        edges=[]
        for tid,slot in self.slots.items():
            for cid,c in available.items():
                cost=geometry_cost(slot.box,c.latest.box,self.tracker.plus.geometry)
                if cost <= cfg.max_cost:
                    edges.append((cost,tid,cid))
        for cost,tid,cid in sorted(edges):
            if tid not in self.slots or cid not in available or cost>cfg.max_cost-cfg.ambiguity_margin: continue
            if any((j==tid)!=(k==cid) and d<=cost+cfg.ambiguity_margin for d,j,k in edges): continue
            slot=self.slots[tid]; cand=available.pop(cid); obs=cand.latest
            overlap=box_iou(slot.box,obs.box)
            app=appearance(slot.memory,obs.features)
            if app is not None and app>cfg.appearance_max_cost:
                self.cancel(frame,tid,"appearance_conflict")
                continue
            slot.cid=cid; slot.last_hit=frame
            self.private_detections[int(obs.det_index)]=tid
            slot.hits=[x for x in slot.hits if x[0]>frame-cfg.window]
            slot.hits.append((frame,float(obs.score),overlap))
            recent=slot.hits[-cfg.min_hits:]
            strong=obs.score>self.tracker.track_high_thresh or (app is not None and app<=cfg.appearance_max_cost-cfg.ambiguity_margin)
            stable=(self.temporal and slot.score_floor is not None and len(recent)==cfg.min_hits
                    and recent[0][0]==frame-cfg.min_hits+1 and min(x[2] for x in recent)>=.5
                    and float(np.median([x[1] for x in recent]))>=slot.score_floor and obs.score>=slot.score_floor)
            # A plausible competing old identity vetoes promotion even if its own B gate deferred it.
            contested=any(c["candidate_id"]==cid and c["track_id"]!=tid and c["reason"] in ("plausible","accepted","ambiguous") for c in cells)
            enough=len(recent)==cfg.min_hits and not contested and (strong or stable)
            self.log(frame,tid,"probe",cid=cid,det=int(obs.det_index),score=float(obs.score),iou=overlap,
                     appearance=app,cost=cost,strong=bool(strong),stable=bool(stable),contested=contested)
            if enough:
                track=eligible.get(tid,slot.track)
                if tid not in eligible:
                    self.tracker.lost_stracks=[s for s in self.tracker.lost_stracks if int(s.id)!=tid]
                # Re-anchor the private prediction at release. Only the ordinary recovery below observes this frame.
                b=slot.box
                prior=np.r_[(b[:2]+b[2:])*.5,b[2:]-b[:2]]
                track.mean,track.covariance=track.kalman_filter.initiate(prior)
                eligible[tid]=track
                self.tracker.removed_stracks=[s for s in self.tracker.removed_stracks if int(s.id)!=tid]
                if tid not in self.tracker.memories and slot.memory is not None:
                    self.tracker.memories[tid]=slot.memory
                decision.matches.append(RecoveryMatch(tid,cid,int(obs.det_index),cost,cost,app,frame-slot.last_seen,len(recent)))
                decision.reserved_candidate_ids.discard(cid)
                if self.probation and not strong: self.provisional[tid]=slot.score_floor
                self.log(frame,tid,"release",cid=cid,det=int(obs.det_index),age=frame-slot.last_seen,
                         waiting=frame-slot.created,reason="strong" if strong else "temporal",score=float(obs.score))
                busy_ids.add(tid); occupied.add(cid)
                del self.slots[tid]
            else:
                slot.box=obs.box.copy()

        for item in sorted(cells,key=lambda x:x["cost"]):
            if item["reason"] not in ("weak_observations","uncertain_camera_recovery") or item["hits"]<cfg.min_hits: continue
            tid=int(item["track_id"]); cid=int(item["candidate_id"])
            if tid in self.slots or tid in busy_ids or tid not in eligible or cid not in available: continue
            if any(slot.cid==cid for slot in self.slots.values()): continue
            track=eligible[tid]; seen=int(track.end_frame)
            if self.opened.get(tid)==seen or self._rival(cells,tid,cid)<=item["cost"]+cfg.ambiguity_margin: continue
            cand=available.pop(cid)
            prior=self.history.get(tid,())
            score_floor=float(np.median([score for _,score in prior])) if len(prior)==cfg.window else None
            self.slots[tid]=Hypothesis(tid,seen,frame,frame,cid,cand.latest.box.copy(),deepcopy(track),
                                       deepcopy(self.tracker.memories.get(tid)),score_floor)
            self.opened[tid]=seen
            self.private_detections[int(cand.latest.det_index)]=tid
            self.log(frame,tid,"defer",cid=cid,det=int(cand.latest.det_index),last_seen=seen,score_floor=score_floor,
                     reason=item["reason"],cost=item["cost"])
