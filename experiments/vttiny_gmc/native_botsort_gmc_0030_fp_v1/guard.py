"""Causal alternatives for a LightGlue decision. No clip, frame, track or GT identifiers."""
import numpy as np
from boxmot.trackers.botsort.basetrack import TrackState
from boxmot.utils.matching import linear_assignment,iou_distance
from boxmot.utils.ops import xywh2xyxy

POLICIES=("dual_low","pixel_margin")
MARGIN=1.0  # pixels; the original native median-error trigger has this same scale.


def dual_low(policy):
    return policy in ("dual_low","dual_low_center")


def measured_centers(outputs,dets,high):
    """Publish the accepted high-score measurement center, retaining the posterior width/height."""
    if not len(outputs):
        return outputs,0
    eligible=outputs[:,5]>high
    if not eligible.any():
        return outputs,0
    rows=outputs.copy()
    obs=dets[rows[eligible,7].astype(int),:4].astype(float)
    shift=(obs[:,:2]+obs[:,2:]-rows[eligible,:2]-rows[eligible,2:4])*.5
    rows[eligible,:2]+=shift
    rows[eligible,2:4]+=shift
    return rows,int(eligible.sum())


def projected_xyxy(mean,H):
    z=mean[:4].copy()
    z[:2]=H[:,:2]@mean[:2]+H[:,2]
    z[2:4]=H[:,:2]@mean[2:4]
    return np.asarray(xywh2xyxy(z))


def iou(a,b):
    low=np.maximum(a[:2],b[:2])
    high=np.minimum(a[2:],b[2:])
    inter=float(np.maximum(0.,high-low).prod())
    return inter/max(float((a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod())-inter,1e-12)


def decide(policy,full,backup,record):
    if policy!="pixel_margin" or record["choice"]!="lightglue":
        return full
    alt=(record["candidate_errors"]["identity"] if "identity" in record["eligible"] else record["native_error"])
    return full if alt[0]-record["candidate_errors"]["lightglue"][0]>=MARGIN else backup


def capture_priors(tracker,used,backup):
    tracker.backup_boxes={}
    tracker.priors={}
    for st in [*tracker.active_tracks,*tracker.lost_stracks]:
        if st.mean is not None:
            tid=int(st.id)
            tracker.backup_boxes[tid]=projected_xyxy(st.mean,backup)
            if tracker.capture_trace:
                tracker.priors[tid]=(projected_xyxy(st.mean,used),int(st.frame_id),int(st.state),bool(st.is_activated))


class LowMatchGuard:
    """Keep the native second pass; when requested, mask low-score edges failing the backup IoU gate."""
    def _second_association(self,dets_second,activated_stracks,lost_stracks,refind_stracks,u_track_first,strack_pool):
        enabled=dual_low(self.case_policy) and self.matched_camera
        args=(dets_second,activated_stracks,lost_stracks,refind_stracks,u_track_first,strack_pool)
        if not enabled and not self.capture_trace:
            return super()._second_association(*args)
        pool=[strack_pool[i] for i in u_track_first if strack_pool[i].state==TrackState.Tracked]
        blocked={}
        if len(dets_second) and pool:
            for i,st in enumerate(pool):
                prior=self.backup_boxes.get(int(st.id))
                if prior is None:
                    raise RuntimeError("Missing fallback projection for an active track")
                for j,det in enumerate(dets_second):
                    native_iou=iou(st.xyxy,det[:4])
                    alt_iou=iou(prior,det[:4])
                    if enabled and alt_iou<.5:
                        blocked[i,j]=True
                    if self.capture_trace and (native_iou>.03 or alt_iou>.03):
                        self.edges.append(dict(frame=self.frame_count,tid=int(st.id),det=int(det[6]),score=float(det[4]),
                            used_iou=native_iou,backup_iou=alt_iou,masked=int((i,j) in blocked)))
        if not blocked:
            return super()._second_association(*args)
        # Identical ranking and native unmatched/lost transitions. The full and backup geometries
        # must both pass the native 0.5 threshold for a weak detection to continue an active track.
        from boxmot.trackers.botsort.botsort_track import STrack
        low=[STrack(row,max_obs=self.max_obs) for row in dets_second]
        cost=iou_distance(pool,low)
        for i,j in blocked:
            cost[i,j]=1.
        matches,unmatched,unused=linear_assignment(cost,thresh=.5)
        for i,j in matches:
            st=pool[i];st.update(low[j],self.frame_count);activated_stracks.append(st)
        for i in unmatched:
            st=pool[i];st.mark_lost();lost_stracks.append(st)
        return matches,unmatched,unused
