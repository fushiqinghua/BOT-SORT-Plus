"""One process per arm and one active sequence at a time; no GT access."""
from pathlib import Path
import gzip
import json
import numpy as np

ACTIVE=None
FOCUS=("DJI_0173_2","DJI_0277_2")


def move_measurement(z, transform):
    linear=transform[:2,:2]
    return np.r_[linear@z[:2]+transform[:2,2], linear@z[2:]]


def replay(track, saved, last, frame, camera):
    first,mean,cov,z0,_=saved
    count=frame-first
    if count<2:
        return None
    # Map both endpoint measurements into each missing frame's coordinates.
    # Apply the same size and velocity transforms as this BoxMOT XYWH filter.
    matrices=[np.eye(3)]
    for f in range(first+1,frame+1):
        matrices.append(np.vstack([camera[f],[0.,0.,1.]])@matrices[-1])
    if abs(np.linalg.det(matrices[-1][:2,:2]))<1e-8:
        return None
    last_in_first=move_measurement(last,np.linalg.inv(matrices[-1]))
    mean,cov=mean.copy(),cov.copy()
    for k in range(1,count+1):
        if k>1:
            mean[6:8]=0.  # native Lost transition
        mean,cov=track.kalman_filter.predict(mean,cov)
        h=camera[first+k]
        linear=np.kron(np.eye(4),h[:,:2])
        mean=linear@mean
        mean[:2]+=h[:,2]
        cov=linear@cov@linear.T
        if k<count:
            z=move_measurement(z0+(last_in_first-z0)*(k/count),matrices[k])
            if not np.isfinite(z).all() or np.any(z[2:]<=0):
                return None
            mean,cov=track.kalman_filter.update(mean,cov,z)
    if not np.isfinite(mean).all() or not np.isfinite(cov).all() or np.any(mean[2:4]<=0):
        return None
    np.linalg.cholesky(cov)
    return mean,cov


def remember(st, obs, frame):
    st.abl_last=(frame,st.mean.copy(),st.covariance.copy(),obs.xywh.copy(),int(obs.det_ind))


def install(method, sequences, root, smoke):
    import botsort_plus.native_adapter as adapter
    from boxmot.trackers.botsort.botsort_track import STrack
    from adaptive_gmc import Adaptive,Matcher
    native_activate,native_update,native_recover=STrack.activate,STrack.update,STrack.re_activate
    native_gmc=STrack.multi_gmc
    matcher=Matcher() if method=="adaptive_gmc" else None

    def activate(st,kf,frame_id):
        native_activate(st,kf,frame_id)
        remember(st,st,frame_id)

    def update(st,new_track,frame_id):
        native_update(st,new_track,frame_id)
        remember(st,new_track,frame_id)

    def recover(st,new_track,frame_id,new_id=False):
        old=st.abl_last
        assert old[0]==st.frame_id and frame_id==ACTIVE.t and not new_id
        gap=int(frame_id-old[0]-1)
        record=dict(frame=frame_id,old_frame=old[0],tid=int(st.id),gap=gap,
                    old_det=old[4],det=int(new_track.det_ind),virtual=0)
        if method=="oru" and gap>0:
            try:
                state=replay(st,old,new_track.xywh,frame_id,ACTIVE.camera)
            except np.linalg.LinAlgError:
                state=None
            if state is not None:
                st.mean,st.covariance=state
                record["virtual"]=gap
            else:
                record["fallback"]=True
        native_recover(st,new_track,frame_id,new_id)
        remember(st,new_track,frame_id)
        ACTIVE.log("recoveries",record)

    def gmc(stracks,H=np.eye(2,3)):
        if ACTIVE.pool_calls==0 and smoke and method=="native":
            ACTIVE.log("prestate",dict(frame=ACTIVE.t,tracks=[
                [int(t.id),int(t.frame_id),float(t.mean[0]),float(t.mean[1])] for t in stracks]))
        ACTIVE.pool_calls+=1
        return native_gmc(stracks,H)

    STrack.activate,STrack.update,STrack.re_activate=activate,update,recover
    STrack.multi_gmc=staticmethod(gmc)
    seqiter=iter(sequences)

    class Arm(adapter.BoTSORTPlus):
        def __init__(self,**kwargs):
            super().__init__(**kwargs)
            self.seq=next(seqiter)
            self.t=0
            self.camera={}
            self.files={}
            self.adaptive=Adaptive(matcher) if matcher else None
            self.orig_cmc=self.cmc.apply
            self.cmc.apply=self.camera_warp
            for name in ["warps","recoveries"]+(["prestate"] if smoke and method=="native" else []):
                p=Path(root)/"audit"/method/name/(self.seq+".jsonl.gz")
                p.parent.mkdir(parents=True,exist_ok=True)
                self.files[name]=gzip.open(p,"xt",encoding="utf-8",compresslevel=3)
        def log(self,name,obj):
            self.files[name].write(json.dumps(obj,separators=(",",":"),allow_nan=False)+"\n")
        def camera_warp(self,img,dets):
            native=self.orig_cmc(img,dets)
            used,details=native,{}
            if self.adaptive is not None:
                used,details=self.adaptive.apply(img,dets,native)
            self.camera[self.t]=np.asarray(used).copy()
            self.log("warps",dict(frame=self.t,native=native.tolist(),used=used.tolist(),**details))
            self.warp_calls+=1
            return used
        def update(self,*args,**kwargs):
            global ACTIVE
            ACTIVE=self
            self.t+=1
            self.warp_calls,self.pool_calls=0,0
            result=super().update(*args,**kwargs)
            if self.warp_calls!=1 or self.pool_calls!=2 or self.t!=self.frame_count:
                raise RuntimeError((self.t,self.warp_calls,self.pool_calls,self.frame_count))
            return result
        def finish_sequence_events(self):
            for stream in self.files.values():
                stream.close()
            return super().finish_sequence_events()
    adapter.BoTSORTPlus=Arm
