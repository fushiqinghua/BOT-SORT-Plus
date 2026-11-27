"""One native BoT-SORT process per arm. GT is opened only by the separate scorer."""
from collections import Counter
from contextlib import ExitStack
import gzip
import hashlib
import importlib
import json
import sys
from unittest.mock import patch

import cv2
import numpy as np
from boxmot.trackers.botsort.botsort_track import STrack
from boxmot.trackers.botsort.basetrack import TrackState
import botsort_plus.native_adapter as adapter
import run_botsort_plus as runner
from occlusion import occlusion_fractions
from perspective import Registration, association_cost
from process_noise import UNIT_Q,features,variance
from settings import BAM,CAMERA,QARMS,RESULT,ROOT,REPO,SETTINGS,SRC,PILOT,read,save,sha

native_module=importlib.import_module('boxmot.trackers.botsort.botsort')


def variant(base, method, sequences, output, weights):
    sequence=iter(sequences)
    original_cost=native_module.iou_distance
    original_predict=STrack.multi_predict
    original_gmc=STrack.multi_gmc
    original_update=STrack.update
    original_recover=STrack.re_activate
    trace_state=method in ('native',*QARMS)
    trace_low=method in ('native',*BAM)

    class Arm(base):
        def __init__(self,**kwargs):
            super().__init__(**kwargs)
            self.seq=next(sequence); self.n=Counter(); self.last_overlap={}; self.old_boxes={}
            self.H=self.Hinv=None; self.camera_digest=hashlib.sha256()
            folder=output/'diagnostics345'/method;folder.mkdir(parents=True,exist_ok=True)
            self.log=gzip.open(folder/(self.seq+'.jsonl.gz'),'wt',encoding='utf-8',compresslevel=2)
            if method in CAMERA:
                self.registration=Registration(); self.sof=self.cmc.apply; self.cmc.apply=self.camera

        def event(self,kind,**fields):
            self.n[kind]+=1
            self.log.write(json.dumps(dict(frame=self.frame_count,event=kind,**fields),allow_nan=False,separators=(',',':'))+'\n')

        def camera(self,image,dets):
            native=self.sof(image,dets)
            self.H,stats=self.registration.step(image,dets,native,method!='affine_forward')
            self.Hinv=None
            if method=='homography_bidir' and self.H is not None:
                inverse=np.linalg.inv(self.H);self.Hinv=inverse/inverse[2,2]
            self.camera_digest.update(stats['choice'].encode()+np.asarray(native if self.H is None else self.H,dtype='<f8').tobytes())
            self.n['selected_'+stats['choice']]+=1
            self.event('registration',**stats)
            return native

        def match_cost(self,tracks,dets):
            if self.H is not None:self.n['projected_pairs']+=len(tracks)*len(dets)
            return association_cost(original_cost,tracks,dets,self.old_boxes,self.H,self.Hinv)

        def transport(self,tracks,warp):
            if method in CAMERA and self.H is not None:
                self.old_boxes.update({id(t):t.xyxy.copy() for t in tracks})
            original_gmc(tracks,warp)
            if trace_state:
                for t in tracks:
                    if t.state==TrackState.Lost:
                        self.event('lost',tid=int(t.id),anchor=int(t.frame_id),anchor_det=int(t.det_ind),
                                   mean=t.mean[:2].tolist(),cov=t.covariance[:2,:2].tolist())

        def predict(self,tracks):
            original_predict(tracks)
            for t in tracks:
                if t.state==TrackState.Lost:
                    t.covariance=t.covariance+t.accel_variance*UNIT_Q
                    self.n['injected_Q']+=1

        def update(self,dets,img,*args,**kwargs):
            self.old_boxes.clear()
            def correct(track,obs,t):
                previous=self.last_overlap.get(int(track.id))
                if not self.track_low_thresh<obs.conf<self.track_high_thresh or previous is None or previous[0]!=t-1 or previous[1]<=0:
                    return original_update(track,obs,t)
                _,overlap,bottom=previous
                prior=track.mean.copy(); sim=1.-float(original_cost([track],[obs])[0,0]); sim=float(np.clip(sim,0.,1.))
                area=overlap if method=='bam_overlap' else bottom if method=='bam_bottom' else 0.
                weight=1. if method=='native' else sim*(1.-area)
                original_update(track,obs,t)
                posterior=track.mean.copy()
                if method!='native':track.mean=prior+weight*(posterior-prior)
                self.event('crowded_low',tid=int(track.id),det=int(obs.det_ind),iou=sim,area=overlap,bottom=bottom,
                           weight=weight,prior=prior[:2].tolist(),native_post=posterior[:2].tolist(),post=track.mean[:2].tolist())
            def recovered(track,obs,t,new_id=False):
                anchor=int(track.frame_id); idx=int(track.det_ind); before=track.mean[:2].copy()
                original_recover(track,obs,t,new_id=new_id)
                self.event('recovered',tid=int(track.id),det=int(obs.det_ind),anchor=anchor,anchor_det=idx,
                           prior=before.tolist(),post=track.mean[:2].tolist())
            with ExitStack() as stack:
                if method in CAMERA or trace_state:
                    stack.enter_context(patch.object(STrack,'multi_gmc',staticmethod(self.transport)))
                if method in CAMERA:stack.enter_context(patch.object(native_module,'iou_distance',self.match_cost))
                if method in QARMS:stack.enter_context(patch.object(STrack,'multi_predict',staticmethod(self.predict)))
                if trace_state:stack.enter_context(patch.object(STrack,'re_activate',recovered))
                if trace_low:stack.enter_context(patch.object(STrack,'update',correct))
                rows=super().update(dets,img,*args,**kwargs)
            active=[t for t in self.active_tracks if t.frame_id==self.frame_count]
            if method in QARMS:
                for t in active:t.accel_variance=float(variance(features(t.mean),weights))
            if trace_low:
                boxes=[dets[int(t.det_ind),:4] for t in active]
                all_overlap,bottom=occlusion_fractions(boxes)
                self.last_overlap={int(t.id):(self.frame_count,float(a),float(b)) for t,a,b in zip(active,all_overlap,bottom)}
            return rows

        def finish_sequence_events(self):
            self.log.close()
            save(output/'diagnostics345'/method/(self.seq+'.json'),dict(counts=dict(self.n),geometry_sha256=self.camera_digest.hexdigest()))
            return super().finish_sequence_events()
    return Arm


def main(stage,method):
    if method not in ('native',*CAMERA,*BAM,*QARMS):raise ValueError(method)
    output=ROOT/stage
    cfg=read(REPO/'configs/botsort_plus/12_baseline_ref.json'); cfg['name']=method
    path=output/'configs'/(method+'.json');save(path,cfg)
    s=read(SETTINGS)
    params=['--config',str(path),'--output-dir',str(output)]
    for field in ('baseline_root','features_dir','images_root','runtime_root'):params+=['--'+field.replace('_','-'),s[field]]
    if stage=='pilot':params+=['--sequences',*PILOT]
    prepare=runner.prepare
    def configure(args):
        cfg,paths,p=prepare(args); assert not cfg.enabled
        weights=None
        record=dict(method=method,source_sha256={f:sha(SRC/f) for f in ('run_arm.py','perspective.py','process_noise.py','occlusion.py')})
        if method in QARMS:
            policy=RESULT/'q_policy.json'; weights=read(policy)['params'][method];record['policy_sha256']=sha(policy)
        p['motion345']=record
        adapter.BoTSORTPlus=variant(adapter.BoTSORTPlus,method,paths,output,weights)
        return cfg,paths,p
    runner.prepare=configure
    return runner.run(runner.parser().parse_args(params))


if __name__=='__main__':sys.exit(main(*sys.argv[1:]))
