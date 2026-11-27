"""Replay original BoT-SORT with a learned covariance at the native prediction/update sites."""
from collections import Counter
from contextlib import ExitStack
import gzip
import json
import sys
from unittest.mock import patch
import cv2
import numpy as np
from boxmot.trackers.botsort.botsort_track import STrack
import botsort_plus.native_adapter as adapter
import run_botsort_plus as runner
from qr_settings import ROOT, SRC, ORIG, OUT, SETTINGS, read, write, sha

def posterior(mean, cov, measurement, variance):
    # H selects the first four XYWH components. Use Joseph's symmetric covariance update.
    cross = cov[:, :4]
    gain = np.linalg.solve(cov[:4,:4]+np.diag(variance),cross.T).T
    complement = np.eye(8)
    complement[:,:4] -= gain
    post_cov = complement@cov@complement.T + (gain*variance)@gain.T
    return mean+gain@(measurement-mean[:4]), (post_cov+post_cov.T)*0.5

def variant(base, method, sequences, proto):
    order=iter(sequences)
    full=ROOT/'full'
    q_enabled=method in ('learned_q','learned_qr')
    r_enabled=method in ('learned_r','learned_qr')
    if q_enabled:
        from q_motion import QMotion
    if r_enabled:
        from r_quality import RPolicy
    predict=STrack.multi_predict
    update=STrack.update
    recover=STrack.re_activate
    class Arm(base):
        def __init__(self,**kwargs):
            super().__init__(**kwargs)
            self.seq=next(order)
            self.counts=Counter()
            folder=full/'qr_events'/method
            folder.mkdir(parents=True,exist_ok=True)
            self.trace=gzip.open(folder/(self.seq+'.jsonl.gz'),'wt',compresslevel=1)
            if q_enabled:self.q=QMotion(OUT/'q_policy.json',method='q_mlp')
            if r_enabled:
                folder=ROOT/'quality_cache'
                meta=read(folder/(self.seq+'.json'))
                if meta['detection_fingerprint']!=proto['detection_fingerprints'][self.seq] or sha(folder/(self.seq+'.npz'))!=meta['feature_sha256'] or meta['feature_source_sha256']!=sha(SRC/'quality_features.py'):
                    raise ValueError('Frame feature cache mismatch: '+self.seq)
                with np.load(folder/(self.seq+'.npz')) as data:
                    self.offsets=data['offsets']
                    self.r_variance=RPolicy(OUT/'r_policy.json',mode='full').variance(data['features'])
                assert self.r_variance.shape==(self.offsets[-1],4) and np.isfinite(self.r_variance).all() and (self.r_variance>0).all()

        def update(self,dets,img,*args,**kwargs):
            frame=int(self.frame_count)+1
            recovered=[]
            if r_enabled:
                start,end=self.offsets[frame-1:frame+1]
                assert end-start==len(dets)
                variances=self.r_variance[start:end]

            def apply_update(original,track,obs,t,*more,**kw):
                if not r_enabled:return original(track,obs,t,*more,**kw)
                var=variances[int(obs.det_ind)]
                def with_r(mean,cov,measurement):
                    return posterior(mean,cov,measurement,var)
                self.counts['R_matched_updates']+=1
                self.counts['R_low' if obs.conf<self.track_high_thresh else 'R_high']+=1
                with patch.object(track.kalman_filter,'update',with_r):
                    return original(track,obs,t,*more,**kw)
            def corrected(track,obs,t):
                return apply_update(update,track,obs,t)
            def reactivated(track,obs,t,new_id=False):
                anchor,previous=int(track.frame_id),int(track.det_ind)
                result=apply_update(recover,track,obs,t,new_id=new_id)
                recovered.append((int(track.id),anchor,previous,int(obs.det_ind)))
                return result
            def predicted(tracks):
                predict(tracks)
                self.q.apply(tracks)
            with ExitStack() as stack:
                stack.enter_context(patch.object(STrack,'re_activate',reactivated))
                if q_enabled:stack.enter_context(patch.object(STrack,'multi_predict',staticmethod(predicted)))
                if r_enabled:stack.enter_context(patch.object(STrack,'update',corrected))
                rows=super().update(dets,img,*args,**kwargs)
            assert frame==self.frame_count
            if q_enabled:
                for track in self.active_tracks:
                    if track.frame_id==frame:self.q.observe(track)
            self.trace.write(json.dumps(dict(frame=frame,rows=[(int(row[4]),int(row[7])) for row in rows],
                                             recovered=recovered),separators=(',',':'))+'\n')
            return rows
        def finish_sequence_events(self):
            self.trace.close()
            write(full/'qr_events'/method/(self.seq+'.json'),dict(counts=dict(self.counts),frames=self.frame_count))
            return super().finish_sequence_events()
    return Arm

def main(method):
    assert method in ('native','learned_q','learned_r','learned_qr')
    cv2.setNumThreads(1)
    full=ROOT/'full'
    cfg=read(ORIG/'configs/botsort_plus/12_baseline_ref.json')
    cfg['name']=method
    path=full/'configs'/(method+'.json')
    write(path,cfg)
    s=read(SETTINGS)
    args=['--config',str(path),'--output-dir',str(full)]
    for key in ('baseline_root','features_dir','images_root','runtime_root'):args+=['--'+key.replace('_','-'),s[key]]
    original=runner.prepare
    def prepare(options):
        config,paths,proto=original(options)
        assert not config.enabled and len(paths)==30
        policy_files={'learned_q':['q_policy.json'],'learned_r':['r_policy.json'],'learned_qr':['q_policy.json','r_policy.json']}.get(method,[])
        proto['learned_covariance']={'method':method,'policies':{f:sha(OUT/f) for f in policy_files},'replay_sha256':sha(SRC/'replay.py')}
        adapter.BoTSORTPlus=variant(adapter.BoTSORTPlus,method,paths,proto)
        return config,paths,proto
    runner.prepare=prepare
    return runner.run(runner.parser().parse_args(args))

if __name__=='__main__':sys.exit(main(sys.argv[1]))
