"""Replay score-restricted R updates with the frozen native KF inputs."""
import argparse
from collections import Counter
from contextlib import ExitStack
import gzip
import importlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import cv2
import numpy as np
from rf_settings import ROOT, QR, read, write, sha

# The archived experiment supplies the exact Joseph update and NumPy model.
sys.path.insert(0,str(QR))
from replay import posterior
from r_quality import RPolicy
from qr_settings import ORIG, SETTINGS
from boxmot.trackers.botsort.botsort_track import STrack
import botsort_plus.native_adapter as adapter
import run_botsort_plus as runner

BOTSORT = importlib.import_module('boxmot.trackers.botsort.botsort')
STAGES = ('high', 'low', 'unconfirmed')
CASE = 'DJI_0309_4'
ARMS = ('native','learned_r','low_r','high_r')

def configure_arm(base, method, paths, protocol, output, detailed):
    sequence_order = iter(paths)
    original_update, original_reactivate = STrack.update, STrack.re_activate
    original_activate = STrack.activate
    original_iou, original_assignment = BOTSORT.iou_distance, BOTSORT.linear_assignment
    policy = RPolicy(QR/'results/r_policy.json',mode='full')

    class RestrictR(base):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.seq = next(sequence_order)
            self.counts = Counter()
            cache = QR/'quality_cache'
            meta = read(cache/(self.seq+'.json'))
            arrays = cache/(self.seq+'.npz')
            if (meta['detection_fingerprint'] != protocol['detection_fingerprints'][self.seq]
                    or meta['feature_source_sha256'] != sha(QR/'quality_features.py')
                    or meta['feature_sha256'] != sha(arrays)):
                raise ValueError('Frozen quality cache mismatch: '+self.seq)
            with np.load(arrays,allow_pickle=False) as data:
                self.offsets = data['offsets']
                self.variances = policy.variance(data['features'])
            if (len(self.offsets)!=protocol['sequences'][self.seq]+1 or self.offsets[0]!=0
                    or self.variances.shape!=(self.offsets[-1],4) or not np.isfinite(self.variances).all()
                    or np.any(self.variances<=0)):
                raise ValueError('Invalid frozen measurement variances')
            self.outputs = {}
            for key in (('qr_events','trace') if detailed else ('qr_events',)):
                folder = output/key/method
                folder.mkdir(parents=True,exist_ok=True)
                self.outputs[key] = gzip.open(folder/(self.seq+'.jsonl.gz'),'wt',compresslevel=2)

        def update(self, dets, img, *args, **kwargs):
            frame = self.frame_count+1
            lo,hi = self.offsets[frame-1:frame+1]
            if hi-lo != len(dets): raise ValueError(f'{self.seq}/{frame}: detection row count mismatch')
            rvars = self.variances[lo:hi]
            decisions, updates, births, recoveries = [], [], [], []

            def capture_iou(tracks, observations):
                distance = original_iou(tracks, observations)
                stage = STAGES[len(decisions)]
                decisions.append(dict(stage=stage,tids=[int(t.id) for t in tracks],
                                      dets=[int(o.det_ind) for o in observations],
                                      prior_boxes=[t.xyxy.tolist() for t in tracks],
                                      det_boxes=[o.xyxy.tolist() for o in observations],
                                      iou_cost=distance.tolist()))
                return distance

            def capture_assignment(cost, thresh):
                result = original_assignment(cost,thresh=thresh)
                pairs,_,_ = result
                decision = decisions[-1]
                if 'threshold' in decision or cost.shape!=(len(decision['tids']),len(decision['dets'])):
                    raise ValueError('Native cost matrix does not correspond to its IoU rows')
                decision.update(cost=cost.tolist(), threshold=float(thresh),
                                matches=[[decision['tids'][i],decision['dets'][j]] for i,j in pairs])
                return result

            def accepted(original, action, track, obs, t, **extra):
                score,det = float(obs.conf),int(obs.det_ind)
                low = self.track_low_thresh < score < self.track_high_thresh
                high = score > self.track_high_thresh
                if not (low or high) or not 0<=det<len(dets) or score!=float(dets[det,4]):
                    raise ValueError('Match outside native partitions or detector-row mismatch')
                apply = method=='learned_r' or (method=='low_r' and low) or (method=='high_r' and high)
                self.counts['updates_low' if low else 'updates_high'] += 1
                if apply:self.counts['applied_low' if low else 'applied_high'] += 1
                if not detailed and not apply:
                    return original(track,obs,t,**extra)
                kf = track.kalman_filter
                native = kf.update
                var = rvars[det]
                def update_measurement(mean, cov, measurement):
                    adaptive = posterior(mean,cov,measurement,var)
                    if not detailed:return adaptive
                    native_result = native(mean,cov,measurement)
                    _, projected = kf.project(mean,cov)
                    native_var = np.diag(projected-cov[:4,:4])
                    native_gain = np.linalg.solve(projected, cov[:4,:]).T
                    adaptive_gain = np.linalg.solve(cov[:4,:4]+np.diag(var), cov[:4,:]).T
                    decision=decisions[-1]
                    if [int(track.id),det] not in decision['matches']:raise ValueError('Update with no matching decision')
                    actual = adaptive if apply else native_result
                    updates.append(dict(tid=int(track.id),det=det,score=score,stage=decision['stage'],
                                        action=action,applied=apply,prior=mean.tolist(),obs=measurement.tolist(),
                                        native_post=native_result[0].tolist(),r_post=adaptive[0].tolist(),
                                        actual_post=actual[0].tolist(),native_r=native_var.tolist(),learned_r=var.tolist(),
                                        native_gain=np.diag(native_gain[:4]).tolist(),r_gain=np.diag(adaptive_gain[:4]).tolist()))
                    return actual
                with patch.object(kf,'update',update_measurement):
                    return original(track,obs,t,**extra)

            def update_track(track, obs, t):
                return accepted(original_update,'update',track,obs,t)
            def reactivate_track(track, obs, t, new_id=False):
                anchor,idx = int(track.frame_id),int(track.det_ind)
                result = accepted(original_reactivate,'reactivate',track,obs,t,new_id=new_id)
                recoveries.append([int(track.id),anchor,idx,int(obs.det_ind)])
                return result
            def activate_track(track, kf, t):
                result = original_activate(track,kf,t)
                births.append(dict(tid=int(track.id),det=int(track.det_ind),score=float(track.conf)))
                return result

            with ExitStack() as stack:
                if detailed:
                    stack.enter_context(patch.object(BOTSORT,'iou_distance',capture_iou))
                    stack.enter_context(patch.object(BOTSORT,'linear_assignment',capture_assignment))
                    stack.enter_context(patch.object(STrack,'activate',activate_track))
                stack.enter_context(patch.object(STrack,'update',update_track))
                stack.enter_context(patch.object(STrack,'re_activate',reactivate_track))
                rows = super().update(dets,img,*args,**kwargs)
            if frame!=self.frame_count:raise ValueError('Tracker and oracle frame ledgers disagree')
            row_ids = [[int(row[4]),int(row[7])] for row in rows]
            event = dict(frame=frame,rows=row_ids,recovered=recoveries)
            self.outputs['qr_events'].write(json.dumps(event,allow_nan=False,separators=(',',':'))+'\n')
            if detailed:
                if len(decisions)!=3 or any('threshold' not in d for d in decisions):
                    raise ValueError('Expected exactly three native associations')
                event.update(updates=updates,births=births,associations=decisions)
                self.outputs['trace'].write(json.dumps(event,allow_nan=False,separators=(',',':'))+'\n')
            return rows

        def finish_sequence_events(self):
            for stream in self.outputs.values():stream.close()
            write(output/'qr_events'/method/(self.seq+'.json'),dict(frames=self.frame_count,counts=dict(self.counts)))
            return super().finish_sequence_events()
    return RestrictR

def main(stage, method):
    output = ROOT/stage
    detailed = stage=='pilot'
    cv2.setNumThreads(1)
    config_path = output/'configs'/(method+'.json')
    config = read(ORIG/'configs/botsort_plus/12_baseline_ref.json')
    config['name'] = method
    write(config_path,config)
    settings = read(SETTINGS)
    args = ['--config',str(config_path),'--output-dir',str(output)]
    for key in ('baseline_root','features_dir','images_root','runtime_root'):
        args += ['--'+key.replace('_','-'),settings[key]]
    if detailed:args+=['--sequences',CASE]
    previous=read(QR/'full/protocols/learned_r.json')
    policy_sha=sha(QR/'results/r_policy.json')
    if policy_sha!=previous['learned_covariance']['policies']['r_policy.json']:
        raise ValueError('R weights differ from archived full-R arm')
    old_prepare=runner.prepare
    def prepare(params):
        cfg,paths,p=old_prepare(params)
        if cfg.enabled or (set(paths)!={CASE} if detailed else len(paths)!=30):raise ValueError('Incorrect native subset')
        for key in ('detection_fingerprints','feature_hashes','sequences'):
            if any(p[key][s]!=previous[key][s] for s in paths):raise ValueError('Historical input mismatch: '+key)
        for key in ('native_config','source_protocol_sha256','detector_sha256','rgb_encoder_sha256'):
            if p[key]!=previous[key]:raise ValueError('Historical configuration mismatch: '+key)
        p['r_forensics']=dict(method=method,policy_sha256=policy_sha,code_sha256=sha(Path(__file__)),
                              strict_low='low_threshold < score < high_threshold',strict_high='score > high_threshold',
                              native_births=True)
        adapter.BoTSORTPlus=configure_arm(adapter.BoTSORTPlus,method,paths,p,output,detailed)
        return cfg,paths,p
    runner.prepare=prepare
    status=runner.run(runner.parser().parse_args(args))
    result=read(output/'protocols'/(method+'.json'))
    if status!=0 or result['status']!='COMPLETE': raise RuntimeError('Incomplete replay')
    old=read(QR/'full/protocols'/('native.json' if method=='native' else 'learned_r.json'))
    for seq in result['sequences']:
        if result['decoded_image_hashes'][seq]!=old['decoded_image_hashes'][seq]:raise ValueError('Decoded RGB mismatch')
        if method in ('native','learned_r') and result['output_prediction_hashes'][seq]!=old['output_prediction_hashes'][seq]:
            raise ValueError('Reference trajectories differ from source archive')
    print('VERIFIED',stage,method,len(result['sequences']),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('pilot','full'))
    parser.add_argument('method',choices=ARMS)
    opt=parser.parse_args()
    main(opt.stage,opt.method)
