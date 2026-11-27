"""Compose the archived B-stage BAM and a train-fitted, score-restricted R."""
from collections import Counter
from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


@contextmanager
def source_path(path):
    sys.path.insert(0, str(path))
    try:
        yield
    finally:
        sys.path.remove(str(path))


def load_archive(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_artifacts(project, low_r):
    integrations = project.parent / 'VT_Tiny_MOT_Baseline_v1/integrations'
    motion = integrations / 'native_motion_345_v1'
    evidence = read(motion / 'full/protocols/bam_overlap.json')['motion345']['source_sha256']
    for filename, expected in evidence.items():
        if sha(motion / filename) != expected:
            raise ValueError(f'BAM source differs from the archived experiment: {filename}')
    os.environ['VT_INTEGRATIONS'] = str(integrations)
    with source_path(motion):
        archive = importlib.import_module('run_arm')
    if Path(archive.__file__).resolve() != motion / 'run_arm.py':
        raise ValueError('BAM variant is not the archived source')
    if not low_r:
        return archive, evidence, None, None, None
    fitted = project / 'm3ot_branch_replay_v1/tracking/learned/fit_m3ot'
    fit = read(fitted / 'results/fit_receipt.json')
    if fit['status'] != 'COMPLETE' or fit['val_gt_used'] or fit['test_gt_used']:
        raise ValueError('R must be the existing training-side M3OT refit')
    path = fitted / 'results/r_policy.json'
    if sha(path) != fit['policies']['r_policy.json']:
        raise ValueError('M3OT R weights changed')
    neural = integrations / 'native_neural_qr_v1'
    kernel = neural / 'replay.py'
    qr = read(neural / 'full/protocols/learned_qr.json')['learned_covariance']
    if sha(kernel) != qr['replay_sha256'] or sha(neural/'quality_features.py') != sha(fitted/'quality_features.py'):
        raise ValueError('R kernel or observation feature source changed')
    with source_path(neural):
        posterior = load_archive(kernel, '_m3ot_low_stack_original_qr').posterior
        policy = load_archive(neural/'r_quality.py', '_m3ot_low_stack_original_r').RPolicy(path, mode='full')
    proof = dict(policy_sha256=sha(path), joseph_kernel_source_sha256=sha(kernel),
                 network_source_sha256=sha(neural/'r_quality.py'), fit_groups=fit['groups'])
    return archive, evidence, fitted, (policy, posterior), proof


def frame_variances(fitted, policy, seq, protocol):
    cache = fitted / 'quality_cache'
    arrays = cache / (seq + '.npz')
    meta = read(cache / (seq + '.json'))
    if (meta['detection_fingerprint'] != protocol['detection_fingerprints'][seq]
            or meta['feature_source_sha256'] != sha(fitted/'quality_features.py')
            or sha(arrays) != meta['feature_sha256']):
        raise ValueError(f'Frozen R image/detection feature order changed: {seq}')
    with np.load(arrays, allow_pickle=False) as data:
        offsets = data['offsets']
        result = policy.variance(data['features'])
    if (len(offsets) != protocol['sequences'][seq]+1 or offsets[0] != 0 or np.any(np.diff(offsets)<0)
            or result.shape != (int(offsets[-1]),4) or not np.isfinite(result).all() or np.any(result<=0)):
        raise ValueError(f'Invalid R variance array: {seq}')
    return offsets, result, meta['feature_sha256']


def compose_low_update(base, project, output, arm, paths, protocol, *, bam=False, low_r=False):
    """Return a tracker subclass; use after any association-only decorator.

    A matched low-score observation receives a Joseph update under learned R.
    The archived BAM then scales that update's mean displacement when its
    previous-overlap gate accepts the same match. It does not rewrite covariance.
    """
    from boxmot.trackers.botsort.botsort_track import STrack

    archive, bam_sources, fitted, r_model, r_proof = load_artifacts(project, low_r)
    original_update = STrack.update
    context = {}

    def observe_update(track, obs, frame):
        owner = context['owner']
        if not owner.track_low_thresh < obs.conf < owner.track_high_thresh:
            return original_update(track, obs, frame)
        key = (int(track.id), int(obs.det_ind))
        if key in owner.low_keys or not 0 <= key[1] < len(owner.dets):
            raise ValueError('Invalid accepted low-stage event key')
        if float(obs.conf) != float(owner.dets[key[1],4]):
            raise ValueError('Accepted low-stage observation differs from frozen detection')
        owner.low_keys.add(key)
        if not low_r:
            return original_update(track, obs, frame)

        kf = track.kalman_filter
        var = owner.variances[owner.low_offset + key[1]]
        calls = 0

        def apply_r(mean,cov,measurement):
            nonlocal calls
            calls += 1
            _, projected = kf.project(mean,cov)
            ref = np.diag(projected-cov[:4,:4]).copy()
            if np.any(ref <= 0):
                raise ValueError('Positive original measurement variance required')
            owner.r_ratio.append((var/ref).tolist())
            posterior, covariance = r_model[1](mean,cov,measurement,var)
            owner.r_data[key] = (track, posterior.copy(), covariance.copy())
            return posterior, covariance

        with patch.object(kf,'update',apply_r):
            result=original_update(track,obs,frame)
        if calls != 1:
            raise ValueError('An accepted low-stage update did not invoke R exactly once')
        return result

    # The pinned BAM factory captures STrack.update when constructing the class.
    # Its exact weighting code runs around the scoped measurement hook.
    with patch.object(STrack, 'update', observe_update):
        parent = archive.variant(base, 'bam_overlap' if bam else 'native', paths,
                                 output/'archive_traces'/arm, None)

    class LowUpdateTracker(parent):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.trace_counts=Counter()
            self.weights=[]
            self.r_ratio=[]
            self.sof_digest=hashlib.sha256()
            self.original_warp=self.cmc.apply
            self.cmc.apply=self.capture_warp
            self.feature_sha=None
            if low_r:
                self.offsets,self.variances,self.feature_sha=frame_variances(fitted,r_model[0],self.seq,protocol)

        def capture_warp(self,image,detections):
            matrix=self.original_warp(image,detections)
            self.sof_digest.update(np.asarray(matrix,dtype='<f8').tobytes(order='C'))
            self.trace_counts['frames']+=1
            return matrix

        def event(self, kind, **data):
            if kind=='crowded_low':
                key=(int(data['tid']),int(data['det']))
                if key not in self.low_keys or key in self.bam_keys:
                    raise ValueError('BAM ran outside the accepted low match')
                self.bam_keys.add(key)
                weight = data['weight']
                expected=data['iou']*(1-data['area']) if bam else 1.
                if not (0 < data['area'] <= 1 and np.isclose(weight,expected)):
                    raise ValueError('Archive BAM gate or weight unexpectedly changed')
                if bam:
                    if not 0<=weight<1:
                        raise ValueError('BAM overlap event did not reduce measurement weight')
                    self.weights.append(weight)
                if key in self.r_data:
                    track, mean, cov = self.r_data[key]
                    if not np.array_equal(track.covariance,cov) or not np.array_equal(mean[:2],data['native_post']):
                        raise ValueError('BAM changed R covariance or used a different posterior')
                    expected_post = np.asarray(data['prior']) + weight*(mean[:2]-np.asarray(data['prior']))
                    if not np.array_equal(expected_post,track.mean[:2]):
                        raise ValueError('BAM did not scale the R-conditioned posterior')
            return super().event(kind,**data)

        def update(self,dets,image,*args,**kwargs):
            self.low_keys=set()
            self.bam_keys=set()
            self.r_data={}
            self.dets=dets
            frame=self.frame_count+1
            if low_r:
                lo,hi=self.offsets[frame-1:frame+1]
                if hi-lo != len(dets):
                    raise ValueError('Frame R feature offsets leave the ordered detection rows')
                self.low_offset=lo
            context['owner']=self
            try:
                rows=super().update(dets,image,*args,**kwargs)
            finally:
                context.clear()
            matched={(int(e['track_id']),int(e['det_index'])) for e in self.last_events
                     if e['event']=='match' and e['stage']=='low'}
            if matched!=self.low_keys or (low_r and set(self.r_data)!=matched):
                raise ValueError('B native low-stage audit and update hook disagree')
            self.trace_counts['accepted_low']+=len(matched)
            self.trace_counts['overlap_eligible']+=len(self.bam_keys)
            self.trace_counts['r_updates']+=len(self.r_data)
            self.trace_counts['bam_updates']+=len(self.bam_keys) if bam else 0
            self.trace_counts['both_updates']+=len(self.r_data.keys() & self.bam_keys) if bam else 0
            self.trace_counts['r_frames']+=bool(self.r_data)
            self.trace_counts['bam_frames']+=bool(self.bam_keys) if bam else 0
            self.trace_counts['b_recoveries']+=sum(e['event']=='recovered' for e in self.last_events)
            return rows

        def finish_sequence_events(self):
            if self.trace_counts['frames'] != protocol['sequences'][self.seq]:
                raise ValueError('Replay did not process the full sequence')
            ratios=np.asarray(self.r_ratio)
            weights=np.asarray(self.weights)
            r={}
            if ratios.size:
                r=dict(median_ratio=np.median(ratios,axis=0).tolist(),
                       increased_components=int((ratios>1).sum()),
                       equal_components=int((ratios==1).sum()),decreased_components=int((ratios<1).sum()))
            b=dict(weight_median=float(np.median(weights)),weight_mean=float(weights.mean())) if weights.size else {}
            save(output/'low_update_events'/arm/(self.seq+'.json'),
                 dict(sequence=self.seq,**{k:self.trace_counts[k] for k in (
                      'frames','accepted_low','overlap_eligible','r_updates','bam_updates','both_updates',
                      'r_frames','bam_frames','b_recoveries')},native_sof_sha256=self.sof_digest.hexdigest(),
                      r_quality_feature_sha256=self.feature_sha,r_variance_vs_native=r,bam_weight=b))
            return super().finish_sequence_events()

    protocol['low_updates'] = dict(bam=bool(bam),measurement_r=bool(low_r),bam_sources=bam_sources,
        only_accepted_native_low_stage=True, r=r_proof,
        combined_update='Joseph R-conditioned posterior followed by archive BAM mean-step scaling; R covariance retained')
    return LowUpdateTracker
