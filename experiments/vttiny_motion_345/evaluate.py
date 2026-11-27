"""Read official test GT only after prediction. All track ownership comes from TrackEval CLEAR."""
import gzip
import itertools
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from unittest.mock import patch

import numpy as np

from settings import BAM, CAMERA, EVAL, HISTORY, PILOT, QARMS, REPO, RESULT, ROOT, SETTINGS, environment, read, save, sha, table

METHODS = ('native', *CAMERA, *BAM, *QARMS)
RUN = ROOT / 'full'
LIMIT = 5.991465


def validate():
    keys = ('feature_hashes', 'decoded_image_hashes', 'detection_fingerprints', 'sequences', 'encoder_contracts',
            'native_config', 'boxmot_source_hashes', 'detector_sha256', 'rgb_encoder_sha256', 'source_protocol_sha256')
    history = read(HISTORY)
    seqs = history['sequences']
    assert len(seqs) == 30 and sum(seqs.values()) == 12590
    digests = {}
    for method in METHODS:
        full = read(RUN/'protocols'/(method+'.json'))
        pilot = read(ROOT/'pilot'/'protocols'/(method+'.json'))
        assert full['status'] == pilot['status'] == 'COMPLETE'
        for k in keys:
            assert full[k] == history[k], (method, k)
        assert set(pilot['sequences']) == set(PILOT)
        for k in ('output_prediction_hashes', 'decoded_image_hashes', 'detection_fingerprints', 'feature_hashes'):
            assert pilot[k] == {s: full[k][s] for s in PILOT}, (method, k, 'pilot')
        assert full['motion345']['source_sha256'] == pilot['motion345']['source_sha256']
        if method in QARMS:
            assert full['motion345']['policy_sha256'] == pilot['motion345']['policy_sha256'] == sha(RESULT/'q_policy.json')
        if method == 'native':
            assert full['output_prediction_hashes'] == history['output_prediction_hashes']
        digests[method] = sha(RUN/'protocols'/(method+'.json'))
    for s in seqs:
        pair = [read(RUN/'diagnostics345'/m/(s+'.json'))['geometry_sha256'] for m in CAMERA[1:]]
        assert pair[0] == pair[1], (s, 'forward/bidirectional warps')
    save(RESULT/'input_audit.json', dict(sequences=len(seqs), frames=sum(seqs.values()), history_native_exact=len(seqs),
         pilot_full_exact={m:len(PILOT) for m in METHODS}, matching_geometry_sequences=len(seqs),
         shared_protocol_fields=list(keys), policy_sha256=sha(RESULT/'q_policy.json'), protocol_sha256=digests,
         reference_protocol_sha256=sha(HISTORY), detector_sha256=history['detector_sha256'],
         rgb_encoder_sha256=history['rgb_encoder_sha256']))
    return seqs


def records(method, seq):
    with gzip.open(RUN/'diagnostics345'/method/(seq+'.jsonl.gz'), 'rt') as f:
        for line in f:
            yield json.loads(line)


def camera_diagnosis(seqs):
    rows, quadrants = [], []
    for seq in seqs:
        for method in CAMERA[:2]:
            counts = Counter()
            vals = defaultdict(list)
            for e in records(method, seq):
                choice = e['choice']
                counts[choice] += 1
                if 'native_error' in e:
                    vals['native'].append(e['native_error'][0])
                    vals['chosen'].append(e[choice+'_error'][0])
                if choice != 'homography':
                    continue
                a, h = e['reference_quadrants'], e['chosen_quadrants']
                vectors = [np.array(v) for v in a if v is not None]
                opposite = any(x @ y < 0 for x,y in itertools.combinations(vectors,2))
                for i, (before, after) in enumerate(zip(a,h)):
                    if before is None: continue
                    quadrants.append(dict(sequence=seq, frame=e['frame'], quadrant=i, reference_dx=before[0],
                          reference_dy=before[1], homography_dx=after[0], homography_dy=after[1],
                          opposite_reference=int(opposite)))
                if len(vectors) >= 2:
                    vals['opposite'].append(int(opposite))
            assert sum(counts.values()) == seqs[seq]
            rows.append(dict(sequence=seq, method=method, frames=seqs[seq], native=counts['native'],
                affine=counts['affine'], homography=counts['homography'], heldout_frames=len(vals['native']),
                sum_native_median_px=sum(vals['native']), sum_selected_median_px=sum(vals['chosen']),
                h_frames_opposite=sum(vals['opposite']), h_frames_multiple_quadrants=len(vals['opposite'])))
    table(RESULT/'perspective_by_sequence.csv', rows)
    table(RESULT/'perspective_quadrants.csv', quadrants)


def owners(method, seq, frames, score):
    from evaluate_vttiny import VTTinyObjects
    from trackeval.metrics import CLEAR
    import trackeval.metrics.clear as clear_module
    ds = VTTinyObjects(dict(GT_FOLDER=read(SETTINGS)['gt_root'], TRACKERS_FOLDER=str(RUN/'trackers'),
         TRACKERS_TO_EVAL=[method], BENCHMARK='VT-Tiny-MOT', SPLIT_TO_EVAL='test', SKIP_SPLIT_FOL=True,
         DO_PREPROC=False, PRINT_CONFIG=False, SEQ_INFO={seq:frames}))
    raw = ds.get_raw_seq_data(method,seq)
    data = ds.get_preprocessed_seq_data(raw,'object')
    times = iter(i for i in range(frames) if len(data['gt_ids'][i]) and len(data['tracker_ids'][i]))
    matched = [{} for _ in range(frames+1)]
    truth = [{}]
    centers = [{}]
    for t in range(frames):
        keep = np.asarray(raw['gt_extras'][t]['zero_marked']) != 0
        ids = np.asarray(raw['gt_ids'][t])[keep]
        boxes = np.asarray(raw['gt_dets'][t]).reshape(-1,4)[keep]
        truth.append({int(i):b[:2]+b[2:]/2 for i,b in zip(ids,boxes)})
        centers.append({int(i):b[:2]+b[2:]/2 for i,b in zip(raw['tracker_ids'][t],raw['tracker_dets'][t])})
    solver = clear_module.linear_sum_assignment
    def capture(cost):
        t = next(times)
        ii, jj = solver(cost)
        ok = -cost[ii,jj] > np.finfo(float).eps
        gids = list(truth[t+1])
        for a,b in zip(ii[ok],jj[ok]):
            matched[t+1][int(raw['tracker_ids'][t][b])] = gids[a]
        return ii,jj
    with patch.object(clear_module,'linear_sum_assignment',capture):
        res = CLEAR({'PRINT_CONFIG':False}).eval_sequence(data)
    assert next(times,None) is None and sum(map(len,matched)) == res['CLR_TP']
    assert all(int(res[k]) == score[v] for k,v in (('CLR_FP','FP'),('CLR_FN','FN'),('IDSW','IDSW')))
    return matched, truth, centers


def retain(gt, ids, t, tid, gid, horizon):
    if t+horizon >= len(gt) or any(gid not in g for g in gt[t:t+horizon+1]):
        return None
    return int(all(row.get(tid)==gid for row in ids[t:t+horizon+1]))


def pack_prior(e, target):
    delta = np.asarray(e['mean'])-target
    cov = np.asarray(e['cov'])
    xx,xy,yy = cov[0,0],cov[0,1],cov[1,1]
    det = xx*yy-xy**2
    assert det > 0
    x,y = delta
    d2 = (yy*x*x-2*xy*x*y+xx*y*y)/det
    return dict(covered=int(d2<=LIMIT), area=np.pi*LIMIT*np.sqrt(det), nll=.5*(np.log(det)+d2),
                prior_error=np.linalg.norm(delta))


def state_events(method, seq, ids, gt):
    priors, recov = {}, {}
    counts = Counter()
    for e in records(method,seq):
        kind = e['event']
        if kind not in ('lost','recovered'): continue
        counts[kind] += 1
        t, a, tid = e['frame'], e['anchor'], e['tid']
        gid = ids[a].get(tid)
        if gid is None or gid not in gt[t]: continue
        key = (a,e['anchor_det'],gid,t)
        if kind == 'lost':
            age = t-a
            if not 1 <= age <= 30: continue
            assert key not in priors, (method,seq,key)
            band = '1-2' if age<=2 else '3-5' if age<=5 else '6-10' if age<=10 else '11-30'
            priors[key] = (band, pack_prior(e,gt[t][gid]))
        else:
            counts['annotated_old_owner'] += 1
            if ids[t].get(tid) != gid: continue
            counts['same_gt'] += 1
            key = (*key,e['det'])
            assert key not in recov
            recov[key] = dict(post_error=np.linalg.norm(np.asarray(e['post'])-gt[t][gid]),
                    **{f'retain_{h}':retain(gt,ids,t,tid,gid,h) for h in (3,5,10)})
    return priors,recov,counts


def low_events(method,seq,ids,gt,centers,detections):
    events = {}
    for e in records(method,seq):
        if e['event']!='crowded_low': continue
        t,tid = e['frame'],e['tid']
        gid=ids[t-1].get(tid)
        if gid is None or ids[t].get(tid)!=gid: continue
        key = (t,e['det'],gid)
        assert key not in events
        box=detections[t][e['det']]
        assert .1 < box[4] < .5
        target=gt[t][gid]
        det_error=np.linalg.norm((box[:2]+box[2:4])*.5-target)
        prior_error=np.linalg.norm(np.asarray(e['prior'])-target)
        obs = dict(post_error=np.linalg.norm(np.asarray(e['post'])-target),det_error=det_error,
                   prior_error=prior_error)
        for h in (1,3):
            same = retain(gt,ids,t,tid,gid,h)
            obs[f'retain_{h}'] = same
            obs[f'error_{h}'] = (np.linalg.norm(centers[t+h][tid]-gt[t+h][gid]) if same else None)
        events[key]=obs
    return events


def append_values(groups,key,record):
    target=groups[key]
    for k,v in record.items():
        if v is not None: target[k].append(v)


def summarize(path, groups, columns):
    rows=[]
    for key, fields in sorted(groups.items()):
        row=dict(zip(columns,key))
        for name, v in fields.items():
            row[name+'_n']=len(v)
            row[name+'_mean']=float(np.mean(v))
            row[name+'_median']=float(np.median(v))
            if name.startswith('retain') or name=='covered':row[name+'_yes']=int(sum(v))
        rows.append(row)
    table(RESULT/path,rows)


def official_diagnosis(seqs, metrics):
    from botsort_plus.io import detection_files, detection_fingerprint, read_detection_cache, sequence_name
    group=lambda:defaultdict(lambda:defaultdict(list))
    clouds, connects, crowd, biased = group(),group(),group(),group()
    native=read(RUN/'protocols/native.json')
    sources={sequence_name(p):p for p in detection_files(Path(native['input_roots']['baseline'])/'detections')}
    coverage_counts=[]
    for seq,n in seqs.items():
        ps,rr,ll={},{},{}
        cache=read_detection_cache(sources[seq])
        assert detection_fingerprint(cache)==native['detection_fingerprints'][seq]
        dets={row['frame']:np.asarray(row['detections'],dtype=np.float32).reshape(-1,6) for row in cache['frames']}
        for m in ('native',*BAM,*QARMS):
            ids,gt,centers=owners(m,seq,n,metrics[m,seq])
            if m in ('native',*QARMS):
                ps[m],rr[m],c=state_events(m,seq,ids,gt)
                coverage_counts.append(dict(sequence=seq,method=m,**dict(c),anchored_priors_1_30=len(ps[m])))
            if m in ('native',*BAM):ll[m]=low_events(m,seq,ids,gt,centers,dets)
        common_p=set.intersection(*(set(v) for v in ps.values()))
        common_r=set.intersection(*(set(v) for v in rr.values()))
        common_l=set.intersection(*(set(v) for v in ll.values()))
        for k in sorted(common_l):
            ref=ll['native'][k]
            subset='det_farther_than_native_prior' if ref['det_error']>ref['prior_error'] else 'det_at_least_as_close'
            for m,v in ll.items():
                record=dict(v[k])
                for h in (1,3):
                    if all(g[k][f'retain_{h}']==1 for g in ll.values()):
                        record[f'shared_error_{h}']=v[k][f'error_{h}']
                append_values(biased,(subset,m),record)
        for m,p in ps.items():
            for k,(band,row) in p.items():
                append_values(clouds,('all_eligible',band,m),row)
                if k in common_p:append_values(clouds,('common',band,m),row)
            for k,row in rr[m].items():
                append_values(connects,('all_same_gt',m),row)
                if k in common_r:append_values(connects,('common',m),row)
        for m,data in ll.items():
            for k,row in data.items():
                append_values(crowd,('all_eligible',m),row)
                if k in common_l:append_values(crowd,('common',m),row)
            coverage_counts.append(dict(sequence=seq,method=m,low_native_eligible=len(ll['native']),low_eligible=len(data),
                                        low_common=len(common_l)))
        print('Audited',seq,flush=True)
    summarize('official_gap_calibration.csv',clouds,['cohort','age','method'])
    summarize('recovery_identity.csv',connects,['cohort','method'])
    summarize('crowded_low_followup.csv',crowd,['cohort','method'])
    summarize('crowded_low_bias_strata.csv',biased,['native_observation','method'])
    table(RESULT/'event_denominators.csv',coverage_counts)


def main():
    seqs=validate()
    out=RESULT/'metrics'
    subprocess.run([sys.executable,'-Bu',str(REPO/'tools/evaluate_botsort_plus.py'),'--run-dir',str(RUN),
         '--trackers',*METHODS,'--reference-tracker','native','--gt-root',read(SETTINGS)['gt_root'],
         '--trackeval-root',str(EVAL),'--output-dir',str(out)],check=True,env=environment())
    rows=read(out/'summary.json')['results']
    table(RESULT/'tracking_by_sequence.csv',rows)
    table(RESULT/'tracking_combined.csv',[v for v in rows if v['sequence']=='COMBINED_SEQ'])
    metrics={(v['tracker'],v['sequence']):v for v in rows}
    camera_diagnosis(seqs)
    official_diagnosis(seqs,metrics)
    print('OFFICIAL AUDIT COMPLETE',flush=True)


if __name__ == '__main__':
    main()
