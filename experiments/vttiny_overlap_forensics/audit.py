"""Compare frozen detections, output rows, and official CLEAR results across controlled replays."""
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from unittest.mock import patch
from workflow import FOCUS, JOINT, ROOT
from joint_settings import COMBINED, CONFIG, FIX, protocol, read, write, write_csv

OUT=ROOT/'results'
GROUPS={'fixed':FIX,'combo':COMBINED}


def trace(path):
    with gzip.open(path,'rt') as f:return [json.loads(line) for line in f]


def name(mode):
    return GROUPS.get(mode,COMBINED)


def paths(mode):
    return ROOT/'runs'/mode/'full'


def output_matches(mode,seq,frames,detections):
    from evaluate_vttiny import VTTinyObjects
    from trackeval.metrics import CLEAR
    import trackeval.metrics.clear as clear
    method=name(mode)
    ds=VTTinyObjects(dict(GT_FOLDER=read(CONFIG)['gt_root'],TRACKERS_FOLDER=str(paths(mode)/'trackers'),
       TRACKERS_TO_EVAL=[method],BENCHMARK='VT-Tiny-MOT',SPLIT_TO_EVAL='test',SKIP_SPLIT_FOL=True,
       DO_PREPROC=False,PRINT_CONFIG=False,SEQ_INFO={seq:frames}))
    raw=ds.get_raw_seq_data(method,seq)
    data=ds.get_preprocessed_seq_data(raw,'object')
    traced=trace(ROOT/'runs'/mode/'trace'/(seq+'.jsonl.gz'))
    assert len(traced)==frames
    times=iter(t for t in range(frames) if len(data['gt_ids'][t]) and len(data['tracker_ids'][t]))
    captured={}
    def solve(cost):
        t=next(times)
        ii,jj=linear_sum_assignment(cost)
        ok=-cost[ii,jj]>np.finfo(float).eps
        captured[t]={int(j):int(i) for i,j in zip(ii[ok],jj[ok])}
        return ii,jj
    with patch.object(clear,'linear_sum_assignment',solve):
        score=CLEAR({'PRINT_CONFIG':False}).eval_sequence(data)
    assert next(times,None) is None
    outputs={}; switches={}; misses=set(); totals=Counter()
    last={}; lineage={}; recent={}
    gt_rows=[]
    for ti,entry in enumerate(traced):
        t=ti+1
        assert t==entry['frame']
        dsix={int(x[0]):int(x[1]) for x in entry['outputs']}
        gt_mask=np.asarray(raw['gt_extras'][ti]['zero_marked'])!=0
        gt=np.asarray(raw['gt_ids'][ti])[gt_mask]
        gt_boxes=np.asarray(raw['gt_dets'][ti]).reshape(-1,4)[gt_mask]
        dets=np.array(detections[ti]['detections'],np.float32).reshape(-1,6)
        dboxes=dets[:,:4].astype(float).copy(); dboxes[:,2:]-=dboxes[:,:2]
        raw_iou=ds._calculate_similarities(gt_boxes,dboxes)
        similarities=data['similarity_scores'][ti]
        assigned=captured.get(ti,{})
        truth_now={int(gt[i]):int(raw['tracker_ids'][ti][j]) for j,i in assigned.items()}
        for i,g in enumerate(gt):
            gt_rows.append(dict(sequence=seq,frame=t,GT=int(g),x=gt_boxes[i,0],y=gt_boxes[i,1],
                         width=gt_boxes[i,2],height=gt_boxes[i,3],ID=truth_now.get(int(g)),
                         det=int(dsix[truth_now[int(g)]]) if int(g) in truth_now else ''))
        decisions={}
        for e in entry['events']:
            kind=e['event']; tid=e.get('track_id')
            if kind in ('match','recovered','native_birth','motion_output') and tid is not None:
                decisions[tid]=e
        for e in entry['bam']:
            if e['decision']=='weight':recent[e['tid']]=(t,e['det'],e['weight'])
        for j,tid in enumerate(map(int,raw['tracker_ids'][ti])):
            d=dsix[tid]
            assert 0<=d<len(dets) and float(dets[d,4])==raw['tracker_confidences'][ti][j],(mode,seq,t,tid,d)
            assert (t,d) not in outputs,(mode,seq,t,d)
            e=decisions.get(tid,{})
            action=e.get('stage',e.get('event','other'))
            if action=='recovered':
                lineage[tid]=('B',t,d,e.get('age'),e.get('cost'),e.get('appearance_cost'),e.get('hits'),e.get('geometry_cost'))
            elif tid not in lineage or last.get(tid)!=t-1:
                lineage[tid]=('native',t,d,None,None,None,None,None)
            last[tid]=t
            g=int(gt[assigned[j]]) if j in assigned else None
            box=raw['tracker_dets'][ti][j]
            max_pred=float(similarities[:,j].max()) if len(gt) else 0.
            max_det=float(raw_iou[:,d].max()) if len(gt) else 0.
            origin,origin_t,origin_d,age,cost,app,hits,geom=lineage[tid]
            last_bam=recent.get(tid,(None,None,None))
            outputs[t,d]=dict(sequence=seq,frame=t,det=d,tid=tid,stage=action,gt=g,
                status='TP' if g is not None else 'FP',score=float(dets[d,4]),pred_iou=max_pred,det_iou=max_det,
                nearest_gt=int(gt[int(similarities[:,j].argmax())]) if max_pred>0 else None,
                nearest_det_gt=int(gt[int(raw_iou[:,d].argmax())]) if max_det>0 else None,
                x=float(box[0]),y=float(box[1]),width=float(box[2]),height=float(box[3]),
                det_x1=float(dets[d,0]),det_y1=float(dets[d,1]),det_x2=float(dets[d,2]),det_y2=float(dets[d,3]),
                origin=origin,origin_frame=origin_t,origin_det=origin_d,recovery_age=age,recovery_cost=cost,
                recovery_appearance=app,recovery_hits=hits,recovery_geometry=geom,prior_weight_frame=last_bam[0],
                prior_weight_det=last_bam[1],prior_weight=last_bam[2])
            if g is not None:
                old=last.get(('gt',g))
                if old is not None and old[1]!=tid:switches[t,g]=(old[0],old[1],tid)
                last['gt',g]=(t,tid)
            totals['TP' if g is not None else 'FP']+=1
        misses.update((t,int(g)) for g in gt if int(g) not in truth_now)
    totals.update(FN=len(misses),IDSW=len(switches))
    assert all(totals[k]==int(score[v]) for k,v in (('TP','CLR_TP'),('FP','CLR_FP'),('FN','CLR_FN'),('IDSW','IDSW')))
    return outputs,switches,misses,totals,gt_rows


def compare(seq,before,after):
    b,b_sw,b_fn,b_tot,b_gt=before
    a,a_sw,a_fn,a_tot,a_gt=after
    transitions=Counter(); causes=Counter(); changes=[]
    episodes=defaultdict(list)
    for key in sorted(b.keys()|a.keys()):
        left,right=b.get(key),a.get(key)
        x=left['status'] if left else 'absent'
        y=right['status'] if right else 'absent'
        if (x=='FP')==(y=='FP'):continue
        entered=y=='FP'
        fp=right if entered else left
        kind=('no_raw_gt_overlap' if fp['det_iou']==0 else 'raw_det_below_half' if fp['det_iou']<.5 else
              'raw_det_passes_but_output_below_half' if fp['pred_iou']<.5 else 'official_assignment_conflict')
        move=x+'->'+y
        origin=fp['origin']
        transitions[move]+=1
        causes['new' if entered else 'removed',origin,kind]+=1
        row=dict(sequence=seq,frame=key[0],det=key[1],transition=move,spatial_class=kind)
        fields=list(fp)[3:]
        for prefix,o in [('fixed',left),('combo',right)]:
            row.update({prefix+'_'+k:o[k] if o else '' for k in fields})
        changes.append(row)
        episodes[('added' if entered else 'removed',fp['tid'],origin,fp['origin_frame'],fp['origin_det'])].append((key[0],fp))
    episode_rows=[]
    for (side,tid,origin,start,sd),v in sorted(episodes.items(),key=lambda x:-len(x[1])):
        obj=v[0][1]
        episode_rows.append(dict(sequence=seq,side=side,tid=tid,origin=origin,start_frame=start,start_det=sd,
            count=len(v),first_FP=v[0][0],last_FP=v[-1][0],frames=' '.join(str(x[0]) for x in v),
            recovery_age=obj['recovery_age'],recovery_cost=obj['recovery_cost'],
            recovery_appearance=obj['recovery_appearance'],recovery_hits=obj['recovery_hits'],
            last_bam_at_first_FP=obj['prior_weight_frame']))
    switches=[]
    for t,g in sorted(b_sw.keys()|a_sw.keys()):
        switches.append(dict(sequence=seq,frame=t,GT=g,fixed=int((t,g) in b_sw),combo=int((t,g) in a_sw),
            fixed_history=b_sw.get((t,g)),combo_history=a_sw.get((t,g))))
    summaries=dict(sequence=seq,fixed={k:b_tot[k] for k in ('FP','FN','IDSW')},combo={k:a_tot[k] for k in ('FP','FN','IDSW')},
        transitions=dict(transitions),causes={'|'.join(k):v for k,v in sorted(causes.items())},
        new_misses=len(a_fn-b_fn),restored=len(b_fn-a_fn))
    assert sum(v if k.endswith('>FP') else -v for k,v in transitions.items())==a_tot['FP']-b_tot['FP']
    return summaries,changes,episode_rows,switches


def main():
    from botsort_plus.io import detection_files,read_detection_cache,sequence_name,detection_fingerprint
    original=read(JOINT/'results/metrics/summary.json')['results']
    official={(r['tracker'],r['sequence']):r for r in original}
    inputs={sequence_name(p):p for p in detection_files(Path(read(CONFIG)['baseline_root'])/'detections')}
    modes=['fixed','combo']
    for root in sorted((ROOT/'runs').iterdir()):
        p=root/'full/protocols'/(COMBINED+'.json')
        if root.name not in modes and not root.name.startswith('full_') and p.exists() and read(p)['status']=='COMPLETE':
            modes.append(root.name)
    summaries=[];diff=[];episodes=[];switches=[];table=[];trace_rows=[]
    interventional=[]
    timelines=[]
    for mode in modes:
        p=protocol(paths(mode),name(mode))
        old=protocol(JOINT/'runs'/name(mode)/'full',name(mode))
        for k in ('feature_hashes','decoded_image_hashes','detection_fingerprints','native_config'):
            assert p[k]==({s:old[k][s] for s in FOCUS} if k!='native_config' else old[k]),(mode,k)
        if mode in GROUPS:assert p['output_prediction_hashes']=={s:old['output_prediction_hashes'][s] for s in FOCUS}
    for seq in FOCUS:
        cache=read_detection_cache(inputs[seq])
        ref=protocol(paths('combo'),COMBINED)
        assert ref['detection_fingerprints'][seq]==detection_fingerprint(cache)
        results={}
        for mode in modes:
            result=output_matches(mode,seq,ref['sequences'][seq],cache['frames'])
            for key in ('FP','FN','IDSW'):
                if mode in GROUPS:assert result[3][key]==official[name(mode),seq][key]
            actual=read(ROOT/'runs'/mode/'trace'/(seq+'.json'))
            assert sum(e[0]=='block_b' for e in actual['applied'])==actual['expected_block_b'],(mode,seq)
            assert sum(e[0]=='undo' for e in actual['applied'])>=actual['expected_undo'],(mode,seq)
            table.append(dict(sequence=seq,method=mode,FP=result[3]['FP'],FN=result[3]['FN'],
                              IDSW=result[3]['IDSW'],TP=result[3]['TP'],interventions=actual['applied']))
            results[mode]=result
            if mode=='output_only':
                original_trace=trace(ROOT/'runs/combo/trace'/(seq+'.jsonl.gz'))
                output_trace=trace(ROOT/'runs/output_only/trace'/(seq+'.jsonl.gz'))
                assert [(r['events'],r['outputs']) for r in original_trace]==[(r['events'],r['outputs']) for r in output_trace]
            windows={'DJI_0028_3':((58,273,353),(46,275,352)),
                     'DJI_0061_2':((149,117,177),(142,112,151)),
                     'DJI_0169_3':((15,127,201),)}
            for tid,first,last in windows[seq]:
                indexed={v['frame']:v for v in result[0].values() if v['tid']==tid}
                for t in range(first,last+1):
                    item=indexed.get(t,{})
                    fields=('det','stage','status','gt','score','det_iou','pred_iou','origin','origin_frame',
                            'recovery_hits','recovery_cost','prior_weight_frame','prior_weight')
                    timelines.append(dict(sequence=seq,method=mode,tid=tid,frame=t,
                                          **{k:item.get(k,'') for k in fields}))
        s,d,e,w=compare(seq,results['fixed'],results['combo'])
        summaries.append(s);diff.extend(d);episodes.extend(e);switches.extend(w)
        original_new={(r['frame'],r['det']) for r in d if r['transition'].endswith('->FP')}
        combo_fp={k for k,v in results['combo'][0].items() if v['status']=='FP'}
        for mode in modes[2:]:
            trial_fp={k for k,v in results[mode][0].items() if v['status']=='FP'}
            removed_switches=results['fixed'][1].keys()-results['combo'][1].keys()
            new_switches=results[mode][1].keys()-results['combo'][1].keys()
            interventional.append(dict(sequence=seq,method=mode,original_gross_FP_removed=len(original_new-trial_fp),
                combo_FP_removed=len(combo_fp-trial_fp),extra_FP_vs_combo=len(trial_fp-combo_fp),
                removed_switch_positions_return=len(removed_switches&new_switches),
                combo_FN_to_TP=len(results['combo'][2]-results[mode][2]),combo_TP_to_FN=len(results[mode][2]-results['combo'][2])))
        for mode in GROUPS:
            for v in results[mode][0].values():trace_rows.append(dict(method=mode,**v))
        # Coordinates and assignments of every annotated target in each frame.
        fixed_gt=results['fixed'][4]
        combo_gt=results['combo'][4]
        rows=[]
        for a,b in zip(fixed_gt,combo_gt,strict=True):
            assert all(a[k]==b[k] for k in ('sequence','frame','GT','x','y','width','height'))
            rows.append(dict(a,combo_ID=b['ID'],combo_det=b['det']))
        write_csv(OUT/(seq+'_逐帧官方真值配对.csv'),rows)
    write(OUT/'summary.json',summaries)
    write_csv(OUT/'case_metrics.csv',table)
    write_csv(OUT/'changed_false_positives.csv',diff)
    write_csv(OUT/'fp_episodes.csv',episodes)
    write_csv(OUT/'switches.csv',switches)
    write_csv(OUT/'typical_track_timelines.csv',timelines)
    if interventional:write_csv(OUT/'intervention_effects.csv',interventional)
    plain=OUT/'output_rows.csv'
    write_csv(plain,trace_rows)
    with gzip.open(OUT/'output_rows.csv.gz','wb',compresslevel=6) as f:f.write(plain.read_bytes())
    plain.unlink()
    print('Official paired FP net',[(s['sequence'],s['combo']['FP']-s['fixed']['FP']) for s in summaries],flush=True)


if __name__=='__main__':main()
