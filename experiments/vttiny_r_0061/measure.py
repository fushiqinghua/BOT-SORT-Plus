"""Evaluate exact 29/30-sequence pools and audit the association term for every HOTA alpha."""
from collections import Counter
import csv
from unittest.mock import patch
import numpy as np
from rf_settings import QR,RF,OUT,CASE,SETTINGS,RUNS,read,write,table,sha
from evaluate_vttiny import VTTinyObjects
from trackeval.metrics import HOTA,CLEAR,Identity
import trackeval.metrics.hota as hota_module

def dataset(m,seqs):
    return VTTinyObjects(dict(GT_FOLDER=str(read(SETTINGS)['gt_root']),TRACKERS_FOLDER=str(RUNS[m]/'trackers'),
        TRACKERS_TO_EVAL=[m],OUTPUT_FOLDER=str(OUT/'tmp'),BENCHMARK='VT-Tiny-MOT',SPLIT_TO_EVAL='test',
        SKIP_SPLIT_FOL=True,SEQ_INFO=seqs,DO_PREPROC=False,PRINT_CONFIG=False))

def audit_hota(hota,data,raw):
    # TrackEval fixes the Hungarian assignment before it applies each alpha threshold.
    frames=iter(t for t,(g,p) in enumerate(zip(data['gt_ids'],data['tracker_ids'])) if len(g) and len(p))
    calls=[]
    solve=hota_module.linear_sum_assignment
    def record(cost):
        r,c=solve(cost)
        calls.append((next(frames),r,c))
        return r,c
    with patch.object(hota_module,'linear_sum_assignment',record):
        result=hota.eval_sequence(data)
    assert next(frames,None) is None
    gt=[arr[np.asarray(e['zero_marked'])!=0].astype(int) for arr,e in zip(raw['gt_ids'],raw['gt_extras'])]
    tids=raw['tracker_ids']
    lens=Counter(int(v) for a in gt for v in a)
    tlens=Counter(int(v) for a in tids for v in a)
    per_gt=[]; pairs=[];at_half=[]
    for i,alpha in enumerate(hota.array_labels):
        counts=Counter()
        for t,r,c in calls:
            good=data['similarity_scores'][t][r,c]>=alpha-np.finfo(float).eps
            counts.update((int(g),int(p)) for g,p in zip(gt[t][r[good]],tids[t][c[good]]))
            if np.isclose(alpha,.5):
                at_half.extend(dict(frame=t+1,gt=int(g),tid=int(p),iou=float(v)) for g,p,v in zip(
                    gt[t][r[good]],tids[t][c[good]],data['similarity_scores'][t][r[good],c[good]]))
        tp=sum(counts.values())
        assert tp==result['HOTA_TP'][i]
        numer=Counter()
        for (g,p),n in counts.items():
            assoc=n/(lens[g]+tlens[p]-n)
            numer[g]+=n*assoc
            if np.isclose(alpha,.5):pairs.append(dict(gt=g,tid=p,matched=n,gt_frames=lens[g],output_frames=tlens[p],association_iou=assoc))
        assert np.isclose(sum(numer.values())/max(tp,1),result['AssA'][i])
        per_gt.extend(dict(gt=g,alpha=float(alpha),assoc_contribution_pp=100*numer[g]/max(tp,1)) for g in sorted(lens))
    return result,per_gt,pairs,at_half

def values(h,c,j):
    return {**{k:float(np.mean(h[k])*100) for k in ('HOTA','DetA','AssA','AssRe','AssPr','LocA')},
            'IDF1':float(100*j['IDF1']),'MOTA':float(100*c['MOTA']),
            **{k:int(c[f]) for k,f in [('FP','CLR_FP'),('FN','CLR_FN'),('IDSW','IDSW'),('TP','CLR_TP')]}}

def main():
    with (RF/'results/four_way_metrics.csv').open(encoding='utf-8-sig') as file:
        official={(r['tracker'],r['sequence']):r for r in csv.DictReader(file)}
    checkpoint=read(QR/'results/score_audit.json')
    from pathlib import Path
    seqs=checkpoint['sequences'];gt_root=Path(read(SETTINGS)['gt_root'])
    for seq in seqs:assert sha(gt_root/seq/'gt/gt.txt')==checkpoint['gt_sha256'][seq]
    gtrows=[];pairrows=[];framepairs=[];poolrows=[];byseq=[];alphas=[];casevals={}
    for m,root in RUNS.items():
        proto=read(root/'protocols'/(m+'.json'))
        ctrl=read(QR/'full/protocols/native.json')
        for k in ('sequences','detection_fingerprints','feature_hashes','native_config','decoded_image_hashes'):
            assert proto[k]==ctrl[k]
        ds=dataset(m,seqs)
        metrics=(HOTA(),CLEAR({'PRINT_CONFIG':False}),Identity({'PRINT_CONFIG':False}))
        scores=[{} for _ in metrics]
        for seq in sorted(seqs):
            assert sha(root/'trackers'/m/'data'/(seq+'.txt'))==proto['output_prediction_hashes'][seq]
            raw=ds.get_raw_seq_data(m,seq)
            data=ds.get_preprocessed_seq_data(raw,'object')
            if seq==CASE:
                h,contrib,pairs,fr=audit_hota(metrics[0],data,raw)
                framepairs.extend(dict(method=m,**r) for r in fr)
                gtrows.extend(dict(method=m,**r) for r in contrib)
                pairrows.extend(dict(method=m,**r) for r in pairs)
                casevals[m]=h
                for i,a in enumerate(metrics[0].array_labels):
                    alphas.append(dict(method=m,alpha=round(float(a),2),**{k:float(h[k][i]) for k in ('HOTA','AssA','DetA','AssRe','AssPr','LocA','HOTA_TP','HOTA_FP','HOTA_FN')}))
            else:h=metrics[0].eval_sequence(data)
            scores[0][seq]=h
            for k in (1,2):scores[k][seq]=metrics[k].eval_sequence(data)
            row=values(*(s[seq] for s in scores))
            for key in ('HOTA','IDF1','DetA','AssA','MOTA','FP','FN','IDSW'):
                assert np.isclose(row[key],float(official[m,seq][key]),rtol=0,atol=1e-9),(m,seq,key)
            byseq.append(dict(method=m,sequence=seq,**row))
        for scope,subset in [('case',[CASE]),('other29',[s for s in seqs if s!=CASE]),('all30',list(seqs))]:
            row=values(*(metric.combine_sequences({s:store[s] for s in subset}) for metric,store in zip(metrics,scores)))
            if scope=='all30':
                for key in ('HOTA','IDF1','DetA','AssA','MOTA','FP','FN','IDSW'):
                    assert np.isclose(row[key],float(official[m,'COMBINED_SEQ'][key]),rtol=0,atol=1e-9)
            poolrows.append(dict(method=m,scope=scope,sequences=len(subset),frames=sum(seqs[s] for s in subset),**row))
    base,alt=casevals['native'],casevals['low_r']
    h00=base['HOTA'];h11=alt['HOTA']
    h10=np.sqrt(alt['DetA']*base['AssA']);h01=np.sqrt(base['DetA']*alt['AssA'])
    det=float(50*np.mean(h10-h00+h11-h01));assoc=float(50*np.mean(h01-h00+h11-h10))
    assert np.isclose(det+assoc,100*np.mean(h11-h00))
    tables={'pool_metrics':poolrows,'sequence_metrics':byseq,'hota_thresholds':alphas,'gt_association_thresholds':gtrows,'matched_id_pairs_alpha050':pairrows,'hota_matches_alpha050':framepairs}
    for name,rows in tables.items():table(OUT/(name+'.csv'),rows)
    write(OUT/'metric_audit.json',dict(status='PASS',historical_parity=True,gt_sha256=checkpoint['gt_sha256'][CASE],
          two_factor_order_averaged_hota_delta_pp={'DetA':det,'AssA':assoc},other29=sorted(set(seqs)-{CASE})))
    print(poolrows,flush=True)
if __name__=='__main__':main()
