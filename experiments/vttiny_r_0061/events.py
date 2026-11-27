"""Recover exact CLEAR identity histories, including gaps and coincident switches."""
from collections import Counter,defaultdict
import csv
import gzip
from pathlib import Path
import numpy as np
from rf_settings import ROOT,RF,QR,CASE,RUNS,OUT,read,write,table,SETTINGS,sha
from measure import dataset
from truth_events import load_original_scorer,frame_maps
from botsort_plus.io import read_detection_cache,detection_fingerprint

def save_frames(rows):
    with gzip.open(OUT/'official_gt_frames.csv.gz','wt',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)

def time_pairs(a,b,radius,shared):
    # Preserve ordering. Maximize the number paired and then minimize displacement.
    from functools import lru_cache
    @lru_cache(None)
    def best(i,j):
        if i==len(a) or j==len(b):return (0,0,())
        options=[best(i+1,j),best(i,j+1)]
        d=abs(a[i]-b[j])
        if d<=radius and not any(min(a[i],b[j])<t<max(a[i],b[j]) for t in shared):
            n,cost,pairs=best(i+1,j+1)
            options.append((n+1,cost-d,((a[i],b[j]),)+pairs))
        return max(options,key=lambda item:item[:2])
    return best(0,0)[2]

def main():
    OUT.mkdir(exist_ok=True)
    cfg=read(SETTINGS);ctrl=read(QR/'results/score_audit.json')
    assert sha(Path(cfg['gt_root'])/CASE/'gt/gt.txt')==ctrl['gt_sha256'][CASE]
    cache=read_detection_cache(Path(cfg['baseline_root'])/'detections'/(CASE+'.json.gz'))
    dets=[np.asarray(f['detections'],np.float32).reshape(-1,6) for f in cache['frames']]
    layers={};counts={}
    scorer=load_original_scorer(QR)
    with (RF/'results/four_way_metrics.csv').open(encoding='utf-8-sig') as f:
        target={r['tracker']:r for r in csv.DictReader(f) if r['sequence']==CASE}
    for m in RUNS:
        prot=read(RUNS[m]/'protocols'/(m+'.json'))
        assert prot['detection_fingerprints'][CASE]==detection_fingerprint(cache)
        layers[m],counts[m]=frame_maps(scorer,dataset(m,{CASE:len(dets)}),m,CASE,RUNS[m],dets,target[m])
    rows=[];by_gt=defaultdict(dict);seen={m:{} for m in RUNS};switches={m:defaultdict(list) for m in RUNS}
    for it,(base,low) in enumerate(zip(layers['native'],layers['low_r']),1):
        assert base[0]==low[0] and np.array_equal(base[1],low[1])
        for i,g in enumerate(base[0]):
            g=int(g)
            row=dict(frame=it,gt=g,**dict(zip(('gt_x','gt_y','gt_w','gt_h'),map(float,base[1][i]))))
            for m in RUNS:
                hit=layers[m][it-1][2].get(g)
                prev=seen[m].get(g)
                changed=bool(hit and prev and hit[0]!=prev[1])
                row.update({m+'_'+k:v for k,v in zip(('tid','det','score','iou','x','y','w','h'),hit or (None,)*8)})
                row.update({m+'_last_frame':prev[0] if prev else None,m+'_last_tid':prev[1] if prev else None,m+'_switch':int(changed)})
                if hit:seen[m][g]=(it,hit[0])
                if changed:switches[m][g].append(it)
            rows.append(row);by_gt[g][it]=row
    save_frames(rows)
    assert all(sum(map(len,switches[m].values()))==counts[m]['IDSW'] for m in RUNS)
    ledgers=[];gtrows=[];pairrows=[]
    for g,hist in sorted(by_gt.items()):
        a,b=(set(switches[m][g]) for m in RUNS)
        same=a&b
        for rad in (1,3,5,10):
            for t,v in time_pairs(tuple(sorted(a-b)),tuple(sorted(b-a)),rad,same):
                pairrows.append(dict(gt=g,radius=rad,native_frame=t,low_frame=v,shift=v-t))
        for m,other,frames,otherframes in [('native','low_r',a,b),('low_r','native',b,a)]:
            for t in sorted(frames):
                row=hist[t];p=row[m+'_last_frame'];last=hist[p]
                u=last[other+'_tid'];v=row[other+'_tid']
                common_dets=bool(u is not None and v is not None and last[m+'_det']==last[other+'_det'] and row[m+'_det']==row[other+'_det'])
                intervening=sum(p<f<=t for f in otherframes)
                if t in same:kind='same_frame_switch'
                elif v is None:kind='other_misses_current'
                elif u is None:kind='other_misses_previous_anchor'
                elif u==v and not intervening:kind='retained_across_endpoints'
                else:kind='other_transition_in_interval'
                entry=dict(gt=g,method=m,frame=t,previous=p,gap=t-p,old_tid=row[m+'_last_tid'],new_tid=row[m+'_tid'],
                           detection=row[m+'_det'],score=row[m+'_score'],other_previous_tid=u,other_current_tid=v,
                           other_last_observed_frame=row[other+'_last_frame'],other_last_tid=row[other+'_last_tid'],
                           other_detection=row[other+'_det'],other_switches_between=intervening,same_detector_both_endpoints=common_dets,
                           kind=kind)
                for horizon in (3,5,10):
                    windows=[hist.get(k) for k in range(t,t+horizon+1)]
                    entry[f'continuous_at_{horizon}']=('unobservable' if any(w is None for w in windows) else
                        'same_id' if u is not None and all(w[other+'_tid']==u for w in windows) else 'not_held')
                ledgers.append(entry)
        gtrows.append(dict(gt=g,frames=len(hist),native_switches=len(a),low_r_switches=len(b),net_saved=len(a)-len(b),
                           both_same_frame=len(same),native_only=len(a-b),low_r_only=len(b-a),
                           native_tp=sum(r['native_tid'] is not None for r in hist.values()),
                           low_r_tp=sum(r['low_r_tid'] is not None for r in hist.values())))
    table(OUT/'switch_ledger.csv',ledgers);table(OUT/'by_gt.csv',gtrows);table(OUT/'nearby_switches.csv',pairrows)
    stat={m:dict(Counter(r['kind'] for r in ledgers if r['method']==m)) for m in RUNS}
    write(OUT/'event_audit.json',dict(status='PASS',counts=counts,classification=stat,
        near_pairs={str(rad):sum(r['radius']==rad for r in pairrows) for rad in (1,3,5,10)},frames=len(dets)))
    print(stat,flush=True)
if __name__=='__main__':main()
