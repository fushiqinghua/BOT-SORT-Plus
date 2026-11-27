"""Undo selected low-R updates on one track; evaluate the subsequent identity independently."""
import argparse
import csv
import gzip
from pathlib import Path
import sys
import subprocess
from unittest.mock import patch
import numpy as np
from rf_settings import CASE,ROOT,OUT,RF,QR,SETTINGS,RUNS,read,write,sha,environment
from measure import dataset
from truth_events import load_original_scorer,frame_maps
from botsort_plus.io import read_detection_cache
import branch_replay as br
from boxmot.trackers.botsort.botsort_track import STrack

# These are retrospective interventions on recorded events, not tuning configurations.
CASES={'gt31_last':(31,63,104,104,109),'gt31_window':(31,63,101,104,109),
       'gt82_window':(82,240,205,208,228),'gt101_last':(101,120,108,108,119)}

def run(name):
    gt,tid,start,end,event=CASES[name]
    directory=ROOT/'event_replays'/name
    br.CASE=CASE;br.ROOT=directory
    original_factory=br.configure_arm
    br.configure_arm=lambda base,method,paths,proto,output,detailed:original_factory(base,method,paths,proto,output,False)
    changed=[]
    def instrument(original):
        def step(track,obs,t,*args,**kwargs):
            if start<=t<=end and track.id==tid and .10<float(obs.conf)<.50:
                def native(mean,cov,z,var):
                    changed.append(dict(frame=t,tid=tid,det=int(obs.det_ind),score=float(obs.conf)))
                    return type(track.kalman_filter).update(track.kalman_filter,mean,cov,z)
                with patch.object(br,'posterior',native):return original(track,obs,t,*args,**kwargs)
            return original(track,obs,t,*args,**kwargs)
        return step
    with patch.object(STrack,'update',instrument(STrack.update)),patch.object(STrack,'re_activate',instrument(STrack.re_activate)):
        br.main('pilot','low_r')
    assert len(changed)>0 and len({z['frame'] for z in changed})==len(changed)
    def previous(path):
        return [l for l in path.read_bytes().splitlines(keepends=True) if int(l.split(b',')[0])<start]
    mot=Path('trackers/low_r/data')/(CASE+'.txt')
    run_dir=directory/'pilot'
    assert previous(run_dir/mot)==previous(RUNS['low_r']/mot)
    write(directory/'interventions.json',changed)

def score_probe(name):
    gt,tid,start,end,event=CASES[name]
    directory=ROOT/'event_replays'/name
    run_dir=directory/'pilot'
    changed=read(directory/'interventions.json')
    cfg=read(SETTINGS);dc=read_detection_cache(Path(cfg['baseline_root'])/'detections'/(CASE+'.json.gz'))
    ds=[np.asarray(f['detections'],np.float32).reshape(-1,6) for f in dc['frames']]
    data=dataset('low_r',{CASE:len(ds)})
    data.tracker_fol=str(run_dir/'trackers')
    own,counts=frame_maps(load_original_scorer(QR),data,'low_r',CASE,run_dir,ds,None)
    with gzip.open(OUT/'official_gt_frames.csv.gz','rt',encoding='utf-8-sig') as file:
        gt_hist={int(r['frame']):r for r in csv.DictReader(file) if int(r['gt'])==gt}
    evidence=[]
    for t in range(start,min(event+11,len(ds)+1)):
        if t not in gt_hist:continue
        row=gt_hist[t]
        current=own[t-1][2].get(gt)
        evidence.append(dict(frame=t,old_r_track=tid,
            native_id=int(row['native_tid']) if row['native_tid'] else None,
            low_id=int(row['low_r_tid']) if row['low_r_tid'] else None,
            probe_id=current[0] if current else None,probe_det=current[1] if current else None))
    write(OUT/(name+'.json'),dict(gt=gt,event=event,interventions=changed,prior_output_parity=True,counts=counts,frames=evidence))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name',nargs='?',choices=list(CASES));p.add_argument('--score',action='store_true');a=p.parse_args()
    if a.name:(score_probe if a.score else run)(a.name)
    else:
        for name in CASES:
            if not (ROOT/'event_replays'/name/'interventions.json').is_file():
                subprocess.run([sys.executable,'-Bu',__file__,name],env=environment(),check=True)
            subprocess.run([sys.executable,'-Bu',__file__,name,'--score'],env=environment(),check=True)
        print('Verified four event interventions',flush=True)
