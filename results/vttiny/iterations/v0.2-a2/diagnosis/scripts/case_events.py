"""Extract original online decisions for the illustrated windows and validate spatial FP examples."""
import argparse
import csv
import gzip
import json
import sqlite3
from collections import Counter,defaultdict
from pathlib import Path

from build_evidence import METHODS
from render_cases import CASES, box, iou


# Each pivot is an observed A2B output event, in one-based MOT time.
PIVOTS=[(8,32),(6,7),(132,50),(5,12),(302,33),(181,266),(188,250),(78,1)]


def read_rows(path):
    with path.open(encoding="utf-8-sig") as f:return list(csv.DictReader(f))


def collect(full, diag):
    analysis=diag/"analysis"; switches=read_rows(analysis/"switches.csv");targets=read_rows(analysis/"targets.csv")
    db=sqlite3.connect(analysis/"evidence.sqlite");db.row_factory=sqlite3.Row
    summaries=[]
    for case,(pivot,pivot_id) in zip(CASES,PIVOTS):
        folder,seq,kind,gtid,windows,keys,title,_=case
        dest=diag/"cases"/folder
        timeline=read_rows(dest/"timeline.csv")
        visible=set(t for a,b in windows for t in range(a,b+1))
        all_rows=defaultdict(list)
        for x in db.execute("SELECT * FROM outputs WHERE sequence=?",(seq,)):
            all_rows[x['method'],x['frame']].append(dict(x))
        o=next(x for x in all_rows['A2B',pivot] if x['tid']==pivot_id)
        det=o['det_idx']
        example=dict(sequence=seq,title=title,kind=kind,focus_id=gtid,
                     pivot_frame=pivot,pivot_A2B_tid=pivot_id,pivot_det_index=det,methods={})
        if kind=='gt':
            example['target']=next(z for z in targets if z['sequence']==seq and int(z['gt'])==gtid)
        else:
            traj=[x for (m,_),part in all_rows.items() if m=='A2B' for x in part if x['tid']==gtid]
            example['whole_track']=dict(output_frames=len(traj),first=min(x['frame'] for x in traj),
                last=max(x['frame'] for x in traj),max_iou_gt=max(x['max_iou'] for x in traj))
            for m in ('B_ref','A2B_score020'):
                example['whole_track'][m+'_max_spatial_iou']=max(
                    (iou(box(x),box(y)) for x in traj for y in all_rows[m,x['frame']]),default=0.)
                example['whole_track'][m+'_same_detector_row_frames']=sum(
                    any(y['det_idx']==x['det_idx'] for y in all_rows[m,x['frame']]) for x in traj)
        with gzip.open(dest/"selected_events.jsonl.gz","wt",encoding='utf-8',newline='\n') as clip:
            for m in METHODS:
                track_ids={int(x[m+'_shown_tid']) for x in timeline if int(x['frame']) in visible and x[m+'_shown_tid']}
                window_switches=[x for x in switches if x['method']==m and x['sequence']==seq and
                                 kind=='gt' and int(x['gt'])==gtid and int(x['frame']) in visible]
                for sw in window_switches:track_ids.update(int(sw[k]) for k in ('old_tid','new_tid'))
                events=[]
                path=full/'events'/m/(seq+'.jsonl')
                source=path.open(encoding='utf-8') if path.exists() else gzip.open(str(path)+'.gz','rt',encoding='utf-8')
                with source as f:
                    for line in f:
                        item=json.loads(line)
                        if item['frame'] in visible:events.append(item)
                pivot_events=[e for e in events if e['frame']==pivot]
                focus_cid=next((e['candidate_id'] for e in pivot_events if e.get('det_index')==det and
                               e['event'] in ('candidate_seed','candidate_hit','birth_committed','recovered')),None)
                candidates={e['candidate_id'] for e in events if e['event'] in ('recovered','birth_committed')
                            and e['track_id'] in track_ids}
                if focus_cid is not None:candidates.add(focus_cid)
                states=[e for e in pivot_events if focus_cid is not None and e.get('candidate_id')==focus_cid]
                guard=next((e for e in states if e['event']=='birth_identity'),{})
                prior={int(x['old_tid']) for x in window_switches if int(x['frame'])==pivot}
                example['methods'][m]=dict(pivot_candidate=focus_cid,
                    actual_row_updates=[e for e in pivot_events if e.get('det_index')==det and
                                        e['event'] in ('match','birth_committed','native_birth','recovered')],
                    pivot_evidence=next((e for e in states if e['event']=='birth_evidence'),None),
                    identity_status=guard.get('status'),guard_reason_counts=dict(Counter(v['reason'] for v in guard.get('diagnostics',[]))),
                    previous_GT_id_guard=[v for v in guard.get('diagnostics',[]) if v['track_id'] in prior],
                    previous_GT_id_recovery_check=[v for v in states if v['event']=='recovery_check' and v['track_id'] in prior],
                    switches_in_window=window_switches)
                for e in events:
                    if e.get('track_id') in track_ids or e.get('candidate_id') in candidates:
                        clip.write(json.dumps({'method':m,**e},ensure_ascii=False,separators=(',',':'))+'\n')
        (dest/"case_evidence.json").write_text(json.dumps(example,indent=2,ensure_ascii=False),encoding='utf-8')
        summaries.append((dest,example))
    assert summaries[3][1]['whole_track']['B_ref_max_spatial_iou']==0
    assert summaries[3][1]['whole_track']['A2B_score020_same_detector_row_frames']==170
    assert summaries[4][1]['whole_track']['A2B_score020_max_spatial_iou']==0
    for m in METHODS[1:]:
        rec=summaries[5][1]['methods'][m]['previous_GT_id_guard']
        assert len(rec)==1 and rec[0]['reason']=='current_appearance_excludes' and rec[0]['current_appearance']>.65
    db.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(__doc__);p.add_argument("--run-dir",type=Path,default=Path(__file__).resolve().parents[2]/"full30");p.add_argument("result",type=Path)
    args=p.parse_args();collect(args.run_dir,args.result)
