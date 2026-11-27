"""Join archived crossings, head interventions and matched fit neighbors by row."""
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent


def read(name):
    with (HERE/name).open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def geometry(box):
    values=np.array(json.loads(box))
    return (values[:2]+values[2:])/2, values[2:]-values[:2]


def summarize(rows):
    result={}
    for label,mask in {
        'added_down':lambda r:r['crossing']=='down' and r['transition']=='added',
        'track32_added':lambda r:r['crossing']=='down' and r['transition']=='added' and r['sequence']=='DJI_0067_2' and r['track_id']=='32',
        'exclude_track32':lambda r:r['crossing']=='down' and (r['sequence'],r['track_id'])!=('DJI_0067_2','32'),
        'exclude_0067':lambda r:r['crossing']=='down' and r['sequence']!='DJI_0067_2',
    }.items():
        ss=[r for r in rows if mask(r)]
        out=dict(count=len(ss),transitions=dict(Counter(r['transition'] for r in ss)))
        for key in ('dx','dy','needed_dx','needed_dy','pixel_shift','true_gain','archived_abs_gain',
                    'archived_paired_gain','paired_gain_visual','paired_gain_coordinates','cosine',
                    'visual_relative_l2','raw_width','raw_height','neighbor_mean_gain_5','neighbor_mean_gain_20'):
            out['median_'+key]=float(np.median([float(r[key]) for r in ss]))
        out['move_against_gt_direction']=sum(r['gt_shift_dot']<0 for r in ss)
        for k in ('archived_abs_gain','archived_paired_gain','paired_gain_visual','paired_gain_coordinates'):
            out[k+'_ge_005']=sum(float(r[k])>=.005 for r in ss)
        for n in (5,20):
            out['neighbors_positive_'+str(n)]=sum(float(r['neighbor_mean_gain_'+str(n)])>0 for r in ss)
        result[label]=out
    return result


def main():
    cases=read('crossings.csv')
    features={r['row']:r for r in read('feature_and_loss_controls.csv')}
    assert len(features)==3057 and len(cases)==86
    merged=[]
    for c in cases:
        f=dict(features[c['row']])
        assert all(abs(float(f[k])-float(c[v]))<2e-6
                   for k,v in [('absolute_only_gain','abs_gain'),('paired_gain_gain','paired_gain')])
        c['archived_abs_gain']=c.pop('abs_gain')
        c['archived_paired_gain']=c.pop('paired_gain')
        for key in ('absolute_only','paired_gain','paired_no_direction','paired_no_regression','paired_unit_cross'):
            f['replay_'+key]=f.pop(key)
        a,_=geometry(c['raw_xyxy'])
        b,_=geometry(c['official_gt_xyxy'])
        delta=b-a
        merged.append({**c,**f,'needed_dx':float(delta[0]),'needed_dy':float(delta[1]),
                       'gt_shift_dot':float(delta @ [float(c['dx']),float(c['dy'])])})
    with (HERE/'crossing_evidence.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(merged[0]))
        w.writeheader()
        w.writerows(merged)
    summary=summarize(merged)
    summary['unit_cross_vs_full_gate_differences_86']=sum(
        (float(r['replay_paired_unit_cross'])>=.005)!=(float(r['archived_paired_gain'])>=.005) for r in merged)
    assert summary['added_down']['count']==29 and summary['track32_added']['count']==21
    (HERE/'attribution_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    print({k:v['transitions'] for k,v in summary.items() if isinstance(v,dict)})


if __name__=='__main__':
    main()
