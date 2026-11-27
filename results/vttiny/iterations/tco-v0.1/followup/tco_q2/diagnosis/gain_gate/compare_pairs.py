"""Compare archived calibration scores row for row and verify IoUs against official training GT."""
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
INPUT = HERE / 'inputs'
THRESHOLD = 0.005


def load_gt():
    result = defaultdict(list)
    for file in (INPUT/'gt').glob('*.txt'):
        with file.open() as f:
            for r in csv.reader(f):
                if float(r[6]) <= 0:
                    continue
                x, y, w, h = map(float, r[2:6])
                result[file.stem, int(r[0]), int(r[1])].append([x, y, x+w, y+h])
    return result


def iou(a, b):
    inter = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    return inter / (np.prod(a[2:]-a[:2]) + np.prod(b[2:]-b[:2]) - inter)


def transition(first, second):
    return ('both' if first else 'added') if second else ('removed' if first else 'neither')


def metrics(group):
    if not group:
        return {'n':0}
    result = dict(n=len(group), sequences=len(set(r['sequence'] for r in group)),
                  trajectories=len(set((r['sequence'], r['track_id']) for r in group)),
                  gates=dict(Counter(r['transition'] for r in group)))
    for name in ('raw_iou','refined_iou','true_gain','abs_gain','paired_gain','score_change',
                 'pixel_shift','raw_width','raw_height','abs_qraw','paired_qraw','abs_qref','paired_qref'):
        a = np.array([r[name] for r in group])
        result[name] = dict(mean=a.mean(), median=float(np.median(a)), q10=float(np.quantile(a,.1)),
                            q90=float(np.quantile(a,.9)), minimum=a.min(), maximum=a.max())
    for gain in ('abs_gain','paired_gain'):
        pred = np.array([r[gain] for r in group])
        target = np.array([r['true_gain'] for r in group])
        result[gain].update(negative=int((pred<0).sum()),passed=int((pred>=THRESHOLD).sum()),
                            sign_correct=int((np.sign(pred)==np.sign(target)).sum()))
    return result


def main():
    fixed = ('row','track_key','source_id','sequence_id','frame','assigned','support_label',
             'support_score','box_valid','raw_iou','refined_iou','raw_xyxy','refined_xyxy','adopted')
    gt = load_gt()
    gts = np.load(INPUT/'gt_xyxy.npy', mmap_mode='r')
    rows = []
    total = 0
    maxerr = 0.
    with (gzip.open(INPUT/'absolute_only.csv.gz','rt') as fa, gzip.open(INPUT/'paired_gain.csv.gz','rt') as fb,
          gzip.open(INPUT/'row_keys.jsonl.gz','rt') as fk):
        for i,(a,b,k) in enumerate(zip(csv.DictReader(fa),csv.DictReader(fb),map(json.loads,fk),strict=True)):
            total += 1
            assert all(a[n]==b[n] for n in fixed) and int(a['row'])==k['row']==i and int(a['frame'])==k['frame']
            if (a['assigned'],a['support_label'],a['box_valid']) != ('True','1','True'):
                continue
            assert a['adopted']=='False'
            official = gt[k['sequence'],k['frame'],k['gt_id']]
            assert len(official)==1 and np.allclose(official[0],gts[i],rtol=0,atol=1e-4)
            raw,ref = (np.array(json.loads(a[n])) for n in ('raw_xyxy','refined_xyxy'))
            truth = np.array(official[0])
            raw_iou,ref_iou = (float(a[n]) for n in ('raw_iou','refined_iou'))
            err = max(abs(iou(raw,truth)-raw_iou),abs(iou(ref,truth)-ref_iou))
            assert err < 1e-5
            maxerr=max(maxerr,err)
            qr,qs,pr,ps=(float(r[n]) for r in (a,b) for n in ('quality_raw','quality_refined'))
            ag,pg=qs-qr,ps-pr
            shift=((ref[:2]+ref[2:])-(raw[:2]+raw[2:]))/2
            rows.append(dict(row=i,sequence=k['sequence'],frame=k['frame'],track_id=k['track_id'],
                det_index=k['det_index'],gt_id=k['gt_id'],raw_xyxy=raw.tolist(),refined_xyxy=ref.tolist(),
                official_gt_xyxy=truth.tolist(),raw_iou=raw_iou,refined_iou=ref_iou,true_gain=ref_iou-raw_iou,
                abs_qraw=qr,abs_qref=qs,abs_gain=ag,paired_qraw=pr,paired_qref=ps,paired_gain=pg,
                score_change=pg-ag,transition=transition(ag>=THRESHOLD,pg>=THRESHOLD),
                crossing='down' if ref_iou<.5<=raw_iou else 'up' if raw_iou<.5<=ref_iou else 'none',
                support=float(a['support_score']),raw_width=raw[2]-raw[0],raw_height=raw[3]-raw[1],
                dx=shift[0],dy=shift[1],pixel_shift=np.linalg.norm(shift)))
    cross=[r for r in rows if r['crossing']!='none']
    down=[r for r in cross if r['crossing']=='down']
    up=[r for r in cross if r['crossing']=='up']
    assert (len(up),len(down))==(14,72)
    with (HERE/'crossings.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(cross[0]))
        w.writeheader()
        w.writerows(cross)
    result=dict(archived_rows=total,verified_gt_rows=len(rows),worst_iou_discrepancy=maxerr,
        groups={n:metrics(rr) for n,rr in dict(all=rows,up=up,down=down,
            positive=[r for r in rows if r['true_gain']>=.005],
            negative=[r for r in rows if r['true_gain']<=-.005],
            **{s:[r for r in down if r['transition']==s] for s in ('both','added','removed','neither')}).items()},
        by_sequence={s:dict(Counter(r['transition'] for r in down if r['sequence']==s))
                     for s in sorted({r['sequence'] for r in down})})
    (HERE/'calibration_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    for key,v in result['groups'].items():
        print(key,v['n'],v.get('gates'),[round(v[f]['median'],5) for f in ('true_gain','abs_gain','paired_gain')] if v['n'] else '')
    print(result['by_sequence'])


if __name__=='__main__':
    main()
