"""Extract real video crops and quantify consecutive, low-score evidence in missed annotations."""
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from diagnose_vttiny_botsort_m import OUTPUT, GT_FOLDER, RUN, frame_path, load, load_detections, mot, overlaps


CASES = [
    ('A', 'DJI_0327_3', 40, (1,195,390), 'Tiny object with no nearby detection'),
    ('B', 'DJI_0030_2', 5, (263,304,345), 'Repeated low-confidence detections'),
    ('C', 'DJI_0141_2', 7, (1,77,155), 'Persistent box overlap below 0.50'),
    ('D', 'DJI_0061_2', 9, (101,108,116), 'Identity change after 14 missed frames'),
]
GREEN, YELLOW, BLUE = '#31eb82', '#ffbd3d', '#4dd6ff'
DARK, LIGHT = '#0e1721', '#e5eff5'


def font(n):
    try:
        return ImageFont.truetype('DejaVuSans.ttf', n)
    except OSError:
        return ImageFont.load_default(size=n)


def context(frame, g, det_box, track_box):
    w,h = frame.size
    extent = min(w,h,max(68, int(3*max(g[2:]))))
    left = min(max(0,int(g[0]+g[2]/2-extent/2)), w-extent)
    top = min(max(0,int(g[1]+g[3]/2-extent/2)), h-extent)
    roi = (left,top,left+extent,top+extent)
    sub = frame.crop(roi).resize((240,240),Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(sub)
    def outline(box, color, width):
        if box is not None:
            x,y,bw,bh = box
            draw.rectangle([(x-left)*240/extent,(y-top)*240/extent,
                            (x+bw-left)*240/extent,(y+bh-top)*240/extent],outline=color,width=width)
    outline(track_box,BLUE,4)
    outline(det_box,YELLOW,2)
    outline(g,GREEN,1)
    return sub,roi


def render(d, out):
    sheet = Image.new('RGB',(818,4*355+82),DARK)
    draw = ImageDraw.Draw(sheet)
    draw.text((20,9),'BoT-SORT  |  VT-Tiny-MOT official test',font=font(23),fill=LIGHT)
    for left, text, color in ((20,'Ground truth',GREEN),(230,'Cached detection',YELLOW),(510,'Output track',BLUE)):
        draw.text((left,46),text,font=font(17),fill=color)
    records = []
    for row,(code,seq,gid,fs,title) in enumerate(CASES):
        cache = load_detections(RUN/'rgbt/detections', seq)['frames']
        n = len(cache)
        gt = mot(GT_FOLDER/seq/'gt/gt.txt',n,True)
        preds = mot(RUN/'rgbt/trackers/botsort_reid/data'/(seq+'.txt'),n)
        y = 77+row*355
        draw.text((20,y),f'{code}   {seq}   GT {gid}   |   {title}',font=font(18),fill=LIGHT)
        for col,f in enumerate(fs):
            entry = d[(d.sequence==seq)&(d.gt_id==gid)&(d.frame==f)].iloc[0]
            g = gt[f-1][gt[f-1][:,0]==gid][0,1:5]
            raw = np.asarray(cache[f-1]['detections']).reshape(-1,6)
            db = raw[:,:4].copy()
            db[:,2:] -= db[:,:2]
            di = overlaps(g[None],db)[0]
            aligned = np.flatnonzero(di >= .5)
            j = (aligned[raw[aligned,4].argmax()] if len(aligned) else di.argmax()) if di.size and di.max()>=.1 else None
            pr = preds[f-1]
            ti = overlaps(g[None],pr[:,1:5])[0]
            t = int(entry.track_id)
            k = np.flatnonzero(pr[:,0]==t)[0] if t >= 0 else (ti.argmax() if ti.size and ti.max()>=.1 else None)
            path = frame_path(cache[f-1]['image'])
            with Image.open(path) as im:
                sub,roi = context(im.convert('RGB'),g,None if j is None else db[j],None if k is None else pr[k,1:5])
            x = 20+col*264
            sheet.paste(sub,(x,y+35))
            draw.text((x,y+278),f'Frame {f}  |  GT {g[2]:g} x {g[3]:g} px',font=font(15),fill=LIGHT)
            ds = 'No det. at IoU >= 0.10' if j is None else f'Det. {raw[j,4]:.2f}  |  IoU {di[j]:.2f}'
            ts = (f'Matched track {t}' if t >= 0 else 'Miss' if k is None else f'Miss  |  nearby track {int(pr[k,0])}')
            draw.text((x,y+301),ds,font=font(15),fill=YELLOW)
            draw.text((x,y+323),ts,font=font(15),fill=BLUE)
            records.append(dict(case=code,sequence=seq,gt_id=gid,frame=f,source=str(path),crop=roi,
                                gt=g.tolist(),matched_track=t,shown_detector=None if j is None else raw[j].tolist(),
                                shown_detector_iou=None if j is None else float(di[j]),
                                shown_track=None if k is None else pr[k].tolist()))
    sheet.save(out/'cases.png',optimize=True)
    (out/'case_index.json').write_text(json.dumps(records,indent=2,ensure_ascii=False),encoding='utf-8')


def extra(d, out):
    d = d.sort_values(['sequence','gt_id','frame'])
    q = d[(d.track_id==-1)&(d.candidate=='low_assoc')].copy()
    run = (q.sequence.ne(q.sequence.shift())|q.gt_id.ne(q.gt_id.shift())|q.frame.diff().ne(1)).cumsum()
    episodes = q.groupby(run).agg(sequence=('sequence','first'),gt_id=('gt_id','first'),start=('frame','min'),
                                 end=('frame','max'),frames=('frame','size'),median_score=('max_aligned_score','median'))
    episodes.to_csv(out/'low_score_runs.csv',index=False)
    counts = {n:int(episodes.loc[episodes.frames>=n,'frames'].sum()) for n in (1,3,5,10)}
    without = d[d.sequence!='DJI_0327_3']
    totals = {}
    for key,group in without.groupby('size'):
        totals[key] = [len(group), int((group.track_id==-1).sum())]
    checks = [('DJI_0327_3',40,1,390,'overlap_lt10'),('DJI_0030_2',5,263,345,'low_assoc'),('DJI_0141_2',7,1,155,'overlap_30_50')]
    for seq,gid,first,last,regime in checks:
        part = d[(d.sequence==seq)&(d.gt_id==gid)&d.frame.between(first,last)]
        assert len(part)==last-first+1 and (part.candidate==regime).all() and (part.track_id==-1).all()
    sw = pd.read_csv(out/'switches.csv')
    assert ((sw.sequence=='DJI_0061_2')&(sw.gt_id==9)&(sw.previous_frame==101)&(sw.frame==116)&(sw.previous_track==1)&(sw.track==15)&(sw.missing_annotations==0)).any()
    series = load(out/'diagnosis.json')['sequences']
    with (out/'sequences.csv').open('w',newline='',encoding='utf-8') as target:
        w = csv.writer(target)
        w.writerow(['sequence','frames','GT','TP','FN','FP','IDSW','HOTA','DetA','AssA','IDF1','MOTA','FN_without_aligned_detection','FN_with_aligned_detection'])
        for r in series:
            m,c = r['official']['main'],r['counts']
            missing=sum(v for k,v in r['fn_candidates'].items() if k.startswith('overlap_'))
            w.writerow([r['sequence'],r['frames'],r['gt'],c['TP'],c['FN'],c['FP'],c.get('IDSW',0),
                        *[m[k] for k in ('HOTA','DetA','AssA','IDF1','MOTA')],missing,c['FN']-missing])
    data = dict(low_score_run_frames_at_least=counts,low_score_runs_at_least_5=int((episodes.frames>=5).sum()),
                size_without_largest_fn_sequence=totals)
    (out/'additional_checks.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    print(json.dumps(data),flush=True)


if __name__=='__main__':
    frames = pd.read_csv(OUTPUT/'gt_frames.csv.gz')
    extra(frames,OUTPUT)
    render(frames,OUTPUT)
