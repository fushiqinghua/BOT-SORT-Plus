"""Join the official GT4 boxes, frozen detections, official assignments and private events after frame 371."""
import argparse
import csv
import gzip
import io
import json
import zipfile
from pathlib import Path


def iou(a, b):
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - w*h
    return w*h/union if union > 0 else 0.


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=root.parent/'gmc_0030_audit/results.zip')
    parser.add_argument('--gt', type=Path)
    parser.add_argument('--detections', type=Path)
    parser.add_argument('--previous-flags', type=Path, default=root.parent/'b_gmc_forensics/results/DJI_0030_2/evidence/evidence_both_affected.csv')
    parser.add_argument('--results', type=Path, default=root/'results')
    args = parser.parse_args()
    if args.gt and args.detections:
        gt_data = args.gt.read_text(encoding='utf-8-sig')
        det_data = args.detections.read_bytes()
    else:
        with zipfile.ZipFile(args.archive) as z:
            gt_data = z.read('source/official_gt_DJI_0030_2.txt').decode('utf-8-sig')
            det_data = z.read('source/frozen_detections_DJI_0030_2.json.gz')
    gt = {}
    for row in csv.reader(io.StringIO(gt_data)):
        frame, gid, x, y, w, h, marked = map(float, row[:7])
        if gid == 4 and marked:
            gt[int(frame)] = [x, y, x+w, y+h]
    frames = json.loads(gzip.decompress(det_data))['frames']
    focus = args.results/'focus2'
    with (focus/'GT4_ID7_frames.csv').open(encoding='utf-8-sig') as f:
        ownership = {(r['method'], int(r['frame'])): r for r in csv.DictReader(f)}
    events = json.loads((focus/'focus_events.json').read_bytes())['pending_guarded']['events']
    with (focus/'focus_diffs/pending_guarded/detection_pairs.csv').open(encoding='utf-8-sig') as f:
        boxes = {(int(r['frame']), int(r['det'])): r for r in csv.DictReader(f)}
    records = []
    for t in range(371, len(frames)+1):
        frame = frames[t-1]
        assert frame['frame'] == t
        old, new = [ownership[m, t] for m in ('evidence_both', 'pending_guarded')]
        obs = [(iou(box, gt[t]), i, box[4]) for i, box in enumerate(frame['detections']) if t in gt and box[4] > .1]
        qualified = [v for v in obs if v[0] >= .5]
        overlap, det, score = max(obs, default=(None, None, None))
        if not overlap:
            det = score = None
        now = [e for e in events if e['frame'] == t and e['tid'] == 7]
        output_det = new['ID7_det']
        output_box = boxes[t, int(output_det)]['after_box'] if output_det else ''
        output_iou = None
        if t in gt and output_box:
            x, y, w, h = json.loads(output_box)
            output_iou = iou((x,y,x+w,y+h), gt[t])
        records.append(dict(frame=t, GT4_box_xyxy=gt.get(t), best_det=det, best_det_score=score,
                            best_det_IoU_GT4=overlap, n_det_IoU_at_least_05=len(qualified),
                            n_high_det_IoU_at_least_05=sum(x[2]>.5 for x in qualified),
                            reference_GT4_tid=old['GT4_tid'], guarded_GT4_tid=new['GT4_tid'],
                            guarded_ID7_status=new['ID7_status'], guarded_ID7_IoU_GT4=output_iou,
                            ID7_events='|'.join(e['event'] for e in now),
                            event_dets='|'.join(str(e.get('det', '')) for e in now)))
    name = args.results/'DJI_0030_2_GT4_检测与原ID逐帧核验.csv'
    with name.open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=records[0]); writer.writeheader(); writer.writerows(records)
    with args.previous_flags.open(encoding='utf-8-sig') as f:
        old_fp = [r for r in csv.DictReader(f) if r['before']=='FP' and r['after']=='absent']
    assert len(old_fp)==67 and len({(r['frame'],r['det']) for r in old_fp})==67
    for stage, method in [('focus','pending_strict'),('focus','pending_temporal'),('focus2','pending_guarded')]:
        with (args.results/stage/'focus_diffs'/method/'detection_pairs.csv').open(encoding='utf-8-sig') as f:
            lookup = {(r['frame'],r['det']):r for r in csv.DictReader(f)}
        for item in old_fp:
            paired = lookup.get((item['frame'],item['det']),{})
            assert not paired.get('before_status')
            item[method] = paired.get('after_status') or 'absent'
            item[method+'_tid'] = paired.get('after_tid')
        counts = json.loads((args.results/stage/'target.json').read_bytes())[method]['old_67']
        assert counts=={s:sum(r[method]==s for r in old_fp) for s in {r[method] for r in old_fp}}
    with (args.results/'DJI_0030_2_历史67个误报逐项复核.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=old_fp[0]); writer.writeheader(); writer.writerows(old_fp)
    print([(r['frame'], round(r['best_det_score'], 4), round(r['best_det_IoU_GT4'], 4))
           for r in records if r['frame']>391 and r['n_det_IoU_at_least_05']])


if __name__ == '__main__':
    main()
