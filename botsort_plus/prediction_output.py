"""Report a short online projection of a previously measured, stable track."""
from collections import deque
import numpy as np


def overlap(box, others):
    if not len(others):
        return np.empty(0)
    others = np.asarray(others, dtype=float).reshape(-1,4)
    common = np.maximum(0., np.minimum(box[2:], others[:,2:]) - np.maximum(box[:2], others[:,:2]))
    intersect = np.prod(common, axis=1)
    union = np.prod(box[2:]-box[:2]) + np.prod(others[:,2:]-others[:,:2], axis=1) - intersect
    return intersect / np.maximum(union, 1e-8)


def extend(tracker, measured, image_shape, detections, live):
    frame = tracker.frame_count
    memory = getattr(tracker, 'recent_output_measurements', {})
    tracker.recent_output_measurements = memory = {k:v for k,v in memory.items() if k in live}
    occupied = [row[:4] for row in measured]
    for row in measured:
        tid = int(row[4])
        slot = memory.setdefault(tid, deque(maxlen=3))
        prediction = tracker.prediction_history.get(tid, {}).get(frame)
        iou = 0. if prediction is None else float(overlap(prediction.box, [detections[int(row[7]),:4]])[0])
        slot.append((frame,float(row[5]),iou))
    present = {int(row[4]) for row in measured}
    proposals = []
    for t in tracker.lost_stracks:
        tid = int(t.id)
        if tid not in live or tid in present or tid not in memory:
            continue
        hist = memory[tid]
        age = frame - hist[-1][0]
        if (not 1 <= age <= tracker.plus.recovery.output_prediction_frames or len(hist) < 3
                or t.end_frame != hist[-1][0] or hist[-1][0] - hist[0][0] != 2):
            continue
        scores = [x[1] for x in hist]
        if max(scores) < .55 or np.mean(scores) < .35 or min(hist[1][2],hist[2][2]) < .35:
            continue
        pred = tracker.prediction_history.get(tid, {}).get(frame)
        if pred is None:continue
        box = np.asarray(pred.box,dtype=float).copy()
        h,w = image_shape[:2]
        if not np.isfinite(box).all() or min(box[2:]-box[:2]) <= 0:continue
        original_area = np.prod(box[2:]-box[:2])
        box = np.clip(box,[0.,0.,0.,0.],[w,h,w,h])
        if min(box[2:]-box[:2]) <= 0 or np.prod(box[2:]-box[:2]) < .8 * original_area:continue
        proposals.append((np.mean(scores) * .8 ** age, tid, box, age))
    emitted=[]
    for score,tid,box,age in sorted(proposals,key=lambda x:(-x[0],x[1])):
        if np.any(overlap(box, occupied) > .40):continue
        emitted.append([*box,tid,score,0,-1])
        occupied.append(box)
        tracker.last_events.append(dict(event='motion_output',track_id=tid,age=age,det_index=None,
                                        xyxy=box.tolist(),score=float(score)))
    return np.vstack([measured,emitted]) if emitted else measured
