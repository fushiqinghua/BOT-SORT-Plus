"""Produce an isolated current-frame supplement from the native B trace."""
import numpy as np
from boxmot.trackers.botsort.basetrack import TrackState

from .geometry import box_iou

MIN_SCORE = 0.05
LOW_THRESHOLD = 0.10
MAX_RECOVERY_AGE = 3
MAX_EMISSIONS_PER_RECOVERY = 2
MIN_PRED_IOU = 0.50
MIN_MARGIN = 0.10
OCCUPIED_IOU = 0.50


def observe(observer, tracker, frame, all_dets, pool, lost, activated, refound):
    """Pass only detached native measurements and geometry to the writer observer."""
    if abs(tracker.track_low_thresh - LOW_THRESHOLD) > 1e-8:
        raise ValueError("The experiment requires a native low threshold of 0.10")
    recoveries = [int(e['track_id']) for e in tracker.last_events if e['event'] == 'recovered']
    newly_lost = {int(t.id): int(t.end_frame) for t in lost
                  if t.state == TrackState.Lost and t.is_activated and t.end_frame == frame - 1}
    eligible = observer.begin(frame, recoveries, newly_lost)
    if not eligible:
        return
    occupied = [tuple(box) for t in activated + refound
                for box in (t.xyxy, all_dets[int(t.det_ind), :4])]
    weak = [tuple(r) for r in all_dets if MIN_SCORE <= float(r[4]) < tracker.track_low_thresh
            and not any(box_iou(r[:4], box) >= OCCUPIED_IOU for box in occupied)]
    predictions = {int(t.id): tuple(tracker.prediction_history[t.id][frame].box) for t in pool}
    observer.score(frame, eligible, weak, predictions)


class WeakOutput:
    """Per-video writer state. Native track objects never enter this class."""

    def __init__(self):
        self.anchors = {}
        self.frame = 0
        self.probes = []
        self.proposed = []

    def begin(self, frame, recoveries, newly_lost):
        if frame != self.frame + 1:
            raise RuntimeError("Expected one native snapshot per video frame")
        self.frame = frame
        self.probes = []
        self.proposed = []
        self.anchors = {tid: anchor for tid, anchor in self.anchors.items()
                        if frame - anchor[0] <= MAX_RECOVERY_AGE}
        for tid in recoveries:
            self.anchors[tid] = (frame, 0)
        return {tid: last for tid, last in newly_lost.items() if tid in self.anchors
                and 1 <= frame - self.anchors[tid][0] <= MAX_RECOVERY_AGE
                and self.anchors[tid][1] < MAX_EMISSIONS_PER_RECOVERY}

    def score(self, frame, eligible, weak, predictions):
        for tid, last_real in eligible.items():
            restore, count = self.anchors[tid]
            scored = sorted(((box_iou(predictions[tid], r[:4]), int(r[6]), r) for r in weak),
                            key=lambda item: (-item[0], item[1]))
            best = scored[0] if scored else None
            row_rival = scored[1][0] if len(scored) > 1 else 0.0
            track_rival = max((box_iou(box, best[2][:4]) for rival, box in predictions.items()
                               if rival != tid), default=0.0) if best is not None else 0.0
            reason = 'accepted'
            if best is None:
                reason = 'no_candidate'
            elif best[0] < MIN_PRED_IOU:
                reason = 'prediction_iou'
            elif best[0] - max(row_rival, track_rival) < MIN_MARGIN:
                reason = 'competition'
            probe = {
                'event': 'post_B_weak_output_check', 'track_id': tid, 'b_recovery_frame': restore,
                'last_native_observation_frame': last_real, 'weak_outputs_before': count,
                'available_candidates': len(weak), 'best_det_index': best[1] if best else None,
                'best_pred_iou': float(best[0]) if best else None, 'row_rival_iou': float(row_rival),
                'track_rival_iou': float(track_rival), 'reason': reason,
            }
            self.probes.append(probe)
            if reason == 'accepted':
                self.proposed.append((tid, last_real, best, probe))
        used = [best[1] for _, _, best, _ in self.proposed]
        if len(used) != len(set(used)):
            raise RuntimeError(f"Frame {frame}: weak output violated mutual margins")

    def emit(self, frame, native_rows):
        if frame != self.frame:
            raise RuntimeError("Weak output and native rows must be from the same frame")
        native_ids = {int(row[4]) for row in native_rows}
        rows = []
        events = self.probes.copy()
        for tid, last_real, (iou, index, det), probe in self.proposed:
            if tid in native_ids or any(box_iou(det[:4], row[:4]) >= OCCUPIED_IOU
                                        for row in native_rows):
                probe['reason'] = 'final_output_occupied'
                continue
            restore, count = self.anchors[tid]
            self.anchors[tid] = (restore, count + 1)
            box = [float(v) for v in det[:4]]
            score = float(det[4])
            rows.append([*box, tid, score, 0, index])
            events.append({'event': 'output_only_low', 'stage': 'post_B_weak_output',
                           'track_id': tid, 'det_index': index, 'score': score, 'xyxy': box,
                           'b_recovery_frame': restore, 'last_native_observation_frame': last_real,
                           'cost': float(1 - iou), 'weak_outputs_in_window': count + 1})
        # Preserve native row dtype and values; use float64 for detached rows.
        return (np.vstack((native_rows, np.asarray(rows, dtype=float))) if rows else native_rows,
                events)
