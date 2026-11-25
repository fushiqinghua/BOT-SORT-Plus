"""Geometry-only fallback for a B-recovered track in its first three frames."""
from boxmot.trackers.botsort.basetrack import TrackState
from boxmot.trackers.botsort.botsort_track import STrack

from .geometry import box_iou


MIN_SCORE = 0.05
MAX_RECOVERY_AGE = 3
MAX_WEAK_UPDATES = 2
MIN_PRED_IOU = 0.50
MIN_MARGIN = 0.10
OCCUPIED_IOU = 0.50


def post_recovery_weak(tracker, frame, all_dets, pool, lost, activated, refound):
    """Use sub-threshold detections only after all original frame assignments."""
    if abs(tracker.track_low_thresh - 0.10) > 1e-8:
        raise RuntimeError("The weak continuation experiment requires native low threshold 0.10")
    anchors = getattr(tracker, "_short_weak_anchors", {})
    tracker._short_weak_anchors = anchors
    for tid, (seen, _) in list(anchors.items()):
        if frame - seen > MAX_RECOVERY_AGE:
            del anchors[tid]
    for event in tracker.last_events:
        if event["event"] == "recovered":
            anchors[int(event["track_id"])] = (frame, 0)
    if not anchors:
        return lost

    # lost contains only tracks newly marked lost by native low association in
    # this frame. B may already have reclaimed some, which removes those ids.
    eligible = [t for t in lost if t.state == TrackState.Lost and t.is_activated
                and t.end_frame == frame - 1 and t.id in anchors
                and 1 <= frame - anchors[t.id][0] <= MAX_RECOVERY_AGE
                and anchors[t.id][1] < MAX_WEAK_UPDATES]
    if not eligible:
        return lost
    occupied = []
    for native in activated + refound:
        occupied.append(native.xyxy.copy())
        occupied.append(all_dets[int(native.det_ind), :4].copy())
    weak = [row for row in all_dets
            if MIN_SCORE <= float(row[4]) < tracker.track_low_thresh
            and not any(box_iou(row[:4], box) >= OCCUPIED_IOU for box in occupied)]
    boxes = {t.id: tracker.prediction_history[t.id][frame].box for t in pool}
    assignments = []
    for t in eligible:
        rec_frame, count = anchors[t.id]
        scored = sorted(((box_iou(boxes[t.id], row[:4]), int(row[6]), row) for row in weak),
                        key=lambda pair: (-pair[0], pair[1]))
        best = scored[0] if scored else None
        row_rival = scored[1][0] if len(scored) > 1 else 0.
        rival = max((box_iou(box, best[2][:4]) for tid, box in boxes.items() if tid != t.id),
                    default=0.) if best is not None else 0.
        reason = ("no_candidate" if best is None else "prediction_iou" if best[0] < MIN_PRED_IOU
                  else "competition" if best[0] - max(row_rival, rival) < MIN_MARGIN else "accepted")
        tracker.last_events.append({
            "event": "post_B_weak_check", "track_id": int(t.id), "b_recovery_frame": rec_frame,
            "weak_updates_before": count, "available_candidates": len(weak),
            "best_det_index": best[1] if best else None,
            "best_pred_iou": float(best[0]) if best else None,
            "row_rival_iou": float(row_rival), "track_rival_iou": float(rival), "reason": reason,
        })
        if reason == "accepted":
            assignments.append((t, best, rec_frame, count))

    indices = [best[1] for _, best, _, _ in assignments]
    if len(indices) != len(set(indices)):
        raise RuntimeError("Mutual margins did not produce a one-to-one weak assignment")
    reclaimed = set()
    for t, (iou, index, row), rec_frame, count in assignments:
        # Update the original object and id. Passing no feature preserves both
        # native smoothed features and the explicit B appearance memory.
        t.update(STrack(row, max_obs=tracker.max_obs), frame)
        activated.append(t)
        reclaimed.add(t.id)
        anchors[t.id] = (rec_frame, count + 1)
        tracker.last_events.append({
            "event": "match", "stage": "post_B_weak", "track_id": int(t.id),
            "det_index": index, "cost": float(1-iou), "score": float(row[4]),
            "b_recovery_frame": rec_frame, "weak_updates_in_window": count+1,
        })
    return [track for track in lost if track.id not in reclaimed]
