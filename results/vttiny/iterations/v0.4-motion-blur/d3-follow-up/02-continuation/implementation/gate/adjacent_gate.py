"""Short post-B fallback for native low detections close to the IoU boundary."""
from __future__ import annotations

import numpy as np


NAME = "B_adjacent_gate"
MAX_RECOVERY_AGE = 3
MAX_USES_PER_RECOVERY = 1
MIN_IOU = 0.40
NATIVE_IOU = 0.50
MAX_CENTER_DISPLACEMENT = 0.50


def select_matches(tracker, frame, low_tracks, low_detections, cost, u_tracks, u_dets, warp):
    """Return disjoint residual pairs without affecting ordinary assignments."""
    previous_boxes = getattr(tracker, "_gate_previous_boxes", None)
    if previous_boxes is None or len(u_tracks) == 0:
        return []
    overlaps = 1.0 - np.asarray(cost)
    residual_columns = np.asarray(u_dets, dtype=int)
    accepted = []
    for index in u_tracks:
        track = low_tracks[int(index)]
        recovered = getattr(track, "_gate_recovery_frame", None)
        uses = getattr(track, "_gate_uses", 0)
        if (recovered is None or not 1 <= frame - recovered <= MAX_RECOVERY_AGE
                or frame - track.end_frame != 1 or uses >= MAX_USES_PER_RECOVERY):
            continue
        proposal = {"event": "adjacent_gate_probe", "frame": frame, "track_id": int(track.id),
                    "recovery_frame": recovered}
        candidates = residual_columns[overlaps[index, residual_columns] >= MIN_IOU]
        if len(candidates) != 1:
            proposal.update(reason="no_candidate" if not len(candidates) else "ambiguous_track",
                            candidate_count=len(candidates))
            tracker.last_events.append(proposal)
            continue
        column = int(candidates[0])
        detection = low_detections[column]
        overlap = float(overlaps[index, column])
        proposal.update(det_index=int(detection.det_ind), predicted_iou=overlap)
        if overlap >= NATIVE_IOU or np.count_nonzero(overlaps[:, column] >= MIN_IOU) != 1:
            proposal["reason"] = "native_band_or_track_competition"
            tracker.last_events.append(proposal)
            continue
        prior_index = int(track.det_ind)
        if not 0 <= prior_index < len(previous_boxes):
            proposal["reason"] = "missing_previous_measurement"
            tracker.last_events.append(proposal)
            continue
        previous = previous_boxes[prior_index]
        size = previous[2:] - previous[:2]
        if np.any(size <= 0.0):
            raise ValueError("Previous measured detection has invalid size")
        prior_center = warp[:, :2] @ ((previous[:2] + previous[2:]) * 0.5) + warp[:, 2]
        current = detection.xyxy
        displacement = float(np.linalg.norm(((current[:2] + current[2:]) * 0.5 - prior_center) / size))
        proposal["normalized_centroid_motion"] = displacement
        proposal["reason"] = "accepted" if displacement <= MAX_CENTER_DISPLACEMENT else "prior_geometry"
        tracker.last_events.append(proposal)
        if displacement <= MAX_CENTER_DISPLACEMENT:
            accepted.append((int(index), column, proposal))
    return accepted
