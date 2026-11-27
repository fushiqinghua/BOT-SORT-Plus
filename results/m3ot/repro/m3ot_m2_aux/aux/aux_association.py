"""Apply calibrated M2 as bounded evidence for native association."""

from collections import Counter
import gzip
import json
from unittest.mock import patch

import numpy as np


SOFT_MAX_PENALTY = .02
SOFT_NLL_SCALE = 4.
SOFT_MAX_MARGIN_SHARE = .5


def soft_cost(native, nll, threshold):
    eligible = np.isfinite(native) & (native <= threshold)
    if not native.size:
        return native
    valid = eligible & np.isfinite(nll)
    row_count = eligible.sum(axis=1, keepdims=True)
    col_count = eligible.sum(axis=0, keepdims=True)
    competition = eligible & ((row_count > 1) | (col_count > 1))
    local_nll = np.where(valid, nll, np.inf)
    row_best = local_nll.min(axis=1, keepdims=True)
    col_best = local_nll.min(axis=0, keepdims=True)
    best = np.minimum(row_best, col_best)
    regret = np.zeros(native.shape)
    np.subtract(nll, best, out=regret, where=np.isfinite(best) & np.isfinite(nll))
    regret[~np.isfinite(nll) & np.isfinite(best)] = SOFT_NLL_SCALE
    penalty = SOFT_MAX_PENALTY * np.clip(regret / SOFT_NLL_SCALE, 0., 1.)
    margin = np.maximum(threshold - native, 0.) * SOFT_MAX_MARGIN_SHARE
    adjusted = native + np.where(competition, np.minimum(penalty, margin), 0.)
    if not np.array_equal(eligible, np.isfinite(adjusted) & (adjusted <= threshold)):
        raise ValueError("M2 auxiliary changed native matching eligibility")
    if np.any((adjusted[eligible] - native[eligible]) > SOFT_MAX_PENALTY + 1e-14):
        raise ValueError("M2 auxiliary exceeded its predeclared cost bound")
    return adjusted


def numerical_checks():
    native = np.array([[.300, .301, np.inf], [.310, .30, .81], [np.inf, .81, .80]])
    nll = np.array([[10., 6., np.inf], [8., np.inf, 1.], [np.inf, 1., 11.]])
    adjusted = soft_cost(native, nll, .80)
    expected = native.copy()
    expected[0, 0] += .020
    expected[1, 1] += .020
    if not np.allclose(adjusted, expected, rtol=0., atol=1e-14):
        raise ValueError("Soft assignment penalty does not match the declared rule")
    edge = np.array([[.796, .2, .8]])
    adjusted = soft_cost(edge, np.array([[11., 2., 13.]]), .8)
    if not np.allclose(adjusted, [[.798, .2, .8]], rtol=0., atol=1e-14):
        raise ValueError("Soft assignment checks failed at the native cost limit")


def compose_aux(base, paths, protocol, out, arm, mode, noise_config, source_config,
                *, with_changes, store):
    from botsort_plus import adapter_v2, native_adapter
    from botsort_research.association import gaussian_cost, assign_with_dummies, stage_config
    from botsort_research.noise import NoiseModel
    from boxmot.trackers.botsort.basetrack import TrackState

    numerical_checks()
    original_assignment = native_adapter.linear_assignment
    noise = NoiseModel.from_dict(noise_config["noise"])
    stages = {name: stage_config(config)
              for name, config in source_config["m2"]["association"].items()}
    sequences = iter(paths)
    high = with_changes(base._associate, [
        ("matches, u_tracks, u_detections = linear_assignment(costs, thresh=threshold)",
         "costs, (matches, u_tracks, u_detections) = "
         "self._aux_match(costs, tracks, detections, stage, threshold)"),
    ])
    low = with_changes(adapter_v2.update_v2, [
        ("matches, unmatched_low_tracks, unmatched_low = linear_assignment(second_cost, thresh=.5)",
         "second_cost, (matches, unmatched_low_tracks, unmatched_low) = "
         "tracker._aux_match(second_cost, low_tracks, low_detections, 'low', .5)"),
    ])

    def pairs(assignment):
        return {tuple(pair) for pair in assignment[0]}

    def safe(number):
        return float(number) if np.isfinite(number) else None

    class AuxiliaryTracker(base):
        _associate = high

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.aux_seq = next(sequences)
            if (self.plus.adapter_mode != "v2" or not self.plus.recovery.enabled
                    or not self.with_reid or self.plus.identity.enabled):
                raise ValueError("The fixed B version and original RGB ReID path are required")
            self.aux_counts = {stage: Counter() for stage in ("high", "low")}
            self.aux_frames = []
            self.aux_recovered = Counter()

        def _aux_match(self, native, tracks, detections, stage, threshold):
            native_assignment = original_assignment(native, thresh=threshold)
            if stage == "unconfirmed":
                return native, native_assignment

            params = stages["first" if stage == "high" else "second"]
            nll, detail = gaussian_cost(
                [t.mean for t in tracks], [t.covariance for t in tracks],
                [d.xywh for d in detections], [d.conf for d in detections], noise, params,
            )
            if stage == "high":
                geom_cost = native_adapter.iou_distance(tracks, detections)
                if self.fuse_first_associate:
                    geom_cost = native_adapter.fuse_score(geom_cost, detections)
            else:
                geom_cost = native
            if geom_cost.shape != native.shape or np.any(native > geom_cost + 1e-10):
                raise ValueError("Raw geometry should upper-bound native RGB-fused cost")

            cost, assignment = native, native_assignment
            if mode == "soft" and stage == "high":
                cost = soft_cost(native, nll, threshold)
                assignment = original_assignment(cost, thresh=threshold)
            elif mode == "gate" and stage == "high":
                cost = np.where(np.isfinite(nll), native, np.inf)
                assignment = original_assignment(cost, thresh=threshold)
            elif (mode, stage) in (("high_nll", "high"), ("low_nll", "low")):
                cost = nll
                assignment = assign_with_dummies(
                    nll, params["miss_cost"], params["unassigned_cost"]
                )
            elif mode == "no_high_reid" and stage == "high":
                cost = geom_cost
                assignment = original_assignment(cost, thresh=threshold)

            previous, selected = pairs(native_assignment), pairs(assignment)
            eligible = np.isfinite(native) & (native <= threshold)
            gate_reason = detail["gate_reason"]
            cutoff = params["miss_cost"] + params["unassigned_cost"]
            if (mode in ("soft", "gate", "no_high_reid") and any(not eligible[i, j] for i, j in selected)):
                raise ValueError("An originally forbidden pair was introduced without NLL replacement")
            native_only = previous - selected
            added = selected - previous
            native_discounted = {(i, j) for i, j in previous if native[i, j] < geom_cost[i, j] - 1e-10}
            counted = self.aux_counts[stage]
            counted.update(
                frames=1, native_pairs=len(previous), applied_pairs=len(selected),
                frames_with_match_change=int(previous != selected),
                native_only=len(native_only), applied_only=len(added),
                native_discounted_by_reid=len(native_discounted),
                native_discounted_above_geom_threshold=sum(
                    1 for i, j in native_discounted if geom_cost[i, j] > threshold
                ),
                native_discounted_removed=len(native_discounted - selected),
                native_matched_rejected_by_m2_gate=sum(
                    1 for i, j in previous if gate_reason[i, j] != 0
                ),
                native_matched_rejected_by_m2_cutoff=sum(
                    1 for i, j in previous if np.isfinite(nll[i, j]) and nll[i, j] >= cutoff
                ),
                applied_pairs_outside_native_eligibility=sum(not eligible[i, j] for i, j in selected),
                applied_low_score_plans=int(stage == "low") * len(selected),
                changed_native_pair_costs=int(np.sum(eligible & (native != cost))),
            )

            if previous != selected:
                # Include all pairs sharing an affected track or detection that
                # either assignment considered, plus eligibility evidence.
                affected = native_only | added
                rows, cols = map(set, zip(*affected))
                relevant = {(i, j) for i, j in (previous | selected)
                            if i in rows or j in cols}
                observed = []
                for i, j in sorted(relevant):
                    track, det = tracks[i], detections[j]
                    observed.append(dict(
                        track_id=int(track.id), det_index=int(det.det_ind), score=float(det.conf),
                        native=(i, j) in previous, applied=(i, j) in selected,
                        was_lost=bool(track.state == TrackState.Lost),
                        native_cost=safe(native[i, j]), geometric_native_cost=safe(geom_cost[i, j]),
                        applied_cost=safe(cost[i, j]), nll=safe(nll[i, j]),
                        covariance_gate_bits=int(gate_reason[i, j]), nll_reject_cutoff=cutoff,
                        native_eligible=bool(eligible[i, j]),
                        physical_iou=float(1 - native_adapter.iou_distance([track], [det])[0, 0]),
                        m2_mahalanobis2=safe(detail["mahalanobis2"][i, j]),
                    ))
                self.aux_frames.append(dict(frame=self.frame_count, stage=stage, pairs=observed))
            return cost, assignment

        def update(self, dets, image, *args, **kwargs):
            with patch.object(adapter_v2, "update_v2", low):
                result = super().update(dets, image, *args, **kwargs)
            self.aux_recovered.update(event["event"] for event in self.last_events
                                      if event["event"] in ("memory_update", "recovered"))
            return result

        def finish_sequence_events(self):
            frames = protocol["sequences"][self.aux_seq]
            if any(counter["frames"] != frames for counter in self.aux_counts.values()):
                raise ValueError("Missing association frame")
            target = out / "aux_decisions" / arm / (self.aux_seq + ".jsonl.gz")
            target.parent.mkdir(parents=True, exist_ok=True)
            lines = "".join(json.dumps(row, allow_nan=False) + "\n" for row in self.aux_frames)
            target.write_bytes(gzip.compress(lines.encode(), mtime=0))
            store(out / "aux_events" / arm / (self.aux_seq + ".json"), {
                "mode": mode, "sequence": self.aux_seq, "frames": frames,
                "stages": self.aux_counts, "b_event_counts": self.aux_recovered,
                "changed_decision_records": len(self.aux_frames),
                "changed_decisions_path": str(target.relative_to(out)),
            })
            return super().finish_sequence_events()

    protocol["m2_aux"] = {
        "mode": mode, "stage_config": stages, "numerical_checks": True,
        "noise": noise.to_dict(), "training": noise_config["provenance"],
        "soft_max_penalty": SOFT_MAX_PENALTY, "soft_scale_nats": SOFT_NLL_SCALE,
        "soft_max_margin_share": SOFT_MAX_MARGIN_SHARE, "gt_open_allowed": False,
    }
    return AuxiliaryTracker
