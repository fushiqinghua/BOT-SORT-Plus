"""Use the archived calibrated M2 assignments inside the pinned B v2 replay."""

from collections import Counter
import hashlib
import inspect
import textwrap
from unittest.mock import patch


def with_changes(function, changes):
    """Replace only the designated assignment and correction call sites."""
    source = textwrap.dedent(inspect.getsource(function))
    for original, replacement in changes:
        if source.count(original) != 1:
            raise ValueError(f"Pinned B call did not match exactly once: {original}")
        source = source.replace(original, replacement)
    namespace = {}
    exec(compile(source, inspect.getsourcefile(function), "exec"),
         function.__globals__, namespace)
    return namespace[function.__name__]


def compose_m2(base, output, arm, paths, protocol, frozen, noise_path, archive_config,
               *, live, full, low_r, store):
    import numpy as np
    from botsort_plus import adapter_v2
    from botsort_plus import native_adapter
    from botsort_research.association import assign_with_dummies, gaussian_cost, stage_config
    from botsort_research.noise import NoiseModel, correct_with_covariance

    if full and not live:
        raise ValueError("A shadow run cannot apply calibrated correction")
    noise = NoiseModel.from_dict(noise_path["noise"])
    stages = {stage: stage_config(cfg) for stage, cfg in archive_config["m2"]["association"].items()}
    native_associate = native_adapter.linear_assignment

    def solve(tracks, detections, stage):
        costs, details = gaussian_cost(
            [track.mean for track in tracks], [track.covariance for track in tracks],
            [det.xywh for det in detections], [det.conf for det in detections],
            noise, stages[stage],
        )
        stage_parameters = stages[stage]
        assignment = assign_with_dummies(
            costs, stage_parameters["miss_cost"], stage_parameters["unassigned_cost"]
        )
        return costs, details, assignment

    # Catch a transposed gate or a different assignment/likelihood convention,
    # independently of the case-by-case shadow parity test.
    from types import SimpleNamespace

    mean = np.array([100., 30., 16., 10., 0., 0., 0., 0.])
    covariance = np.diag([1., 2., 3., 4., 1., 1., 1., 1.])
    measurement = np.array([101., 30., 16., 10.])
    track = SimpleNamespace(mean=mean, covariance=covariance)
    detection = SimpleNamespace(xywh=measurement, conf=0.8)
    distant = SimpleNamespace(xywh=measurement + [100., 0., 0., 0.], conf=0.8)
    for stage in ("first", "second"):
        costs, details, (matched, unmatched_tracks, unmatched_dets) = solve(
            [track], [detection, distant], stage
        )
        variance = (np.square(noise.floor_std)
                    + np.square(noise.size_scale * measurement[[2, 3, 2, 3]])
                    + np.square(noise.score_scale) * (1. - detection.conf) ** 2)
        innovation = np.diag(covariance)[:4] + variance
        expected = .5 * (1. / innovation[0] + np.log(innovation).sum() + 4 * np.log(2 * np.pi))
        if (not np.isclose(costs[0, 0], expected, rtol=0., atol=1e-12)
                or not np.allclose(np.diag(details["measurement_covariance"][0, 0]), variance)
                or not np.isposinf(costs[0, 1]) or not (details["gate_reason"][0, 1] & 2)
                or matched.tolist() != [[0, 0]] or len(unmatched_tracks) or unmatched_dets.tolist() != [1]):
            raise ValueError("Synthetic M2 NLL, gating, or dummy assignment verification failed")
    rejected, missed, unused = assign_with_dummies(np.array([[12.]]), 6., 6.)
    if len(rejected) or missed.tolist() != [0] or unused.tolist() != [0]:
        raise ValueError("An exact dummy-cost tie must reject the match")

    original_v2 = adapter_v2.update_v2
    decorated_v2 = with_changes(original_v2, [
        ("matches, unmatched_low_tracks, unmatched_low = linear_assignment(second_cost, thresh=.5)",
         "second_cost, (matches, unmatched_low_tracks, unmatched_low) = "
         "tracker._m2_match(second_cost, low_tracks, low_detections, 'low', .5)"),
        ("            track.update(detection, frame)\n",
         "            tracker._m2_correct(track, detection, track.update, frame)\n"),
    ])
    decorated_high = with_changes(base._associate, [
        ("matches, u_tracks, u_detections = linear_assignment(costs, thresh=threshold)",
         "costs, (matches, u_tracks, u_detections) = "
         "self._m2_match(costs, tracks, detections, stage, threshold)"),
        ("track.update(det,self.frame_count); activated.append(track)",
         "self._m2_correct(track, det, track.update, self.frame_count); activated.append(track)"),
        ("track.re_activate(det,self.frame_count,new_id=False); refound.append(track)",
         "self._m2_correct(track, det, track.re_activate, self.frame_count, "
         "new_id=False); refound.append(track)"),
    ])

    sequence_names = iter(paths)

    class M2Tracker(base):
        _associate = decorated_high

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.m2_seq = next(sequence_names)
            if (self.plus.adapter_mode != "v2" or not self.plus.enabled
                    or not self.plus.recovery.enabled or not self.with_reid
                    or self.plus.identity.enabled or self.plus.raw_ir.enabled):
                raise ValueError("This experiment requires pinned B v2 and cached RGB appearance")
            self.m2_warp_hash = hashlib.sha256()
            self.m2_stages = {stage: Counter() for stage in ("high", "low", "unconfirmed")}
            self.m2_counts = Counter()
            self.m2_selected = {}
            self.m2_native_cmc = self.cmc.apply
            self.cmc.apply = self._m2_warp

        def _m2_warp(self, image, dets):
            warp = self.m2_native_cmc(image, dets)
            self.m2_warp_hash.update(np.asarray(warp, dtype="<f8").tobytes(order="C"))
            self.m2_counts["sof_frames"] += 1
            return warp

        def _m2_match(self, native_cost, tracks, detections, stage, threshold):
            native = native_associate(native_cost, thresh=threshold)
            counts = self.m2_stages[stage]
            counts.update(frames=1, track_candidates=len(tracks), detection_candidates=len(detections))
            counts["native_pairs_at_same_state"] += len(native[0])
            if stage == "unconfirmed":
                if full:
                    for i, j in native[0]:
                        track, det = tracks[i], detections[j]
                        self.m2_selected[id(track), id(det)] = (
                            stage, noise.covariance(det.xywh, det.conf, track.mean[:4])
                        )
                return native_cost, native

            costs, details, m2 = solve(tracks, detections, "first" if stage == "high" else "second")
            reason = details["gate_reason"]
            counts.update(nll_pairs=len(m2[0]), finite_pairs=int(np.isfinite(costs).sum()),
                          gated_mahalanobis=int(((reason & 1) != 0).sum()),
                          gated_center=int(((reason & 2) != 0).sum()),
                          gated_size=int(((reason & 4) != 0).sum()),
                          gated_state=int(((reason & 8) != 0).sum()))
            nll_pairs = {tuple(pair) for pair in m2[0]}
            native_pairs = {tuple(pair) for pair in native[0]}
            counts.update(nll_only_pairs=len(nll_pairs - native_pairs),
                          native_only_pairs=len(native_pairs - nll_pairs),
                          frames_with_assignment_change=int(nll_pairs != native_pairs))
            if full and (stage == "high" or not low_r):
                for i, j in m2[0]:
                    self.m2_selected[id(tracks[i]), id(detections[j])] = (
                        stage, details["measurement_covariance"][i, j]
                    )
            return (costs, m2) if live else (native_cost, native)

        def _m2_correct(self, track, detection, method, *args, **kwargs):
            selected = self.m2_selected.pop((id(track), id(detection)), None)
            if selected is None:
                return method(detection, *args, **kwargs)
            stage, r = selected

            def explicit_r(mean, covariance, measurement, confidence=0.0):
                self.m2_stages[stage]["calibrated_state_updates"] += 1
                return correct_with_covariance(mean, covariance, measurement, r)

            with patch.object(track.kalman_filter, "update", explicit_r):
                return method(detection, *args, **kwargs)

        def update(self, dets, image, *args, **kwargs):
            with patch.object(adapter_v2, "update_v2", decorated_v2):
                predictions = super().update(dets, image, *args, **kwargs)
            if self.m2_selected:
                raise ValueError("A selected calibrated R was not consumed by its detection")
            for event in self.last_events:
                if event["event"] == "match":
                    self.m2_stages[event["stage"]]["applied_matches"] += 1
                elif event["event"] in ("recovered", "memory_update"):
                    self.m2_counts[event["event"]] += 1
            return predictions

        def finish_sequence_events(self):
            raw_sof = self.m2_warp_hash.hexdigest()
            default = frozen["action_counts"]["B_early_moderate"][self.m2_seq]["native_matrix_sha256"]
            frames = protocol["sequences"][self.m2_seq]
            if (raw_sof != default or self.m2_counts["sof_frames"] != frames
                    or any(counts["frames"] != frames for counts in self.m2_stages.values())):
                raise ValueError(f"M2 replay changed frame or native SOF coverage for {self.m2_seq}")
            for stage, counts in self.m2_stages.items():
                applied = "nll_pairs" if live and stage in ("high", "low") else "native_pairs_at_same_state"
                if counts["applied_matches"] != counts[applied]:
                    raise ValueError(f"{stage} match count disagrees with the assignment")
                expected = counts["applied_matches"] if full and (stage != "low" or not low_r) else 0
                if counts["calibrated_state_updates"] != expected:
                    raise ValueError(f"{stage} calibrated R scope or precedence failed")
            store(output / "m2_events" / arm / (self.m2_seq + ".json"), {
                "sequence": self.m2_seq, "frames": frames, "native_sof_matrix_sha256": raw_sof,
                "stages": self.m2_stages, "counts": self.m2_counts,
                "association": "M2 NLL" if live else "native replay, M2 observation only",
                "state_measurement_covariance": {
                    "high": "M2" if full else "native",
                    "unconfirmed": "M2" if full else "native",
                    "low": "learned low R" if low_r else ("M2" if full else "native"),
                    "explicit_B_recovery": "native",
                },
            })
            return super().finish_sequence_events()

    protocol["m2_association"] = {
        "live": live, "calibrated_kf_update": full, "learned_low_r_precedence": full and low_r,
        "stages": stages, "noise": noise.to_dict(), "training": noise_path["provenance"],
        "synthetic_nll_and_rejection_checks": True,
        "geometry": "NLL of camera-compensated XYWH prediction and frozen detection",
        "appearance": "RGB cache and memory unchanged; native unconfirmed and explicit B recovery retain "
                      "appearance. Live NLL replaces the native first/second assignment cost without RGB blending.",
        "comparison": "Native pair counts are counterfactual assignments on each arm's current states, "
                      "not prior-arm full trajectory counts.",
        "gt_open_allowed": False,
    }
    return M2Tracker
