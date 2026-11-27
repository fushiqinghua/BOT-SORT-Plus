"""Compare M3OT low-stage-only M2 with native BoT-SORT and B controls."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch


ARMS = {
    "native_m2_low_shadow": ("native", False),
    "b_m2_low_shadow": ("B_early_moderate", False),
    "native_m2_low_nll": ("native", True),
    "b_m2_low_nll": ("B_early_moderate", True),
}
REPOS = {
    "native": "botsort_plus_v0_4_motion_blur/BOT-SORT-Plus-followup-c3e142e6",
    "B_early_moderate": "b_gmc_joint_r1/repo",
}


def compose(base, control, live, paths, protocol, store, out, arm, noise_model,
            stage_params, with_changes, frozen):
    import numpy as np
    from boxmot.trackers.botsort.botsort import BotSort
    from boxmot.utils.matching import linear_assignment
    from botsort_plus import adapter_v2
    from botsort_research.association import assign_with_dummies, gaussian_cost, stage_config
    from botsort_research.noise import NoiseModel

    sequence_order = iter(paths)
    noise = NoiseModel.from_dict(noise_model["noise"])
    low = stage_config(stage_params)
    if control == "native":
        native_second = with_changes(BotSort._second_association, [
            ("matches, u_track, u_detection = linear_assignment(dists, thresh=0.5)",
             "dists, (matches, u_track, u_detection) = "
             "self._compare_m2_low(dists, r_tracked_stracks, detections_second, 0.5)"),
        ])
    else:
        b_update = with_changes(adapter_v2.update_v2, [
            ("matches, unmatched_low_tracks, unmatched_low = linear_assignment(second_cost, thresh=.5)",
             "second_cost, (matches, unmatched_low_tracks, unmatched_low) = "
             "tracker._compare_m2_low(second_cost, low_tracks, low_detections, .5)"),
        ])

    def pairs(assignment):
        return {tuple(row) for row in assignment[0]}

    def safe(value):
        return float(value) if np.isfinite(value) else None

    class Audit(base):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            if not self.with_reid or self.plus.enabled != (control != "native"):
                raise ValueError("Original native or B control with active RGB ReID is required")
            if control != "native" and (self.plus.adapter_mode != "v2" or not self.plus.recovery.enabled):
                raise ValueError("Unexpected B tracker configuration")
            self.low_seq = next(sequence_order)
            self.trace_counts = Counter()
            self.sof = hashlib.sha256()
            self.raw_cmc = self.cmc.apply
            self.cmc.apply = self._capture_cmc
            self.changed_frames = []

        def _capture_cmc(self, image, detections):
            warp = self.raw_cmc(image, detections)
            self.sof.update(np.asarray(warp, dtype="<f8").tobytes(order="C"))
            self.trace_counts["sof_frames"] += 1
            return warp

        def _compare_m2_low(self, native_cost, tracks, detections, threshold):
            if threshold != .5:
                raise ValueError("Native low-stage threshold must be 0.5")
            nll, detail = gaussian_cost(
                [t.mean for t in tracks], [t.covariance for t in tracks],
                [d.xywh for d in detections], [d.conf for d in detections], noise, low,
            )
            native = linear_assignment(native_cost, thresh=threshold)
            m2 = assign_with_dummies(nll, low["miss_cost"], low["unassigned_cost"])
            native_pairs = pairs(native)
            m2_pairs = pairs(m2)
            gate = detail["gate_reason"]
            limit = low["miss_cost"] + low["unassigned_cost"]
            self.trace_counts.update(
                low_stage_frames=1,
                native_pairs_at_same_state=len(native_pairs),
                m2_candidate_pairs=len(m2_pairs),
                actual_low_matches=len(m2_pairs if live else native_pairs),
                differing_frames=int(native_pairs != m2_pairs),
                native_only_pairs=len(native_pairs - m2_pairs),
                nll_only_pairs=len(m2_pairs - native_pairs),
                nll_pairs_outside_native_iou=sum(1 for i, j in m2_pairs if native_cost[i, j] > threshold),
                native_pairs_rejected_by_hard_gate=sum(1 for i, j in native_pairs if gate[i, j] != 0),
                native_pairs_rejected_by_finite_nll_cutoff=sum(
                    1 for i, j in native_pairs if np.isfinite(nll[i, j]) and nll[i, j] >= limit
                ),
            )
            if native_pairs != m2_pairs:
                entries = []
                for i, j in sorted(native_pairs ^ m2_pairs):
                    entries.append(dict(
                        track_id=int(tracks[i].id), det_index=int(detections[j].det_ind),
                        score=float(detections[j].conf), in_native=(i, j) in native_pairs,
                        in_m2=(i, j) in m2_pairs, iou=float(1 - native_cost[i, j]),
                        native_eligible=bool(native_cost[i, j] <= threshold),
                        nll=safe(nll[i, j]), mahalanobis2=safe(detail["mahalanobis2"][i, j]),
                        gate_bits=int(gate[i, j]), nll_reject_cutoff=limit,
                    ))
                self.changed_frames.append(dict(frame=int(self.frame_count), pair_differences=entries))
            return (nll, m2) if live else (native_cost, native)

        def finish_sequence_events(self):
            expected_frames = protocol["sequences"][self.low_seq]
            expected_sof = frozen["action_counts"][control][self.low_seq]["native_matrix_sha256"]
            if (self.trace_counts["sof_frames"] != expected_frames
                    or self.trace_counts["low_stage_frames"] != expected_frames
                    or self.sof.hexdigest() != expected_sof):
                raise ValueError("Low association scope or fixed SOF stream changed")
            if control != "native" and self.trace_counts["b_low_events"] != self.trace_counts["actual_low_matches"]:
                raise ValueError("B low-stage audit disagrees with the chosen matches")
            decisions = out / "decisions" / arm / (self.low_seq + ".jsonl.gz")
            decisions.parent.mkdir(parents=True, exist_ok=True)
            raw = "".join(json.dumps(row, allow_nan=False) + "\n" for row in self.changed_frames)
            decisions.write_bytes(gzip.compress(raw.encode(), mtime=0))
            store(out / "audit" / arm / (self.low_seq + ".json"), dict(
                sequence=self.low_seq, control=control, frames=expected_frames,
                native_sof_sha256=self.sof.hexdigest(), actual_m2_low=live,
                counts=self.trace_counts, changed_decisions=len(self.changed_frames),
            ))
            return super().finish_sequence_events()

    if control == "native":
        class NativeLow(Audit):
            _second_association = native_second

        tracker = NativeLow
    else:
        class BLow(Audit):
            def update(self, dets, img, *args, **kwargs):
                with patch.object(adapter_v2, "update_v2", b_update):
                    data = super().update(dets, img, *args, **kwargs)
                for event in self.last_events:
                    if event["event"] == "match":
                        self.trace_counts["b_" + event["stage"] + "_events"] += 1
                    elif event["event"] == "recovered":
                        self.trace_counts["b_recovered"] += 1
                return data

        tracker = BLow

    protocol["m2_low_nll"] = dict(
        control=control, actual_m2_low=live, stage_config=low, noise=noise.to_dict(),
        training=noise_model["provenance"], allowed_stage="native second association only",
        high_and_unconfirmed_assignment="original IoU and frozen RGB ReID",
        kf_Q="native", kf_measurement_update="native", raw_sof="frozen",
        gt_open_allowed=False, uses_b=control != "native",
    )
    return tracker


def run(project, out, arm):
    control, live = ARMS[arm]
    sys.path.insert(0, str(out.parent / "m2"))
    from m2_association import with_changes
    from replay import FROZEN, SOURCE_PINS, VAL, load_references, read, sha, store

    first, archived, controls, frozen, _, noise, previous = load_references(project)
    control_path = controls / "protocols" / (control + ".json")
    expected = read(control_path)
    repo = project.parent / "VT_Tiny_MOT_Baseline_v1/integrations" / REPOS[control]
    if expected["sequences"] != VAL or expected["status"] != "COMPLETE":
        raise ValueError("Missing complete paired control")
    sys.path[:0] = [str(p) for p in (repo, repo / "tools", project / "tracking_runtime",
                                      archived / "research_archive")]
    import run_botsort_plus as runner
    from botsort_plus import native_adapter

    base_prepare = runner.prepare
    config = dict(expected["config"], name=arm)
    cfg_path = out / "configs" / (arm + ".json")
    store(cfg_path, config)
    stage = read(archived / "research_archive/configs/research/round2/default.json")["m2"]["association"]["second"]
    native_adapter_class = native_adapter.BoTSORTPlus

    def configure(options):
        cfg, paths, protocol = base_prepare(options)
        if protocol["config"] != config or any(protocol[k] != expected[k] for k in FROZEN if k in protocol):
            raise ValueError("Configuration, implementation or frozen assets differ from the source control")
        native_adapter.BoTSORTPlus = compose(native_adapter_class, control, live, paths, protocol, store,
                                              out, arm, noise, stage, with_changes, frozen)
        protocol["m2_low_nll"].update(
            archive_commit=previous["research_commit"], archive_source_pins=SOURCE_PINS,
            control_protocol_sha256=sha(control_path), wrapper_sha256=sha(Path(__file__)),
        )
        protocol["evaluation_context"] = "M3OT strong detector and frozen RGB ReID, both RGB validation clips"
        return cfg, paths, protocol

    runner.prepare = configure
    assets = first / "inputs/higher"
    params = [
        "--config", str(cfg_path), "--baseline-root", str(assets),
        "--features-dir", str(assets / "features"), "--images-root", str(project / "data/raw/M3OT"),
        "--runtime-root", str(project), "--output-dir", str(out), "--trace",
    ]
    runner.run(runner.parser().parse_args(params))
    protocol_path = out / "protocols" / (arm + ".json")
    protocol = read(protocol_path)
    if protocol["status"] != "COMPLETE" or any(protocol[k] != expected[k] for k in FROZEN):
        raise ValueError("Replay incomplete or changed frozen components")
    sequences = {}
    for seq, frames in VAL.items():
        timing = read(out / "timing" / arm / (seq + ".json"))
        track = out / "trackers" / arm / "data" / (seq + ".txt")
        events = out / "events" / arm / (seq + ".jsonl")
        audit = out / "audit" / arm / (seq + ".json")
        decisions = out / "decisions" / arm / (seq + ".jsonl.gz")
        mot_hash, event_hash = sha(track), sha(events)
        if timing["frames"] != frames or timing["prediction_sha256"] != mot_hash:
            raise ValueError("Incomplete new val output")
        parity = None
        if not live:
            original_trace = controls / "events" / control / events.name
            parity = dict(tracks=mot_hash == expected["output_prediction_hashes"][seq],
                          events=event_hash == sha(original_trace))
            if not all(parity.values()):
                raise ValueError(f"Shadow differs from the paired control for {seq}")
        sequences[seq] = dict(frames=frames, rows=timing["output_rows"], sha256=mot_hash,
                              events_sha256=event_hash, audit_sha256=sha(audit),
                              decisions_sha256=sha(decisions), byte_parity=parity)
    store(out / "receipts" / (arm + ".json"), dict(
        arm=arm, control=control, live=live, status="COMPLETE", sequences=sequences,
        protocol_sha256=sha(protocol_path), detector_sha256=protocol["detector_sha256"],
        rgb_encoder_sha256=protocol["rgb_encoder_sha256"],
        calibration_sha256=SOURCE_PINS["research_assets/m2_noise.json"],
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("arm", choices=ARMS)
    options = parser.parse_args()
    run(options.project, options.out, options.arm)
