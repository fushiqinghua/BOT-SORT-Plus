"""One causal tracker process. This module never imports the GT analyzer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .common import (REPO, METHODS, SCHEMA, BASE_COMMIT, frozen_config, read_json,
                     source_hashes, sha256, write_json)


def run(settings, method, output, sequences=None, *, probe_mode="off", preflight=False):
    if method not in METHODS or probe_mode not in ("off", "signals", "full"):
        raise ValueError("Unsupported method or probe mode")
    if "gt_root" in settings or "references" in settings:
        raise ValueError("Use prediction_settings(): prediction must not receive evaluation settings")
    output = Path(output)
    if settings.get("runtime_root"):
        sys.path.insert(0, str(Path(settings["runtime_root"]) / "tracking_runtime"))
    sys.path.insert(0, str(REPO / "tools"))
    import numpy as np
    import cv2
    import torch
    import run_botsort_plus as original
    import botsort_plus.native_adapter as adapter
    from .camera import WarpCamera, setup_matcher
    from experiments.vttiny_gmc.native_botsort_gmc_0030_fp_v1.guard import capture_priors, iou
    from experiments.vttiny_gmc.b_gmc_recovery.forensics.lost_camera_margin import LostCameraMargin

    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    policy = dict(METHODS[method])
    cfg_path = output / "configs" / (method + ".json")
    if cfg_path.exists():
        raise FileExistsError(f"Configuration already exists: {cfg_path}")
    write_json(cfg_path, frozen_config(method))
    argv = ["--config", str(cfg_path), "--baseline-root", settings["baseline_root"],
            "--features-dir", settings["features_dir"], "--images-root", settings["images_root"],
            "--output-dir", str(output), "--trace"]
    if settings.get("runtime_root"):
        argv += ["--runtime-root", settings["runtime_root"]]
    if sequences:
        argv += ["--sequences", *sequences]
    if preflight:
        argv.append("--preflight")
    args = original.parser().parse_args(argv)
    original_prepare, original_tracker = original.prepare, adapter.BoTSORTPlus
    trackers = []

    def prepare(parsed):
        cfg, paths, protocol = original_prepare(parsed)
        if not (cfg.adapter_mode == "v2" and cfg.recovery.enabled and cfg.recovery.early_evidence == "moderate"):
            raise ValueError("Frozen B foundation required")
        if protocol["native_config"].get("cmc_method") != "sof":
            raise ValueError("Native SOF reference is required")
        # Verify real source on preflight too, before any experiment can run.
        adapter.verify_upstream()
        from botsort_plus.adapter_v2 import verify_v2_upstream
        verify_v2_upstream()
        if protocol["status"] != "READY":
            if preflight:
                write_json(output / "preflight.json", protocol)
            return cfg, paths, protocol
        matcher, model = setup_matcher(settings, policy["camera"])
        sequence_iter = iter(paths)
        protocol["joint_batch"] = {
            "schema": SCHEMA, "base_commit": BASE_COMMIT, "arm": method, **policy,
            "gmc_mode": settings.get("gmc_mode", "live"), "probe_mode": probe_mode,
            "kalman": "original_BoxMOT_16.0.11", "center_output_fix": False,
            "pending_identity": False, "future_frames": False, "ground_truth_read": False,
            "source_sha256": source_hashes(), "model_reference": model,
            "camera_manifests": {}, "probe_manifests": {}, "decision_manifests": {},
            "timing_note": "Cached features; cached warp omits GMC estimation; probes add diagnostic overhead. Not end-to-end FPS.",
        }
        if matcher is not None:
            from importlib.metadata import version, PackageNotFoundError
            actual = {"torch": torch.__version__}
            for package in ("kornia", "kornia_rs"):
                try:
                    actual[package] = version(package)
                except PackageNotFoundError:
                    actual[package] = "unavailable_metadata"
            protocol["joint_batch"]["actual_model_runtime"] = actual
        if preflight:
            write_json(output / "preflight.json", protocol)

        class ExperimentTracker(original_tracker):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.seq = next(sequence_iter)
                self.joint_dual_low = policy["dual_low"]
                self.joint_evidence_both = policy["evidence_both"]
                self.joint_evidence_gate = policy["evidence_both"]
                self.capture_trace = False
                self.matched_camera = False
                self.blocked_native_edges = 0
                self.lightglue_frames = 0
                self.camera = WarpCamera(settings, policy["camera"], self.seq, protocol,
                                         self.cmc.apply, matcher, output / "warps" / method / (self.seq + ".jsonl.gz"))
                self.cmc.apply = self.warp
                self.margin = LostCameraMargin(self, require_observation=True) if self.joint_evidence_both else None
                if self.margin is not None:
                    self.recovery_matcher._pair = self.margin.recovery_pair(self.recovery_matcher._pair)
                path = output / "decisions" / method / (self.seq + ".jsonl")
                path.parent.mkdir(parents=True, exist_ok=True)
                self.audit_path = path
                self.audit = path.open("x", encoding="utf-8")
                self.audit_closed = False
                if probe_mode != "off":
                    from .probe import JointProbe
                    self.joint_probe = JointProbe(output / "probes" / method / (self.seq + ".jsonl.gz"), mode=probe_mode)
                trackers.append(self)

            def record(self, event, **fields):
                self.audit.write(json.dumps({"sequence": self.seq, "frame": self.frame_count,
                                             "event": event, **fields}, allow_nan=False) + "\n")

            def _cost(self, tracks, detections, unconfirmed=False):
                result = super()._cost(tracks, detections, unconfirmed=unconfirmed)
                if self.margin is not None and not unconfirmed:
                    self.margin.high_cost(result, tracks)
                return result

            def _associate(self, tracks, detections, threshold, stage, activated, refound, unconfirmed=False):
                before = {int(t.id): {"state_before": int(t.state), "last_seen": int(t.end_frame),
                                      "gap": self.frame_count - int(t.end_frame)} for t in tracks}
                start = len(self.last_events)
                result = super()._associate(tracks, detections, threshold, stage, activated, refound, unconfirmed)
                for event in self.last_events[start:]:
                    if event.get("event") == "match":
                        tid = int(event["track_id"])
                        self.record("native_accepted", stage=stage, track_id=tid,
                                    det_index=int(event["det_index"]), cost=event["cost"], **before[tid])
                return result

            def guard_low_cost(self, cost, tracks, detections):
                if not self.joint_dual_low or not self.matched_camera:
                    return
                for i, track in enumerate(tracks):
                    backup = self.backup_boxes[int(track.id)]
                    for j, detection in enumerate(detections):
                        backup_iou = iou(backup, detection[:4])
                        if cost[i, j] <= .5 and backup_iou < .5:
                            self.record("dual_low_rejected_edge", track_id=int(track.id),
                                        det_index=int(detection[6]), selected_iou=float(1. - cost[i, j]),
                                        backup_iou=float(backup_iou))
                            cost[i, j] = 1.
                            self.blocked_native_edges += 1

            def warp(self, image, detections):
                native, used, backup, record = self.camera.apply(image, detections, self.frame_count)
                self.joint_native_warp = native.copy()
                self.joint_selected_warp = used.copy()
                self.joint_backup_warp = backup.copy()
                self.matched_camera = record["choice"] == "lightglue" and not np.array_equal(used, backup)
                if self.margin is not None:
                    self.margin.observe(self.frame_count, native, used)
                if self.matched_camera:
                    self.lightglue_frames += 1
                    if self.joint_dual_low:
                        capture_priors(self, used, backup)
                return used

            def update(self, detections, image, *a, **kw):
                result = super().update(detections, image, *a, **kw)
                # Exact row identity for offline joins; no inference from score.
                self.record("output_index", rows=[{"track_id": int(r[4]), "det_index": int(r[7])}
                                                   for r in result])
                return result

            def finish_sequence_events(self):
                camera = self.camera.close()
                protocol["joint_batch"]["camera_manifests"][self.seq] = camera
                if hasattr(self, "joint_probe"):
                    protocol["joint_batch"]["probe_manifests"][self.seq] = self.joint_probe.close()
                actions = {"lightglue_frames": self.lightglue_frames,
                           "blocked_native_edges": self.blocked_native_edges,
                           "center_outputs": 0, "dual_low": self.joint_dual_low}
                if self.margin is not None:
                    actions.update(high_rejections=self.margin.high_rejections,
                                   recovery_rejections=self.margin.recovery_rejections,
                                   unsupported_rejections=self.margin.unsupported_rejections)
                write_json(output / "actions" / method / (self.seq + ".json"), actions)
                self.audit.close()
                self.audit_closed = True
                protocol["joint_batch"]["decision_manifests"][self.seq] = {
                    "data_sha256": sha256(self.audit_path), "frames": self.frame_count,
                }
                return super().finish_sequence_events()

        adapter.BoTSORTPlus = ExperimentTracker
        return cfg, paths, protocol

    original.prepare = prepare
    try:
        status = original.run(args)
        if not preflight and status == 0:
            path = output / "protocols" / (method + ".json")
            result = read_json(path)
            for seq, cam in result["joint_batch"]["camera_manifests"].items():
                if cam["source"] and cam["source"]["decoded_image_sha256"] != result["decoded_image_hashes"][seq]:
                    result["status"] = "FAILED"
                    result["error"] = "Frozen warp source uses different decoded image pixels: " + seq
                    write_json(path, result)
                    raise ValueError(result["error"])
        return status
    finally:
        original.prepare, adapter.BoTSORTPlus = original_prepare, original_tracker
        for tracker in trackers:
            tracker.camera.abort()
            if not tracker.audit_closed:
                tracker.audit.close()
            probe = getattr(tracker, "joint_probe", None)
            if probe is not None and not probe.stream.closed:
                probe.stream.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequences", nargs="+")
    parser.add_argument("--probe-mode", choices=("off", "signals", "full"), default="off")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    return run(read_json(args.settings), args.method, args.output_dir, args.sequences,
               probe_mode=args.probe_mode, preflight=args.preflight)


if __name__ == "__main__":
    raise SystemExit(main())
