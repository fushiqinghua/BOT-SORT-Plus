"""Causal T3a replay at the pinned native low-stage update boundary."""
from __future__ import annotations

import argparse
from collections import Counter, deque
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

REPO_COMMIT = "53faeccba267d8988174c18e1a538f759fa6dd07"
ADAPTER_BLOB = "6818f98fb9b02b32c33e89e603e68a0f06928f79"
METHODS = ("B_early_moderate", "evidence_both")
MODES = ("off", "capture", "record", "fallback", "active")
ADOPTION_KEYS = {"support", "refined_quality", "quality_gain"}
SELECTION_KEYS = {"arm", "checkpoint", "checkpoint_sha256", "thresholds"}
CALIBRATION_METADATA = {"seed", "epoch", "assigned", "status", "protocol_status",
                        "crop_manifest_sha256", "policy_sha256", "calibrated_split", "gate_split"}


def pinned_source(module, expected):
    path = Path(module.__file__)
    data = path.read_bytes().replace(b"\r\n", b"\n")
    blob = b"blob " + str(len(data)).encode() + b"\x00" + data
    if hashlib.sha1(blob).hexdigest() != expected:
        raise ValueError(f"Installed tracker source differs from the pinned blob: {path}")
    return data.decode("utf-8")


def instrument_adapter(text, replace_once):
    """After the single native solve, replace only the selected measurement."""
    boundary = "    matches, unmatched_low_tracks, unmatched_low = linear_assignment(second_cost, thresh=.5)\n"
    text = replace_once(text, boundary, boundary +
        "    replacements = tracker.tco.measurements(tracker, low_tracks, low, matches, second_cost, STrack)\n")
    boundary = "            track.update(detection, frame)\n            tracker._write_memory(track, int(detection.det_ind))\n"
    return replace_once(text, boundary,
        "            measure = replacements.get((int(i), int(j)), detection)\n"
        "            track.update(measure, frame)\n"
        "            tracker.tco.after_update(track, int(detection.det_ind))\n"
        "            tracker._write_memory(track, int(detection.det_ind))\n")


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def selection(path):
    """Load only the immutable model pointer and three calibration scalars."""
    path = Path(path).resolve()
    record = json.loads(path.read_bytes())
    if not SELECTION_KEYS <= record.keys() or record.keys() - SELECTION_KEYS - CALIBRATION_METADATA:
        raise ValueError("Unexpected model selection fields")
    record = {name: record[name] for name in SELECTION_KEYS}
    if record["arm"] not in ("P2", "P3"):
        raise ValueError("Only a selected RGB head belongs in this replay")
    checkpoint = (path.parent / record["checkpoint"]).resolve()
    if sha256(checkpoint) != record["checkpoint_sha256"]:
        raise ValueError("The selected checkpoint differs from calibration")
    thresholds = record["thresholds"]
    if set(thresholds) != ADOPTION_KEYS or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1
        for v in thresholds.values()
    ):
        raise ValueError("Expected three finite probability or gain thresholds in [0, 1]")
    return record, checkpoint


def valid_refinement(raw, refined):
    raw, refined = np.asarray(raw, dtype=float), np.asarray(refined, dtype=float)
    if (raw.shape != (4,) or refined.shape != (4,) or not np.isfinite(raw).all()
            or not np.isfinite(refined).all()):
        return False
    orig_size = raw[2:] - raw[:2]
    new_size = refined[2:] - refined[:2]
    displacement = abs((refined[2:] + refined[:2] - raw[2:] - raw[:2]) * .5)
    tol = 1e-4
    return bool((orig_size > 0).all() and
                (new_size >= .8 * orig_size - tol).all() and
                (new_size <= 1.25 * orig_size + tol).all() and
                (displacement <= np.minimum(.25 * orig_size, 3) + tol).all())


class Refiner:
    def __init__(self, path, device):
        import torch
        import model

        self.torch, self.api, self.device = torch, model, torch.device(device)
        self.settings, filename = selection(path)
        self.provenance = {"selection_sha256": sha256(path), "checkpoint_sha256": sha256(filename),
                           "model_source_sha256": sha256(model.__file__), "device": str(self.device),
                           "arm": self.settings["arm"], "thresholds": self.settings["thresholds"]}
        self.model = model.TCO(arm=self.settings["arm"]).to(self.device)
        checkpoint = torch.load(filename, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
        self.model.eval()

    def sample(self, image, box, pred, score, frame, past, load):
        return self.api.make_sample(image, box, pred, score, frame, past, load, arm=self.settings["arm"])

    def infer(self, samples):
        with self.torch.inference_mode():
            outputs = self.model(self.api.stack_samples(samples, self.device))
            accepted = self.api.apply_thresholds(outputs, self.settings["thresholds"]).cpu().tolist()
            boxes, support, quality_raw, quality_refined = (
                outputs[name].detach().cpu().tolist()
                for name in ("refined_xyxy", "support", "quality_raw", "quality_refined"))
        return zip(accepted, boxes, support, quality_raw, quality_refined, strict=True)


def snapshot(track):
    return {"mean": track.mean.tolist(), "covariance": track.covariance.tolist(),
            "xyxy": track.xyxy.tolist()}


def make_observer(collector, mode, refiner):
    """Reuse the exact T0 state-signature writer and frozen cache validation."""
    class Observer(collector.Observer):
        def __init__(self, path, feature, source_video, sequence, method, output, protocol, enabled):
            super().__init__(path, feature, source_video, sequence, method, output, protocol, False)
            self.variant = mode
            if mode == "off":
                return
            from botsort_plus.io import detection_fingerprint, read_detection_cache

            cache = read_detection_cache(path)
            if detection_fingerprint(cache) != self.source["detection_fingerprint"]:
                raise ValueError("TCO received a different detection cache after preflight")
            self.rows = cache["frames"]
            self.images = {}
            self.runtime = dict(crop_seconds=0., infer_seconds=0.)
            destination = Path(output) / "tco_measurements" / method
            destination.mkdir(parents=True, exist_ok=True)
            self.audit_path = destination / f"{sequence}.jsonl"
            self.audit = self.audit_path.open("x", encoding="utf-8", newline="\n")
            self.run_metadata = protocol.setdefault("tco_fixed_pair", {
                "schema": "tco_t3a_v1", "mode": mode, **refiner.provenance,
                "reference": method, "adapter_blob": ADAPTER_BLOB,
                "online_source_sha256": sha256(__file__), "collector_sha256": sha256(collector.__file__),
                "stage": "native_low_assigned_only", "preserves_score_class_det_index_rgb_reid": True,
                "history": {"max_observations": collector.MAX_HISTORY, "max_age": collector.MAX_AGE,
                            "detection_rows": "actual accepted raw rows"},
                "ground_truth_read": False, "sequences": {},
            })

        def begin(self, tracker, detections, image, *, features):
            super().begin(tracker, detections, image, features=features)
            if self.variant == "off":
                return
            entry = self.rows[self.frame - 1]
            self.raw = np.asarray(entry["detections"], np.float32).reshape(-1, 6)
            if (entry["frame"] != self.frame or not np.array_equal(self.raw[:, :5], detections[:, :5])
                    or features is None or len(features) != len(self.raw)):
                raise ValueError("TCO detection and ReID rows must align with the frozen native input")
            self.images[self.frame] = image.copy()
            self.images.pop(self.frame - collector.MAX_AGE - 1, None)
            self.pending = {}

        def measurements(self, tracker, low_tracks, low, matches, cost, stype):
            if self.variant == "off":
                return {}
            if tracker.plus.identity.enabled:
                raise ValueError("The fixed-pair replay requires the frozen references with C2 off")
            replacements, samples, rows = {}, [], []
            claimed_tracks, claimed_detections = set(), set()
            start = time.perf_counter()
            for i, j in matches:
                i, j = int(i), int(j)
                track, native_row = low_tracks[i], low[j]
                tid, did = int(track.id), int(native_row[6])
                if not 0 <= did < len(self.raw):
                    raise ValueError("The native low assignment lost the detector row index")
                raw = self.raw[did]
                if (tid in claimed_tracks or did in claimed_detections
                        or not tracker.track_low_thresh < float(raw[4]) < tracker.track_high_thresh
                        or not np.array_equal(raw[:5], native_row[:5]) or cost[i, j] > .5):
                    raise ValueError("Unexpected identity, detector row, or guard result after the native solve")
                claimed_tracks.add(tid)
                claimed_detections.add(did)
                history = [entry for entry in self.history.get(tid, ())
                           if self.frame - collector.MAX_AGE <= entry["frame"] < self.frame]
                row = dict(frame=self.frame, sequence=self.sequence, track_id=tid, det_index=did,
                           stage="low", guard_cost=float(cost[i, j]), raw_score=float(raw[4]),
                           raw_class=int(raw[5]), native_class=int(native_row[5]), raw_xyxy=raw[:4].tolist(),
                           used_xyxy=raw[:4].tolist(), support=None, quality_raw=None, quality_refined=None,
                           candidate_xyxy=None, history=history, kf_before=snapshot(track),
                           would_apply=False, accepted_by_model=False, applied=False)
                self.pending[(tid, did)] = row
                self.counts["low_accepted"] += 1
                if not track.is_activated:
                    row["action"] = "unconfirmed"
                    continue
                sample, reason = refiner.sample(
                    self.images[self.frame], raw[:4], track.xyxy, float(raw[4]), self.frame, history,
                    lambda item: self.images.get(item["frame"]))
                if sample is None:
                    row["action"] = reason or "sample_unavailable"
                    continue
                samples.append(sample)
                rows.append((row, native_row, i, j))
            self.runtime["crop_seconds"] += time.perf_counter() - start
            if not samples:
                return replacements
            start = time.perf_counter()
            outcomes = list(refiner.infer(samples))
            self.runtime["infer_seconds"] += time.perf_counter() - start
            self.counts["evaluated"] += len(outcomes)
            for (row, native_row, i, j), (keep, box, support, quality_raw, quality_refined) in zip(
                    rows, outcomes, strict=True):
                if not all(math.isfinite(v) for v in (support, quality_raw, quality_refined, *box)):
                    row["action"] = "invalid_prediction"
                    continue
                row.update(support=support, quality_raw=quality_raw, quality_refined=quality_refined,
                           candidate_xyxy=box, accepted_by_model=bool(keep))
                if not keep:
                    row["action"] = "threshold_or_roi"
                    continue
                if not valid_refinement(row["raw_xyxy"], box):
                    row["action"] = "out_of_bounds"
                    continue
                modified = native_row.copy()
                modified[:4] = box
                if np.array_equal(modified[:4], native_row[:4]):
                    row["action"] = "identical"
                    continue
                row["would_apply"] = True
                if self.variant != "active":
                    row["action"] = "record_only" if self.variant == "record" else "forced_fallback"
                    continue
                replacements[(i, j)] = stype(modified, max_obs=tracker.max_obs)
                row.update(applied=True, action="applied", used_xyxy=modified[:4].tolist())
            return replacements

        def after_update(self, track, det_index):
            if self.variant != "off":
                self.pending[(int(track.id), det_index)]["kf_after"] = snapshot(track)

        def end(self, tracker, output):
            super().end(tracker, output)
            if self.variant == "off":
                return
            low_pairs, accepted_dets, accepted_ids = set(), set(), set()
            for tid, did, stage in collector.accepted_pairs(tracker.last_events):
                if tid in accepted_ids or did in accepted_dets or not 0 <= did < len(self.raw):
                    raise ValueError("More than one track consumed the same detector row")
                accepted_ids.add(tid)
                accepted_dets.add(did)
                if stage == "low":
                    low_pairs.add((tid, did))
                obs = dict(frame=self.frame, det_index=did, raw_xyxy=self.raw[did, :4].tolist(),
                           raw_score=float(self.raw[did, 4]), source_stage=stage)
                self.history.setdefault(tid, deque(maxlen=collector.MAX_HISTORY)).append(obs)
            if low_pairs != self.pending.keys():
                raise ValueError("Audited measurements differ from the native low-stage event assignments")
            emitted = {(int(row[4]), int(row[7])): row[:4].tolist() for row in output}
            for key, record in self.pending.items():
                if "kf_after" not in record:
                    raise ValueError("Expected position state after the native measurement update")
                record["output_xyxy"] = emitted.get(key)
                self.audit.write(collector.encode(record).decode() + "\n")
                self.counts[record["action"]] += 1
            live = set(tracker.memories)
            self.history = {tid: entries for tid, entries in self.history.items()
                            if tid in live and entries[-1]["frame"] >= self.frame - collector.MAX_AGE}

        def close(self):
            super().close()
            if self.variant == "off":
                return
            self.audit.close()
            self.images.clear()
            self.history.clear()
            self.run_metadata["sequences"][self.sequence] = {
                "counts": dict(self.counts), "timing_seconds": self.runtime,
                "measurement_file_sha256": sha256(self.audit_path),
            }
    return Observer


def run(args):
    repo = args.repo.resolve()
    actual_commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if actual_commit != REPO_COMMIT:
        raise ValueError(f"Expected pinned checkout {REPO_COMMIT}, found {actual_commit}")
    for folder in (args.collector.parent, Path(__file__).resolve().parents[1], repo):
        sys.path.insert(0, str(folder.resolve()))
    spec = importlib.util.spec_from_file_location("tco_pinned_collector", args.collector)
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)
    worker = importlib.import_module("experiments.vttiny_gmc.round3_joint.worker")
    settings = json.loads(args.settings.read_bytes())
    from experiments.vttiny_gmc.round3_joint.common import prediction_settings as whitelist
    if settings != whitelist(settings):
        raise ValueError("Prediction settings must contain only the Round 3 prediction whitelist")
    source = pinned_source(worker, collector.WORKER_BLOB)
    groups = collector.read_groups(args.groups)
    if groups and args.sequences and not set(args.sequences) <= groups.keys():
        raise ValueError("A replay sequence is absent from the explicit source grouping")
    selection_required = args.mode not in ("off", "capture")
    if selection_required != bool(args.selection):
        raise ValueError("Provide the calibrated RGB checkpoint in record, fallback and active modes")
    observer = collector.Observer
    if args.mode != "capture":
        refiner = Refiner(args.selection, args.device) if selection_required else None
        observer = make_observer(collector, args.mode, refiner)
        if settings.get("runtime_root"):
            sys.path.insert(1, str(Path(settings["runtime_root"]) / "tracking_runtime"))
        adapter = importlib.import_module("botsort_plus.adapter_v2")
        patched = instrument_adapter(pinned_source(adapter, ADAPTER_BLOB), collector.one_replace)
        exec(compile(patched, str(adapter.__file__), "exec"), adapter.__dict__)
    worker.__dict__.update(_TcoObserver=observer, _tco_groups=groups, _tco_enabled=args.mode == "capture")
    exec(compile(collector.instrument(source), str(worker.__file__), "exec"), worker.__dict__)
    return worker.run(settings, args.method, args.output_dir, args.sequences, probe_mode="off")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--collector", required=True, type=Path, help="Source file of the verified T0 collector")
    parser.add_argument("--settings", required=True, type=Path, help="Prediction-only Round 3 configuration JSON")
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sequences", nargs="+")
    parser.add_argument("--groups", type=Path, help="Explicit source-video JSON or CSV")
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--device", default="cpu")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
