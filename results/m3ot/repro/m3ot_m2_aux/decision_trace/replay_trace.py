"""Read-only assignment instrumentation for the archived triple and M2 stack."""
import argparse
import gzip
import json
from pathlib import Path
import sys
from unittest.mock import patch


ARMS = {"b_bam_low_r", "b_bam_low_r_m2"}
BASE = "b_bam_low_r"


def run(project, out, arm):
    prior = project / "m3ot_low_update_stack_v1/tracking"
    m2_home = prior / "m2"
    archived = project / "m3ot_branch_replay_v1/tracking/rules/research_archive"
    sys.path[:0] = [str(m2_home), str(archived), str(project / "tracking_runtime")]
    import numpy as np
    import replay
    from botsort_research import association as ash

    def matrix(a):
        a = np.asarray(a)
        return np.where(np.isfinite(a), a, None).tolist()

    def track_data(obj):
        return {"id": int(obj.id), "last_frame": int(obj.frame_id),
                "last_det": int(obj.det_ind), "state": int(obj.state),
                "pred": matrix(obj.mean[:4]), "pdiag": matrix(np.diag(obj.covariance)[:4])}

    def detection_data(obj):
        return {"det": int(obj.det_ind), "score": float(obj.conf), "box": matrix(obj.xywh)}

    def appearance(tracks, dets):
        def norm(x):
            if x is None:
                return None
            x = np.asarray(x, dtype=float).reshape(-1)
            d = np.linalg.norm(x)
            return x / d if d > 1e-12 and np.isfinite(d) else None
        a = {i: v for i, t in enumerate(tracks) if (v := norm(t.smooth_feat)) is not None}
        b = {i: v for i, t in enumerate(dets) if (v := norm(t.curr_feat)) is not None}
        dist = np.full((len(tracks), len(dets)), np.nan)
        if a and b:
            values = np.clip(1. - np.stack(list(a.values())) @ np.stack(list(b.values())).T, 0, 2)
            dist[np.ix_(list(a), list(b))] = values
        return matrix(dist)

    current = {}
    gaussian = ash.gaussian_cost
    original = replay.compose_m2

    def traced_gaussian(*args, **kwargs):
        costs, details = gaussian(*args, **kwargs)
        if current.get("inside"):
            current.update(costs=costs, details=details, parameters=ash.stage_config(args[5]))
        return costs, details

    def compose(*args, **kwargs):
        with patch.object(ash, "gaussian_cost", traced_gaussian):
            parent = original(*args, **kwargs)
        from botsort_plus import native_adapter

        class Observed(parent):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                d = out / "decisions" / arm
                d.mkdir(parents=True, exist_ok=True)
                self.trace = gzip.open(d / (self.m2_seq + ".jsonl.gz"), "wt",
                                       encoding="utf-8", compresslevel=4)
                self.written = 0

            def _m2_match(self, native_cost, tracks, dets, stage, threshold):
                current.clear()
                current["inside"] = True
                chosen_cost, chosen = super()._m2_match(native_cost, tracks, dets, stage, threshold)
                current["inside"] = False
                n_assign = native_adapter.linear_assignment(native_cost, thresh=threshold)
                iou_cost = native_adapter.iou_distance(tracks, dets)
                no_rgb = (native_adapter.fuse_score(iou_cost.copy(), dets)
                          if stage == "unconfirmed" or (stage == "high" and self.fuse_first_associate)
                          else iou_cost)
                no_rgb_pairs = native_adapter.linear_assignment(no_rgb, thresh=threshold)
                record = {"frame": int(self.frame_count), "stage": stage,
                          "tracks": [track_data(t) for t in tracks], "detections": [detection_data(d) for d in dets],
                          "native_cost": matrix(native_cost), "native_threshold": float(threshold),
                          "native_pairs": np.asarray(n_assign[0]).tolist(),
                          "actual_pairs": np.asarray(chosen[0]).tolist(),
                          "native_iou_cost": matrix(iou_cost), "no_rgb_cost": matrix(no_rgb),
                          "no_rgb_pairs": np.asarray(no_rgb_pairs[0]).tolist(),
                          "rgb_cosine_distance": appearance(tracks, dets)}
                if stage != "unconfirmed":
                    c, det, params = current["costs"], current["details"], current["parameters"]
                    choice = ash.assign_with_dummies(c, params["miss_cost"], params["unassigned_cost"])
                    gated_native = np.where(det["gate_reason"] == 0, native_cost, threshold + 1)
                    gates_only = native_adapter.linear_assignment(gated_native, thresh=threshold)
                    record.update(nll_cost=matrix(c), gate_reason=det["gate_reason"].tolist(),
                                  nll_pairs=np.asarray(choice[0]).tolist(),
                                  gates_only_pairs=np.asarray(gates_only[0]).tolist(),
                                  mahalanobis2=matrix(det["mahalanobis2"]), logdet=matrix(det["logdet"]),
                                  rdiag=matrix(np.diagonal(det["measurement_covariance"], axis1=2, axis2=3)),
                                  reject_cost=params["miss_cost"] + params["unassigned_cost"])
                else:
                    choice = n_assign
                expected = choice if kwargs.get("live", False) and stage != "unconfirmed" else n_assign
                if not np.array_equal(np.asarray(chosen[0]), np.asarray(expected[0])):
                    raise ValueError("Observer did not reproduce tracker assignment")
                self.trace.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
                self.written += 1
                return chosen_cost, chosen

            def finish_sequence_events(self):
                self.trace.close()
                if self.written != 3 * replay.VAL[self.m2_seq]:
                    raise ValueError("Expected one decision row per stage and frame")
                return super().finish_sequence_events()

        return Observed

    if arm == BASE:
        replay.ARMS[BASE] = (False, True, True, False)
    with patch.object(replay, "compose_m2", compose):
        replay.run(project, out, arm)
    new = replay.read(out / "receipts" / (arm + ".json"))
    if arm == BASE:
        folder = prior / "low_update"
        baseline = replay.read(folder / "receipt.json")
        registered = {"sequences": {
            seq: {"sha256": baseline["output_prediction_hashes"][arm][seq],
                  "event_sha256": replay.sha(folder / "events" / arm / (seq + ".jsonl"))}
            for seq in replay.VAL
        }}
    else:
        registered = replay.read(prior / "m2" / "receipts" / (arm + ".json"))
    fields = ("sha256", "event_sha256")
    proof = {}
    for seq in replay.VAL:
        expected, observed = registered["sequences"][seq], new["sequences"][seq]
        if any(expected[k] != observed[k] for k in fields):
            raise ValueError(f"Instrumentation changed frozen MOT or events for {arm}/{seq}")
        path = out / "decisions" / arm / (seq + ".jsonl.gz")
        proof[seq] = dict(mot_sha256=observed["sha256"], events_sha256=observed["event_sha256"],
                          trace_sha256=replay.sha(path), stages=replay.VAL[seq] * 3)
    replay.store(out / "parity" / (arm + ".json"),
                 dict(status="BYTE_IDENTICAL_MOT_AND_EVENTS", arm=arm, sequences=proof))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("project", type=Path)
    cli.add_argument("out", type=Path)
    cli.add_argument("arm", choices=sorted(ARMS))
    opts = cli.parse_args()
    run(opts.project, opts.out, opts.arm)
