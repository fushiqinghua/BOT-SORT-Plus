"""Diagnostic launcher. Only the process-local class is instrumented; source checkout stays frozen."""
import argparse
import importlib
import runpy
import sys
from pathlib import Path

import numpy as np


def instrument(probe):
    from botsort_plus.native_adapter import BoTSORTPlus
    native = importlib.import_module("boxmot.trackers.botsort.botsort")
    original = BoTSORTPlus._prepare_output
    remove = native.remove_duplicate_stracks

    def prepare(self, activated, refound, lost, removed):
        births = {x["track_id"] for x in self.last_events if x["event"] == "weak_birth"}
        if not births:
            return original(self, activated, refound, lost, removed)
        if probe == "native_confirmation":
            # Same birth decision, then use native unconfirmed lifecycle and next high-score association.
            for track in activated:
                if track.id in births:
                    track.is_activated = False
            return original(self, activated, refound, lost, removed)

        def inspect(active, lost_pool):
            kept_active, kept_lost = remove(active, lost_pool)
            kept = {x.id for x in kept_active}
            weak = [x for x in active if x.id in births]
            assert len(weak) == len(births)
            iou = 1.0-native.iou_distance(weak,lost_pool)
            for idx, t in enumerate(weak):
                opponents = [dict(id=int(other.id),last_seen=int(other.end_frame),
                                  iou=float(iou[idx,j]),state=int(other.state))
                             for j,other in enumerate(lost_pool)]
                self.last_events.append(dict(event="weak_handoff", track_id=int(t.id),
                                             kept=t.id in kept, velocity=np.asarray(t.mean[4:]).tolist(),
                                             smooth_feature=t.smooth_feat is not None, lost=opponents))
            return kept_active, kept_lost

        native.remove_duplicate_stracks = inspect
        try:
            return original(self, activated, refound, lost, removed)
        finally:
            native.remove_duplicate_stracks = remove

    BoTSORTPlus._prepare_output = prepare


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--repo",type=Path,required=True)
    p.add_argument("--probe",choices=["audit","native_confirmation"],required=True)
    args,cli=p.parse_known_args()
    sys.path.insert(0,str(args.repo))
    instrument(args.probe)
    entry=args.repo/"tools/run_botsort_plus.py"
    sys.argv=[str(entry),*cli]
    runpy.run_path(str(entry),run_name="__main__")
