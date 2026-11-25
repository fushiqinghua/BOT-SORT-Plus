"""Single-factor checks on the frozen A adapter after its first diagnostic replay."""
import argparse
import runpy
import sys
from pathlib import Path


def instrument(probe):
    from botsort_plus.candidates import CandidatePool
    from botsort_plus.native_adapter import BoTSORTPlus

    if probe == "empty_pool":
        def empty_pool(self, frame, observations, warp=None, occupied_boxes=()):
            self.events = []
            return []
        CandidatePool.update = empty_pool
        return

    original_remove = CandidatePool.remove
    original_prepare = BoTSORTPlus._prepare_output
    candidate_velocities = {}

    def remove(self, candidate_id, reason="consumed"):
        candidate = original_remove(self, candidate_id, reason)
        if reason == "born":
            candidate_velocities[candidate_id] = candidate.velocity.copy()
        return candidate

    def prepare(self, activated, refound, lost, removed):
        tracks = {t.id: t for t in activated}
        for event in self.last_events:
            if event["event"] == "weak_birth":
                t = tracks[event["track_id"]]
                # The native XYWH Kalman state puts center velocity at 4:6.
                # Covariance, extent velocity, confidence and identity are unchanged.
                t.mean[4:6] = candidate_velocities.pop(event["candidate_id"])
        assert not candidate_velocities
        return original_prepare(self, activated, refound, lost, removed)

    CandidatePool.remove = remove
    BoTSORTPlus._prepare_output = prepare


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--probe", choices=["empty_pool", "carry_velocity"], required=True)
    args, cli = parser.parse_known_args()
    sys.path.insert(0, str(args.repo))
    instrument(args.probe)
    entry = args.repo / "tools/run_botsort_plus.py"
    sys.argv = [str(entry), *cli]
    runpy.run_path(str(entry), run_name="__main__")
