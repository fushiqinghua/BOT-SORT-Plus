"""Reproduce the native first-association pool on the two differing empty-pool sequences."""
import argparse
import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    args, cli = parser.parse_known_args()
    sys.path.insert(0, str(args.repo))
    import botsort_plus.native_adapter as adapter
    from probe_botsort_a_followup import instrument

    instrument("empty_pool")
    update = adapter.BoTSORTPlus.update
    joint = adapter.joint_stracks

    def use_native_lost_pool(self, *inputs, **kwargs):
        def first_joint(active, filtered_lost):
            # Verify the call is the specific pool construction under study.
            assert [id(t) for t in filtered_lost] == [
                id(t) for t in self.lost_stracks if t.state == adapter.TrackState.Lost]
            adapter.joint_stracks = joint
            return joint(active, self.lost_stracks)
        adapter.joint_stracks = first_joint
        try:
            return update(self, *inputs, **kwargs)
        finally:
            adapter.joint_stracks = joint

    adapter.BoTSORTPlus.update = use_native_lost_pool
    entry = args.repo / "tools/run_botsort_plus.py"
    sys.argv = [str(entry), *cli]
    runpy.run_path(str(entry), run_name="__main__")
