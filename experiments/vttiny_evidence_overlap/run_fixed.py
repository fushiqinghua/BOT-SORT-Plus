"""Run the archived evidence_both entry point, optionally composing its tracker with the verified BAM arm."""
import hashlib
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import run_botsort_plus as runner
import run_vttiny_gmc as gmc
from joint_settings import COMBINED, CONFIG, FIX, FIX_REPO, MOTION, OVERLAP, SOURCE, VENDOR, history, protocol, read, run_dir, sha, write


def checked_tracker(parent, name, output):
    class Checked(parent):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.warps = hashlib.sha256()
            self.frames_checked = 0
            self.accepted_low = 0
            self.weighted_low = 0
            self.changed_frames = 0

        def warp(self, image, dets):
            matrix = super().warp(image,dets)
            self.warps.update(np.asarray(matrix,dtype='<f8').tobytes())
            self.frames_checked += 1
            return matrix

        def event(self, kind, **data):
            if kind=='crowded_low':
                key=(data['tid'],data['det'])
                assert key not in self.current_updates, (self.seq,self.frame_count,key)
                assert 0 < data['area'] <= 1 and np.isclose(data['weight'],data['iou']*(1.-data['area']))
                self.current_updates.add(key)
            return super().event(kind,**data)

        def update(self, dets, image, *args, **kwargs):
            self.current_updates=set()
            rows=super().update(dets,image,*args,**kwargs)
            accepted={(e['track_id'],e['det_index']) for e in self.last_events
                      if e['event']=='match' and e['stage']=='low'}
            assert self.current_updates <= accepted, (self.seq,self.frame_count,self.current_updates-accepted)
            self.accepted_low += len(accepted)
            self.weighted_low += len(self.current_updates)
            self.changed_frames += bool(self.current_updates)
            return rows

        def finish_sequence_events(self):
            write(output/'joint_audit'/(self.seq+'.json'),dict(method=name,sequence=self.seq,
                  frames=self.frames_checked,matrix_sha256=self.warps.hexdigest(),accepted_low=self.accepted_low,
                  weighted_low=self.weighted_low,weighted_frames=self.changed_frames,lightglue_frames=self.lg_frames))
            return super().finish_sequence_events()
    return Checked


def main(name):
    assert name in (FIX,COMBINED) and gmc.REPOSITORY==runner.REPO==FIX_REPO
    output=run_dir(name)
    actual_run=runner.run
    ref=protocol(history(FIX),FIX)
    bam=protocol(history(OVERLAP),OVERLAP)['motion345']['source_sha256']

    def replay(args):
        import botsort_plus.native_adapter as adapter
        import run_arm
        assert Path(run_arm.__file__).resolve()==MOTION/'run_arm.py' and run_arm.runner is runner
        assert all(sha(MOTION/k)==v for k,v in bam.items())
        cfg=read(args.config)
        assert cfg['name']==FIX
        cfg['name']=name
        args.config=output/'configs'/(name+'.json')
        write(args.config,cfg)
        configure=runner.prepare

        def add_overlap(parsed):
            config,paths,record=configure(parsed)
            assert record['case_fix']==ref['case_fix'] and record['lost_camera_policy']==ref['lost_camera_policy']
            assert config.recovery.enabled and not config.identity.enabled
            assert record['config']==dict(ref['config'],name=name)
            tracker=adapter.BoTSORTPlus
            if name==COMBINED:
                tracker=run_arm.variant(tracker, OVERLAP, paths, output, None)
            adapter.BoTSORTPlus=checked_tracker(tracker,name,output)
            record['overlap_experiment']=dict(arm=name,base=FIX,overlap_enabled=name==COMBINED,
                fixed_source=sha(FIX_REPO/'botsort_plus/adapter_v2.py'),arm_source=sha(SOURCE/'run_fixed.py'),
                overlap_sources=bam,update_gate='accepted active second-pass match with overlapping adjacent previous observation')
            return config,paths,record
        with patch.object(runner,'prepare',add_overlap):
            return actual_run(args)

    argv=['run_vttiny_gmc.py','--settings',str(CONFIG),'--vendor',str(VENDOR),'--output-dir',str(output)]
    with patch.object(runner,'run',replay), patch.object(sys,'argv',argv):
        return gmc.main()


if __name__=='__main__':
    sys.exit(main(sys.argv[1]))
