"""Record exact output detection rows and accepted paths, with optional controlled interventions."""
import gzip
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import run_fixed
import run_arm
from boxmot.trackers.botsort.botsort_track import STrack
from joint_settings import COMBINED,FIX,ROOT,MOTION,read,write

MODE=sys.argv[1]
POLICY=MODE.removeprefix('full_')
SEQUENCES=sys.argv[2:]
RUN=ROOT.parent
TRACE=RUN/'trace'
REPO=Path(__file__).resolve().parent
spec_path=REPO/'interventions.json'
SPEC=read(spec_path).get(MODE,{}) if spec_path.exists() else {}
NATIVE_UPDATE=STrack.update
CAPTURE=None


def keep_posterior(track,obs,frame):
    prior=track.mean.copy() if CAPTURE is not None and CAPTURE.track_low_thresh < obs.conf < CAPTURE.track_high_thresh else None
    result=NATIVE_UPDATE(track,obs,frame)
    if CAPTURE is not None and CAPTURE.track_low_thresh < obs.conf < CAPTURE.track_high_thresh:
        CAPTURE.posteriors[int(track.id),int(obs.det_ind)]=(track,prior,track.mean.copy())
    return result


original_variant=run_arm.variant


def variant(*args):
    with patch.object(STrack,'update',keep_posterior):
        return original_variant(*args)


def traced(parent,name,output):
    class Trace(parent):
        def __init__(self,**kw):
            super().__init__(**kw)
            TRACE.mkdir(exist_ok=True,parents=True)
            self.sink=gzip.open(TRACE/(self.seq+'.jsonl.gz'),'wt',encoding='utf-8',compresslevel=2)
            spec=SPEC.get(self.seq,{})
            self.undo={tuple(x) for x in spec.get('undo',[])}
            self.undo_until=dict(spec.get('undo_until',[]))
            self.block_b={tuple(x) for x in spec.get('block_b',[])}
            self.probe_applied=[]
            self.last_weight={};self.last_score={};self.aux=[];self.guard_frames=0
            match=self.recovery_matcher.match
            def recovery(frame,*args):
                decision=match(frame,*args)
                allowed=[]
                for m in decision.matches:
                    if (frame,int(m.track_id),int(m.det_index)) in self.block_b:
                        self.probe_applied.append(['block_b',frame,int(m.track_id),int(m.det_index)])
                    else:allowed.append(m)
                decision.matches=allowed
                return decision
            self.recovery_matcher.match=recovery
            if POLICY=='b_guard':
                pair=self.recovery_matcher._pair
                def assess(frame,view,candidate,obs,app):
                    info=pair(frame,view,candidate,obs,app)
                    last=self.last_weight.get(view.track_id)
                    if (info.get('reason')=='plausible' and last is not None and
                        0<=view.last_seen-last<self.plus.recovery.window and
                        self.last_score.get(view.track_id,1.)<self.track_high_thresh and
                        all(o.score<self.track_high_thresh for o in obs)):
                        if info.get('appearance_cost') is None or info['appearance_cost']>.55:
                            self.guard_frames+=1
                            return dict(info,reason='bam_weak_appearance')
                        # A third observation is required, retaining rivalry and reservation.
                        return dict(info,two_hit_evidence=False)
                    return info
                self.recovery_matcher._pair=assess

        def event(self,kind,**data):
            if kind=='crowded_low':
                t,tid,d=self.frame_count,data['tid'],data['det']
                track,prior,posterior=self.posteriors[tid,d]
                if (t,tid,d) in self.undo or t<=self.undo_until.get(tid,0):
                    track.mean=posterior
                    self.probe_applied.append(['undo',t,tid,d])
                    self.aux.append(dict(data,decision='undo'))
                else:
                    self.last_weight[tid]=t
                    if POLICY=='velocity':
                        track.mean[4:]=posterior[4:]
                    if POLICY=='weight_floor' and data['weight']<.5:
                        track.mean=prior+.5*(posterior-prior)
                    self.aux.append(dict(data,decision='weight',applied_center=track.mean[:2].tolist()))
            return super().event(kind,**data)

        def update(self,dets,image,*args,**kwargs):
            global CAPTURE
            self.posteriors={};self.aux=[]
            CAPTURE=self
            try: rows=super().update(dets,image,*args,**kwargs)
            finally:CAPTURE=None
            if POLICY=='output_only':
                weighted={(e['tid'],e['det']) for e in self.aux}
                rows=rows.copy()
                for row in rows:
                    key=(int(row[4]),int(row[7]))
                    if key in weighted:
                        _,_,posterior=self.posteriors[key]
                        row[:4]=np.r_[posterior[:2]-posterior[2:4]/2,posterior[:2]+posterior[2:4]/2]
            events=[]
            for e in self.last_events:
                if e['event'] in ('match','recovered','native_birth','weak_birth','suppressed_native_birth',
                                   'deferred_birth','motion_output','recovery_check'):
                    if e['event']!='recovery_check' or e.get('reason') in ('weak_observations','uncertain_camera_recovery'):
                        events.append(e)
            outputs=[(int(row[4]),int(row[7])) for row in rows]
            self.last_score.update((int(row[4]),float(row[5])) for row in rows)
            self.sink.write(json.dumps(dict(frame=self.frame_count,events=events,outputs=outputs,bam=self.aux),
                                       separators=(',',':'),allow_nan=False)+'\n')
            return rows

        def finish_sequence_events(self):
            self.sink.close()
            write(TRACE/(self.seq+'.json'),dict(mode=MODE,applied=self.probe_applied,
                  expected_undo=len(self.undo),expected_block_b=len(self.block_b),guard_rejected_pairs=self.guard_frames))
            return super().finish_sequence_events()
    return Trace


def main():
    original_gmc=run_fixed.gmc.main
    checked=run_fixed.checked_tracker
    def gmc():
        if SEQUENCES:sys.argv.extend(['--sequences',*SEQUENCES])
        return original_gmc()
    def instrument(parent,name,output):
        return traced(checked(parent,name,output),name,output)
    with patch.object(run_fixed.gmc,'main',gmc),patch.object(run_fixed,'checked_tracker',instrument), \
         patch.object(run_arm,'variant',variant),patch.object(run_fixed,'run_dir',lambda m:RUN/'full'):
        return run_fixed.main(FIX if MODE=='fixed' else COMBINED)


if __name__=='__main__':sys.exit(main())
