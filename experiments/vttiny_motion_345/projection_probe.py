"""Count actual forward-versus-bidirectional cost changes on the official replay, without reading GT."""
from collections import Counter
from pathlib import Path
from tempfile import mkdtemp
import numpy as np
import run_arm
from perspective import association_cost
from settings import RESULT,ROOT,read,save,table


def main():
    method='homography_bidir'
    replay_dir=ROOT/'cost_replay'
    replay_dir.mkdir(parents=True,exist_ok=True)
    current=Path(mkdtemp(prefix='run-',dir=replay_dir))
    profiles={}
    upstream=run_arm.native_module.iou_distance
    make=run_arm.variant
    def measured(*args):
        parent=make(*args)
        class Probe(parent):
            def __init__(self,**kw):
                super().__init__(**kw)
                self.audit=Counter()
                self.highest=0.
            def match_cost(self,tracks,dets):
                value=super().match_cost(tracks,dets)
                if self.Hinv is None or not value.size: return value
                forward=association_cost(upstream,tracks,dets,self.old_boxes,self.H,None)
                delta=np.abs(forward-value)
                self.highest=max(self.highest,float(delta.max()))
                self.audit['calls']+=1
                self.audit['pairs']+=int(delta.size)
                self.audit['pairs_with_any_overlap']+=int(((forward<1)|(value<1)).sum())
                self.audit['pairs_delta_gt_1e_6']+=int((delta>1e-6).sum())
                self.audit['pairs_delta_gt_0_01']+=int((delta>.01).sum())
                useful=(np.minimum(forward,value)<1).any(axis=1)
                self.audit['rows_with_overlap']+=int(useful.sum())
                self.audit['row_best_candidate_changed']+=int(((forward.argmin(axis=1)!=value.argmin(axis=1))&useful).sum())
                return value
            def finish_sequence_events(self):
                profiles[self.seq]=dict(self.audit,max_abs_cost_delta=self.highest)
                return super().finish_sequence_events()
        return Probe
    run_arm.variant=measured
    run_arm.ROOT=current
    code=run_arm.main('full',method)
    if code: raise RuntimeError(code)
    diagnostic=read(current/'full/protocols'/(method+'.json'))
    forward=read(ROOT/'full/protocols/homography_forward.json')
    bidir=read(ROOT/'full/protocols'/(method+'.json'))
    assert diagnostic['output_prediction_hashes']==bidir['output_prediction_hashes']
    identical=[]
    for seq in diagnostic['sequences']:
        if bidir['output_prediction_hashes'][seq]==forward['output_prediction_hashes'][seq]:identical.append(seq)
        ours=read(current/'full/diagnostics345'/method/(seq+'.json'))
        theirs=read(ROOT/'full/diagnostics345'/method/(seq+'.json'))
        assert ours==theirs
    keys=('calls','pairs','pairs_with_any_overlap','pairs_delta_gt_1e_6','pairs_delta_gt_0_01',
          'rows_with_overlap','row_best_candidate_changed')
    totals={key:sum(g.get(key,0) for g in profiles.values()) for key in keys}
    save(RESULT/'bidirectional_cost_audit.json',dict(totals=totals,max_abs_cost_delta=max(g['max_abs_cost_delta'] for g in profiles.values()),
         forward_bidir_identical_sequences=identical,probe_exact_output_and_geometry=len(profiles)))
    table(RESULT/'bidirectional_costs_by_sequence.csv',[dict(sequence=s,**{k:v.get(k,0) for k in keys},max_abs_cost_delta=v['max_abs_cost_delta'])
          for s,v in profiles.items()])
    print('Cost replay checked',totals,flush=True)


if __name__=='__main__':
    main()
