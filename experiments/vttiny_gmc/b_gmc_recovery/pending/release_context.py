"""For each guarded release, compare official CLEAR ownership of the same earlier GT in the reference run."""
import csv
from collections import Counter
import numpy as np
from audit import owners
from driver import ROOT, ART, REF, j, save, settings
from common_audit import csvout
from evaluate_vttiny import VTTinyObjects


def main():
    variant = 'pending_guarded'
    directory = ART/'full'
    with (directory/'release_outcomes.csv').open(encoding='utf-8-sig') as f:
        decisions = [r for r in csv.DictReader(f) if r['method'] == variant]
    tracker_dir = ROOT/'scores/full/trackers'
    names = (REF, variant)
    protocol = j(ROOT/'runs/full'/variant/'protocols'/(variant+'.json'))
    ref_meta = j(ROOT/'scores/full/protocols'/(REF+'.json'))
    metrics = {(r['sequence'], r['tracker']): r for r in j(directory/'metrics/summary.json')['results']}
    records = []
    for seq in sorted({r['seq'] for r in decisions}):
        n = protocol['sequences'][seq]
        dataset=VTTinyObjects(dict(GT_FOLDER=settings['gt_root'],TRACKERS_FOLDER=str(tracker_dir),OUTPUT_FOLDER=str(ROOT/'offline'),
            TRACKERS_TO_EVAL=[variant],BENCHMARK='VT-Tiny-MOT',SPLIT_TO_EVAL='test',SKIP_SPLIT_FOL=True,SEQ_INFO={seq:n},
            DO_PREPROC=False,PRINT_CONFIG=False))
        raw=dataset.get_raw_seq_data(variant,seq)
        ledger=j(ROOT/'runs/full'/variant/'pending'/variant/(seq+'.json'))['events']
        probes={(e['frame'],e['tid'],e['det']):e for e in ledger if e['event']=='probe'}
        per_method = {}
        for m in names:
            ids, emitted, score = owners(tracker_dir, m, seq, n)
            official = metrics[seq, m]
            assert (score['CLR_TP'],score['CLR_FP'],score['CLR_FN'],score['IDSW']) == (sum(map(len, ids)),official['FP'],official['FN'],official['IDSW'])
            per_method[m] = ids, emitted
        for r in (e for e in decisions if e['seq'] == seq):
            start = int(r['frame']) - 1
            end = min(n, start+10)
            anchor = int(r['prior_gt']) if r['prior_gt'] else None
            tid = int(r['tid'])
            out = dict(seq=seq, frame=start+1, tid=tid, prior_gt=anchor, current_gt=r['current_gt'], outcome=r['outcome'])
            probe=probes[start+1,tid,int(r['det'])]
            observed=float(r['score']); app=probe['appearance']; gate=protocol['config']['recovery']
            via_high=observed>protocol['native_config']['track_high_thresh']
            via_rgb=app is not None and app<=gate['appearance_max_cost']-gate['ambiguity_margin']
            out.update(det=int(r['det']),score=observed,high_score=via_high,rgb_support=via_rgb,rgb_distance=app,
                       stable_low=probe['stable'],across_gap=probe['bridge'],association_cost=probe['cost'])
            column=np.flatnonzero(raw['tracker_ids'][start]==tid).item()
            mask=np.asarray(raw['gt_extras'][start]['zero_marked'])!=0
            truths=raw['gt_ids'][start][mask]
            overlap=np.asarray(raw['similarity_scores'][start])[mask,column]
            best=int(np.argmax(overlap)) if len(overlap) and max(overlap)>0 else None
            out['nearest_gt']=int(truths[best]) if best is not None else None
            out['nearest_gt_IoU']=float(overlap[best]) if best is not None else 0.
            pos=np.flatnonzero(truths==anchor)
            out['prior_gt_IoU']=float(overlap[pos.item()]) if len(pos) else None
            for m,label in zip(names, ('reference','guarded')):
                ids, _ = per_method[m]
                out[label+'_prior_gt_owner_at_release'] = next((pt for pt, gt in ids[start].items() if gt == anchor),None)
                out[label+'_prior_gt_matched_in_10'] = sum(anchor in f.values() for f in ids[start:end]) if anchor is not None else None
            ids, emitted = per_method[variant]
            out['guarded_original_id_correct_in_10'] = sum(f.get(tid)==anchor for f in ids[start:end]) if anchor is not None else None
            out['guarded_original_id_FP_in_10'] = sum(tid in emitted[t] and tid not in ids[t] for t in range(start,end))
            records.append(out)
    csvout(directory/'guarded_release_context.csv', records, list(records[0]) if records else ['seq','frame','tid','outcome'])
    changed = [s for s,h in protocol['output_prediction_hashes'].items() if h != ref_meta['output_prediction_hashes'][s]]
    recovered = [r for r in records if r['outcome']=='same_gt']
    save(directory/'guarded_release_context.json', dict(releases=len(records), outcomes=dict(Counter(r['outcome'] for r in records)),
         recovered_with_reference_absent=sum(r['reference_prior_gt_owner_at_release'] is None for r in recovered),
         recovered_with_reference_present=sum(r['reference_prior_gt_owner_at_release'] is not None for r in recovered),
         changed_prediction_sequences=changed, unchanged_prediction_sequences=len(protocol['sequences'])-len(changed)))


if __name__ == '__main__':
    main()
