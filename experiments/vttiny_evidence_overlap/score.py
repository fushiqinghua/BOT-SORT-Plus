"""Verify full replays, run official TrackEval, and measure the 2 x 2 interaction and switch events."""
from collections import Counter
from pathlib import Path
import subprocess
import sys
from joint_settings import (BASE_REPO, COMBINED, CONFIG, COUNTS, EVALUATOR, FIX, FORENSICS, MEASURES,
    METHODS, MOTION, NATIVE, OUT, OVERLAP, ROOT, history, protocol, read, run_dir, sha, write, write_csv)

COMMON = ('sequences','native_config','detector_sha256','rgb_encoder_sha256','source_protocol_sha256',
          'detection_fingerprints','feature_hashes','decoded_image_hashes','encoder_contracts','boxmot_source_hashes')
POOLED='COMBINED_SEQ'


def link(target, source):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.is_symlink() or target.exists():
        assert target.resolve()==source.resolve(),target
    else:
        target.symlink_to(source)


def verify_runs():
    joint=ROOT/'full_evaluation'
    runs={m:protocol(run_dir(m),m) for m in METHODS}
    base=runs[NATIVE]
    seqs=base['sequences']
    assert len(seqs)==30 and sum(seqs.values())==12590
    history_hashes={}
    for m,p in runs.items():
        directory=run_dir(m)
        assert p['status']=='COMPLETE' and not p['subset'] and p['causal'] and not p['interpolation']
        assert p['backfilled_frames']==0 and p['missing_count']==0
        for k in COMMON: assert p[k]==base[k], (m,k)
        if m!=COMBINED:
            old=protocol(history(m),m)
            for k in (*COMMON,'output_prediction_hashes'): assert p[k]==old[k], (m,k,'history')
            for s in seqs:
                assert sha(history(m)/'trackers'/m/'data'/(s+'.txt'))==old['output_prediction_hashes'][s]
            history_hashes[m]=len(seqs)
        link(joint/'protocols'/(m+'.json'),directory/'protocols'/(m+'.json'))
        link(joint/'trackers'/m,directory/'trackers'/m)
    for k in ('case_fix','lost_camera_policy','implementation_sha256'):
        assert runs[FIX][k]==runs[COMBINED][k],k
    assert runs[COMBINED]['config']==dict(runs[FIX]['config'],name=COMBINED)
    assert runs[COMBINED]['overlap_experiment']['overlap_sources']==runs[OVERLAP]['motion345']['source_sha256']
    assert runs[FIX]['overlap_experiment']['overlap_enabled'] is False
    assert runs[COMBINED]['overlap_experiment']['overlap_enabled'] is True
    matrices=[]; hashes=[]
    for s,n in seqs.items():
        a,b=[read(run_dir(m)/'joint_audit'/(s+'.json')) for m in (FIX,COMBINED)]
        assert a['frames']==b['frames']==n and a['matrix_sha256']==b['matrix_sha256']
        assert a['lightglue_frames']==b['lightglue_frames'] and a['weighted_low']==0
        d=read(run_dir(COMBINED)/'diagnostics345'/OVERLAP/(s+'.json'))['counts']
        assert b['weighted_low']==d.get('crowded_low',0)<=b['accepted_low']
        matrices.append(dict(sequence=s,frames=n,matrix_sha256=b['matrix_sha256'],lightglue_frames=b['lightglue_frames'],
                        accepted_low_fixed=a['accepted_low'],accepted_low_combined=b['accepted_low'],
                        weighted_low=b['weighted_low'],weighted_frames=b['weighted_frames']))
        row=dict(sequence=s,**{m:runs[m]['output_prediction_hashes'][s] for m in METHODS})
        hashes.append(dict(row,fix_combo_byte_equal=int(row[FIX]==row[COMBINED])))
    assert sum(r['weighted_low'] for r in matrices)>0
    write_csv(OUT/'camera_and_updates.csv',matrices)
    write_csv(OUT/'trajectory_hashes.csv',hashes)
    audit=dict(sequences=len(seqs),frames=sum(seqs.values()),full_replays=len(METHODS),
        replayed_frames_all_methods=sum(seqs.values())*len(METHODS),historical_MOT_files_exact=history_hashes,
        common_input_fields=COMMON,detector_sha256=base['detector_sha256'],rgb_encoder_sha256=base['rgb_encoder_sha256'],
        fixed_combo_camera_exact_frames=sum(seqs.values()),fix_combo_MOT_files_exact=sum(r['fix_combo_byte_equal'] for r in hashes),
        weighted_updates=sum(r['weighted_low'] for r in matrices),accepted_low_combo=sum(r['accepted_low_combined'] for r in matrices),
        weighted_frames=sum(r['weighted_frames'] for r in matrices),base_policy=runs[FIX]['lost_camera_policy'],
        protocol_sha256={m:sha(run_dir(m)/'protocols'/(m+'.json')) for m in METHODS})
    return joint,seqs,audit


def official(joint,seqs):
    subprocess.run([sys.executable,'-Bu',str(BASE_REPO/'tools/evaluate_botsort_plus.py'),
        '--run-dir',str(joint),'--trackers',*METHODS,'--reference-tracker',NATIVE,
        '--gt-root',read(CONFIG)['gt_root'],'--trackeval-root',str(EVALUATOR),'--output-dir',str(OUT/'metrics')],check=True)
    rows=read(OUT/'metrics/summary.json')['results']
    scores={(r['tracker'],r['sequence']):r for r in rows}
    assert len(scores)==len(METHODS)*(len(seqs)+1)
    for m in (NATIVE,OVERLAP,FIX):
        path=(FORENSICS/'results/full/summary.json' if m==FIX else MOTION/'results/metrics/summary.json')
        old=[r for r in read(path)['results'] if r['tracker']==m]
        assert len(old)==len(seqs)+1
        for r in old:
            assert all(abs(scores[m,r['sequence']][k]-r[k]) < 1.e-9 for k in MEASURES)
    write_csv(OUT/'tracking_by_sequence.csv',rows)
    write_csv(OUT/'tracking_combined.csv',[scores[m,POOLED] for m in METHODS])
    return scores


def contrasts(scores,seq):
    n,o,f,c=(scores[m,seq] for m in METHODS)
    return {
        'overlap_on_native':{k:o[k]-n[k] for k in MEASURES},
        'fix_on_native':{k:f[k]-n[k] for k in MEASURES},
        'overlap_on_fix':{k:c[k]-f[k] for k in MEASURES},
        'combined_on_native':{k:c[k]-n[k] for k in MEASURES},
        'additive_expectation':{k:f[k]+o[k]-n[k] for k in MEASURES},
        'interaction':{k:(c[k]-f[k])-(o[k]-n[k]) for k in MEASURES}}


def clear_events(owner_module,method,seq,n,score):
    owner_module.RUN=run_dir(method)
    owners,truth,centers=owner_module.owners(method,seq,n,score)
    previous={}; switches={}; misses=set(); fp=[]
    for t in range(1,n+1):
        assigned=owners[t]
        for tid,gid in assigned.items():
            if gid in previous:
                old_t,old_id=previous[gid]
                if old_id!=tid:switches[t,gid]=(old_t,old_id,tid)
            previous[gid]=(t,tid)
        for gid in truth[t].keys()-set(assigned.values()):misses.add((t,gid))
        fp.append(len(centers[t])-len(assigned))
    assert (len(switches),len(misses),sum(fp))==(score['IDSW'],score['FN'],score['FP'])
    return switches,misses,fp


def event_audit(seqs,scores):
    import evaluate as historical
    assert Path(historical.__file__).resolve()==MOTION/'evaluate.py'
    by_seq=[]; event_rows=[]; all_counts=Counter()
    for seq,n in seqs.items():
        states={m:clear_events(historical,m,seq,n,scores[m,seq]) for m in METHODS}
        nn,oo,ff,cc=(set(states[m][0]) for m in METHODS)
        removed_native,added_native=nn-oo,oo-nn
        removed_fix,added_fix=ff-cc,cc-ff
        fn_before,fn_after=states[FIX][1],states[COMBINED][1]
        fp_difference=[b-a for a,b in zip(states[FIX][2],states[COMBINED][2],strict=True)]
        counts=dict(native_switches_removed=len(removed_native),native_switches_added=len(added_native),
                    fixed_switches_removed=len(removed_fix),fixed_switches_added=len(added_fix),
                    same_removed_switch_frame_gt=len(removed_native&removed_fix),
                    standalone_removals_already_absent_in_fix=len(removed_native-ff),
                    same_added_switch_frame_gt=len(added_native&added_fix),
                    fixed_misses_recovered=len(fn_before-fn_after),combined_new_misses=len(fn_after-fn_before),
                    fewer_FP_on_frames=sum(-x for x in fp_difference if x<0),
                    extra_FP_on_frames=sum(x for x in fp_difference if x>0))
        d=contrasts(scores,seq)['overlap_on_fix']
        assert counts['fixed_switches_added']-counts['fixed_switches_removed']==d['IDSW']
        assert counts['combined_new_misses']-counts['fixed_misses_recovered']==d['FN']
        assert counts['extra_FP_on_frames']-counts['fewer_FP_on_frames']==d['FP']
        by_seq.append(dict(sequence=seq,**counts))
        all_counts.update(counts)
        for t,g in sorted(nn|oo|ff|cc):
            row=dict(sequence=seq,frame=t,GT=g)
            for m in METHODS:
                event=states[m][0].get((t,g))
                row[m]=int(event is not None)
                row[m+'_previous_frame'],row[m+'_previous_ID'],row[m+'_new_ID']=event or ('','','')
            event_rows.append(row)
        print('Checked official events',seq,flush=True)
    write_csv(OUT/'events_by_sequence.csv',by_seq)
    write_csv(OUT/'switch_events.csv',event_rows)
    write(OUT/'event_totals.json',dict(all_counts))


def main():
    joint,seqs,audit=verify_runs()
    scores=official(joint,seqs)
    totals=contrasts(scores,POOLED)
    each=[]
    for seq in sorted(seqs):
        comparison=contrasts(scores,seq)
        row={'sequence':seq,'frames':seqs[seq]}
        for key in ('overlap_on_native','overlap_on_fix','interaction'):
            row.update({key+'_'+k:v for k,v in comparison[key].items()})
        each.append(row)
    assert all(sum(r['overlap_on_fix_'+k] for r in each)==totals['overlap_on_fix'][k] for k in COUNTS)
    write_csv(OUT/'factorial_contrasts.csv',[dict(contrast=kind,**v) for kind,v in totals.items()])
    write_csv(OUT/'sequence_contrasts.csv',each)
    event_audit(seqs,scores)
    audit.update(official_rows=len(scores),controls_with_full_historical_metrics_exact=[NATIVE,OVERLAP,FIX],
                 switch_and_error_counts_checked=len(METHODS)*len(seqs),status='PASS')
    write(OUT/'verification.json',audit)
    from report import generate
    generate(OUT)


if __name__=='__main__':
    main()
