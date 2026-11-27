"""Score a completed M3OT branch replay matrix using existing GT and TrackEval code."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np


FIRST = ('native', 'B_early_moderate', 'dual_low', 'evidence_both')
SECOND = ('native_c2', 'b_c2', 'native_bam', 'b_bam')
SECOND_PAIRS = {'native_c2':'native','native_bam':'native','b_c2':'B_early_moderate','b_bam':'B_early_moderate'}


def _read(path):
    return json.loads(path.read_bytes())


def _sources(project, experiment, specification):
    first = project/'m3ot_first_batch_v1'
    second = project/'m3ot_second_batch_v1'
    prior_first = _read(first/'evaluation/evaluation.json')
    prior_second = _read(second/'evaluation/summary.json')
    if prior_first['status'] != 'PASS' or prior_second['status'] != 'PASS' or prior_first['gt'] != prior_second['gt']:
        raise ValueError('The two historical M3OT benchmark receipts differ in validity or data')
    expected = {(r['arm'],r['sequence']):r for r in prior_first['metrics'] if r['detector']=='higher'}
    expected.update({(r['arm'],r['sequence']):r for r in prior_second['metrics']})
    sources = {
        name: dict(path=first/'tracking/run/higher/trackers'/name/'data', reference='native',
                   hashes=prior_first['prediction_hashes']['higher'][name], family='M3OT 批次一')
        for name in FIRST
    }
    for name in SECOND:
        sources[name] = dict(path=second/'evaluation/trackers'/name/'data', reference=SECOND_PAIRS[name],
                             hashes=prior_second['prediction_hashes'][name], family='M3OT 批次二')
    for name, spec in specification['arms'].items():
        path=(experiment/spec['path']).resolve()
        if not path.is_relative_to(experiment) or name in sources or not name.replace('_','').isalnum():
            raise ValueError(f'Invalid matrix registration {name} {path}')
        sources[name] = dict(spec, path=path)
    if any(spec['reference'] not in sources for spec in sources.values()):
        raise ValueError('A matrix arm refers to a missing comparison group')
    return sources, prior_first['gt'], expected, prior_second


def run(project, experiment, specification, output):
    sys.path.insert(0, str(Path(__file__).resolve().parent/'batch1_evaluator'))
    from evaluate import (COUNT_METRICS,METRICS,TRACK_EVAL_COMMIT,check_prediction,
                          ground_truth,sha,write_csv,write_json)
    sys.path.insert(0, str(project/'tools'))
    from convert_gt_for_trackeval import convert_file
    from validate_evaluation import M3OTVehicle, trackeval

    sources, expected_gt, expected, protocol = _sources(project,experiment,specification)
    commit = subprocess.check_output(
        ['git','-C',str(Path(trackeval.__file__).resolve().parents[1]),'rev-parse','HEAD'],text=True).strip()
    if commit != TRACK_EVAL_COMMIT:
        raise ValueError('TrackEval differs from the previous M3OT batch')
    gt = ground_truth(project,output,convert_file)
    if gt != expected_gt:
        raise ValueError('RGB Val ground truth changed')
    tracker_root=output/'trackers'
    fingerprints={}
    for arm,spec in sources.items():
        src=spec['path']
        if {p.stem for p in src.glob('*.txt')} != set(gt):
            raise ValueError(f'The replay matrix lacks a matched sequence for {arm}: {src}')
        dest=tracker_root/arm/'data'
        dest.mkdir(parents=True)
        fingerprints[arm]={}
        for seq,rec in gt.items():
            source=src/(seq+'.txt')
            digest=sha(source)
            known=spec.get('hashes',{}).get(seq)
            if isinstance(known,dict):
                known=known['sha256']
            if known is not None and known != digest:
                raise ValueError(f'Prediction digest differs from its replay receipt: {arm} {seq}')
            count=check_prediction(source,rec['frames'])
            target=dest/source.name
            shutil.copyfile(source,target)
            if sha(target)!=digest:
                raise ValueError(f'MOT changed while preparing evaluation: {arm} {seq}')
            fingerprints[arm][seq]=dict(sha256=digest,rows=count)
    ds=M3OTVehicle(dict(GT_FOLDER=str(output/'gt'), TRACKERS_FOLDER=str(tracker_root),
                       OUTPUT_FOLDER=str(output/'trackeval'), TRACKERS_TO_EVAL=list(sources),
                       BENCHMARK='M3OT', SPLIT_TO_EVAL='val', SKIP_SPLIT_FOL=True, DO_PREPROC=False,
                       SEQ_INFO={s:x['frames'] for s,x in gt.items()}, PRINT_CONFIG=False))
    evaluator=trackeval.Evaluator(dict(PRINT_CONFIG=False, PRINT_RESULTS=False, TIME_PROGRESS=False,
                                      DISPLAY_LESS_PROGRESS=True, PLOT_CURVES=False, BREAK_ON_ERROR=True,
                                      LOG_ON_ERROR=str(output/'trackeval_errors.log')))
    results,_=evaluator.evaluate([ds],[trackeval.metrics.HOTA(), trackeval.metrics.CLEAR({'PRINT_CONFIG':False}),
                                       trackeval.metrics.Identity({'PRINT_CONFIG':False})])
    rows=[]
    for method,values in results['M3OTVehicle'].items():
        for sequence,classes in values.items():
            metrics=classes['vehicle']
            counts=metrics['CLEAR']
            row=dict(arm=method,family=sources[method]['family'],sequence=sequence,
                     **{k:float(np.mean(metrics['HOTA'][k])*100) for k in ('HOTA','DetA','AssA')},
                     IDF1=float(metrics['Identity']['IDF1']*100), MOTA=float(counts['MOTA']*100),
                     TP=int(counts['CLR_TP']),FP=int(counts['CLR_FP']),FN=int(counts['CLR_FN']),IDSW=int(counts['IDSW']))
            if (method,sequence) in expected and any(row[k]!=expected[method,sequence][k] for k in METRICS):
                raise ValueError(f'Historical score changed: {method} {sequence}')
            if sequence in gt and (row['TP']+row['FN']!=gt[sequence]['vehicle_rows'] or
                                   row['TP']+row['FP']!=fingerprints[method][sequence]['rows']):
                raise ValueError(f'TP, FP or FN count did not reconcile: {method} {sequence}')
            rows.append(row)
    table={(row['arm'],row['sequence']):row for row in rows}
    if len(table)!=len(sources)*(len(gt)+1):
        raise ValueError('Expected pooled and both individual scores for every arm')
    deltas=[]
    for method,spec in sources.items():
        ref=spec['reference']
        for k in COUNT_METRICS:
            if table[method,'COMBINED_SEQ'][k]!=sum(table[method,s][k] for s in gt):
                raise ValueError(f'Pooled count failed to reconcile: {method} {k}')
        for seq in (*gt,'COMBINED_SEQ'):
            values={k:table[method,seq][k]-table[ref,seq][k] for k in METRICS}
            identical=all(fingerprints[method][s]['sha256']==fingerprints[ref][s]['sha256']
                          for s in (gt if seq=='COMBINED_SEQ' else (seq,)))
            if identical and any(values.values()):
                raise ValueError(f'Identical tracks changed score: {method} {seq}')
            deltas.append(dict(arm=method,family=spec['family'],reference=ref,sequence=seq,
                               trajectory_byte_equal=identical,**values))
    write_csv(output/'metrics.csv',rows)
    write_csv(output/'deltas.csv',deltas)
    write_json(output/'evaluation.json',dict(status='PASS',scored_new_arms=list(specification['arms']),
        official_rgb_val_gt=gt,trackeval=commit,prior_scores_reproduced=True,
        frozen_main_detector_sha256=protocol['frozen_detector_sha256'],
        frozen_main_reid_sha256=protocol['frozen_reid_sha256'],
        prediction_sha256=fingerprints,metrics=rows,deltas=deltas))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('project','experiment','specification','output'):
        p.add_argument(name,type=Path)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    run(args.project.resolve(),args.experiment.resolve(),_read(args.specification),args.output)
