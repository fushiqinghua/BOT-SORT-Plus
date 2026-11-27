"""Compare both score-restricted interventions to the frozen native and all-R runs."""
from pathlib import Path
import subprocess
import sys
from rf_settings import ROOT, QR, OUT, ORIG, EVAL, SETTINGS, CASE, read, write, sha, environment, table

METHODS=('native','learned_r','low_r','high_r')
METRICS=('HOTA','DetA','AssA','IDF1','MOTA','FP','FN','IDSW')
KEYS=('sequences','subset','detection_fingerprints','feature_hashes','decoded_image_hashes',
      'source_protocol_sha256','detector_sha256','rgb_encoder_sha256','encoder_contracts',
      'native_config','boxmot_source_hashes')

def main():
    source=QR/'full'
    prior=QR/'results/metrics'
    folder=ROOT/'full'
    metrics=OUT/'metrics'
    ctrl=read(source/'protocols/native.json')
    assert len(ctrl['sequences'])==30 and sum(ctrl['sequences'].values())==12590 and not ctrl['subset']
    for method in METHODS:
        base=source if method in METHODS[:2] else folder
        proto=read(base/'protocols'/(method+'.json'))
        assert proto['status']=='COMPLETE' and all(proto[k]==ctrl[k] for k in KEYS),method
        for seq,h in proto['output_prediction_hashes'].items():
            assert sha(base/'trackers'/method/'data'/(seq+'.txt'))==h,(method,seq)
        pilot=ROOT/'pilot/trackers'/method/'data'/(CASE+'.txt')
        assert sha(pilot)==proto['output_prediction_hashes'][CASE],method
    gs=read(SETTINGS)
    subprocess.run([sys.executable,'-Bu',str(ORIG/'tools/evaluate_botsort_plus.py'),
        '--run-dir',str(folder),'--trackers','low_r','high_r','--reference-tracker','low_r',
        '--gt-root',gs['gt_root'],'--trackeval-root',str(EVAL),'--output-dir',str(metrics)],
        env=environment(),check=True)
    old=read(prior/'evaluation_audit.json')
    current=read(metrics/'evaluation_audit.json')
    for k in ('ground_truth','evaluator'):assert old[k]==current[k],k
    assert sha(prior/'summary.json')==old['summary_sha256'] and sha(metrics/'summary.json')==current['summary_sha256']
    merged=[r for r in read(prior/'summary.json')['results'] if r['tracker'] in METHODS[:2]]
    merged+=read(metrics/'summary.json')['results']
    lookup={(r['tracker'],r['sequence']):r for r in merged}
    assert len(lookup)==len(merged)==4*31
    result=[]
    for seq in ['COMBINED_SEQ',*sorted(ctrl['sequences'])]:
        for method in METHODS:
            row=lookup[method,seq].copy()
            for base in ('native','learned_r'):
                row.update({f'delta_{base}_{k}':row[k]-lookup[base,seq][k] for k in METRICS})
            result.append(row)
    table(OUT/'four_way_metrics.csv',result)
    write(OUT/'four_way_audit.json',dict(status='PASS',sequences=30,frames=12590,
          unchanged_inputs=list(KEYS),pilot_matches_full=list(METHODS),evaluator_commit=current['evaluator']['commit'],
          old_summary=old['summary_sha256'],new_summary=current['summary_sha256'],
          r_policy_sha256=sha(QR/'results/r_policy.json')))
    print('Four-way scores',[(r['tracker'],r['FN'],r['IDSW']) for r in result[:4]],flush=True)

if __name__=='__main__':main()
