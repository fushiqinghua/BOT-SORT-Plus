"""Run official pooled full-test evaluation for an online repair and its frozen controls."""
from pathlib import Path
import subprocess
import sys
from workflow import JOINT,ROOT
from joint_settings import BASE_REPO, COMBINED, CONFIG, EVALUATOR, FIX, METHODS, protocol, read, sha, write, write_csv

FIELDS=('feature_hashes','decoded_image_hashes','detection_fingerprints','sequences','native_config',
        'detector_sha256','rgb_encoder_sha256','source_protocol_sha256','encoder_contracts','boxmot_source_hashes')


def link(dst,src):
    if dst.exists():assert dst.resolve()==src.resolve()
    else:dst.symlink_to(src)


def main(mode):
    assert mode.startswith('full_')
    prospective=ROOT/'runs'/mode/'full'
    current=protocol(prospective,COMBINED)
    old_path=JOINT/'runs'/COMBINED/'full'
    old=protocol(old_path,COMBINED)
    focus=protocol(ROOT/'runs'/mode.removeprefix('full_')/'full',COMBINED)
    assert current['status']==focus['status']=='COMPLETE'
    assert all(current[k]==old[k] for k in (*FIELDS,'case_fix','lost_camera_policy','config'))
    assert focus['output_prediction_hashes']=={s:current['output_prediction_hashes'][s] for s in focus['sequences']}
    for s in old['sequences']:
        assert read(prospective/'joint_audit'/(s+'.json'))['matrix_sha256']==read(old_path/'joint_audit'/(s+'.json'))['matrix_sha256']
        assert sha(prospective/'trackers'/COMBINED/'data'/(s+'.txt'))==current['output_prediction_hashes'][s]
    scoring=ROOT/'scoring'/mode
    out=ROOT/'results'/mode
    for folder in ('trackers','protocols'): (scoring/folder).mkdir(parents=True,exist_ok=True)
    for method in METHODS:
        existing=JOINT/'runs'/method/'full'
        link(scoring/'trackers'/method,existing/'trackers'/method)
        link(scoring/'protocols'/(method+'.json'),existing/'protocols'/(method+'.json'))
    write(scoring/'protocols'/(mode+'.json'),dict(current,name=mode,config=dict(current['config'],name=mode)))
    link(scoring/'trackers'/mode,prospective/'trackers'/COMBINED)
    subprocess.run([sys.executable,'-Bu',str(BASE_REPO/'tools/evaluate_botsort_plus.py'),'--run-dir',str(scoring),
        '--trackers',*METHODS,mode,'--reference-tracker',COMBINED,'--gt-root',read(CONFIG)['gt_root'],
        '--trackeval-root',str(EVALUATOR),'--output-dir',str(out/'metrics')],check=True)
    rows=read(out/'metrics/summary.json')['results']
    old_rows={(r['tracker'],r['sequence']):r for r in read(JOINT/'results/metrics/summary.json')['results']}
    assert len(rows)==155 and all(r==old_rows[r['tracker'],r['sequence']] for r in rows if r['tracker']!=mode)
    write_csv(out/'tracking.csv',rows)
    write(out/'parity.json',dict(full_sequences=30,exact_camera_sequences=30,exact_focus_predictions=len(focus['sequences']),
          controlled_metrics_rows=len(old_rows),mode=mode,protocol_sha256=sha(prospective/'protocols'/(COMBINED+'.json'))))


if __name__=='__main__':main(sys.argv[1])
