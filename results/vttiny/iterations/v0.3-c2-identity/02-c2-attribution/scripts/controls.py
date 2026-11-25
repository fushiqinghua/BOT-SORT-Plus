"""Isolated interventions on existing memory writers and frozen-state assignment."""
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

job=Path(__file__).resolve().parent
stage=job.parents[1]
repo=stage/'BOT-SORT-Plus'
project=stage.parents[1]
official=stage/'experiments/c2_full_v1/full30'
runtime=project.parent/'M3OT_MOT_Baseline_v0_1'
old=stage.parent/'botsort_plus_v0_1_0'
work=job/'controls'
logs=job/'logs'
logs.mkdir(exist_ok=True)
def replace(src,a,b):
    assert src.count(a)==1, a
    return src.replace(a,b)
def experiment(name):
    src=job/'sources'/name
    (src/'tools').mkdir(parents=True)
    shutil.copytree(repo/'botsort_plus',src/'botsort_plus',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(repo/'tools/run_botsort_plus.py',src/'tools/run_botsort_plus.py')
    path=src/'botsort_plus/identity_adapter.py'
    code=path.read_text()
    if name=='fixed_state':
        hook="""    def pairs(matrix):
        return {(int(tracks[i].id),int(detections[j].det_ind)) for i,j in linear_assignment(matrix,thresh=threshold)[0]}
    old_pairs=pairs(reference)
    new_pairs=pairs(proposed)
    tracker.last_events.append(dict(event='fixed_assignment',stage=stage,
        old_count=len(old_pairs),new_count=len(new_pairs),changed_costs=int(np.count_nonzero(np.abs(proposed-reference)>1e-12)),
        discounted=int(np.count_nonzero(base-reference>1e-12)), pairs=base.size, nonempty=int(bool(base.size)),
        old_only=sorted(old_pairs-new_pairs),new_only=sorted(new_pairs-old_pairs)))
"""
        code=replace(code,"    matches, u_tracks, u_detections = linear_assignment",hook+"    matches, u_tracks, u_detections = linear_assignment")
    else:
        tag="    block_write = gate_applied and not decision.allow_write"
        block={"native_only":"False","bank_only":"block_write","release_bank_history":"block_write and decision.reason != 'history_conflict'","release_bank_recovery":"block_write and decision.stage != 'recovery'"}[name]
        code=replace(code,tag,tag+"\n    bank_block = "+block+("\n    block_write = False" if name=="bank_only" else ""))
        code=replace(code,"updated = [] if block_write else","updated = [] if bank_block else")
        tag="    blocked = cfg.memory_enabled and not cfg.observe_only and not accepted"
        code=replace(code,tag,tag+"\n    bank_block = "+("False" if name=="native_only" else "blocked")+("\n    blocked = False" if name=="bank_only" else ""))
        code=replace(code,"updated = [] if blocked else","updated = [] if bank_block else")
    path.write_text(code)
    config=json.loads((repo/'configs/botsort_plus'/('27_B_C2_shadow.json' if name=='fixed_state' else '28_B_C2_memory.json')).read_bytes())
    config['name']=name
    (src/'config.json').write_text(json.dumps(config,indent=2))
    env=dict(os.environ,PATH=str(Path(sys.executable).parent)+':'+os.environ['PATH'],PYTHONUSERBASE=str(old/'python_user'),PYTHONPATH=os.pathsep.join(map(str,[src,repo/'tools',runtime/'tracking_runtime'])),TRACKING_RUNTIME_ROOT=str(runtime),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',YOLO_CONFIG_DIR=str(stage/'settings'))
    cmd=[sys.executable,'-u',str(src/'tools/run_botsort_plus.py'),'--baseline-root',str(repo/'results/vttiny/yolo11m_rgb_rgbt/rgbt'),'--config',str(src/'config.json'),'--runtime-root',str(runtime),'--features-dir',str(old/'features_rgb'),'--images-root',str(project/'data/raw/VT-Tiny-MOT'),'--output-dir',str(work),'--trace']
    with (logs/(name+'.log')).open('x') as out:subprocess.run(cmd,env=env,stdout=out,stderr=subprocess.STDOUT,check=True)
    print('finished', name,flush=True)
if __name__=='__main__':
    names=['native_only','bank_only','release_bank_history','release_bank_recovery','fixed_state']
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:list(pool.map(experiment,names))
    for part in ['trackers/B_ref','protocols/B_ref.json']:
        (work/part).symlink_to(official/part)
    for f in (official/'trackers/B_ref/data').glob('*.txt'):
        assert f.read_bytes()==(work/'trackers/fixed_state/data'/f.name).read_bytes()
    assert len(list((work/'trackers/fixed_state/data').glob('*.txt')))==30
    cmd=[sys.executable,str(repo/'tools/evaluate_botsort_plus.py'),'--run-dir',str(work),'--trackers','B_ref',*names,'--reference-tracker','B_ref','--gt-root',str(project/'data/official_gt/GT/GT_00/test'),'--trackeval-root',str(project/'external/TrackEval')]
    with (logs/'trackeval.log').open('x') as out:subprocess.run(cmd,cwd=repo,stdout=out,stderr=subprocess.STDOUT,check=True)
    print('COMPLETE',flush=True)
