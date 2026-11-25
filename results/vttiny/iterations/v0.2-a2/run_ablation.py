from pathlib import Path
import os,sys,json,subprocess,shutil,hashlib
r=Path(os.environ['VTTINY_ROOT']).resolve()
old=Path(os.environ['TRACKING_RUNTIME_ROOT']).resolve()
s1=Path(os.environ.get('BOTSORT_PLUS_V01_STAGE',r/'integrations/botsort_plus_v0_1_0'))
features=Path(os.environ.get('BOTSORT_PLUS_FEATURES',s1/'features_rgb'))
orig=s1/'BOT-SORT-Plus'
repo=Path(__file__).resolve().parents[4]
s=Path(os.environ.get('BOTSORT_PLUS_STAGE',repo/'outputs/a2_replay')).resolve()
(s/'logs').mkdir(parents=True,exist_ok=True)
exp=s/'experiments'
py=sys.executable
env=dict(os.environ,PYTHONUSERBASE=str(s1/'python_user'),PYTHONPATH=os.pathsep.join(map(str,[repo,repo/'tools',old/'tracking_runtime'])),TRACKING_RUNTIME_ROOT=str(old),TRACKEVAL_ROOT=str(r/'external/TrackEval'),VTTINY_TEST_TRACKEVAL_ROOT=str(r/'external/TrackEval'),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',YOLO_CONFIG_DIR=str(s/'settings'),BOTSORT_PLUS_REPO=str(repo),BOTSORT_PLUS_FEATURES=str(features),VTTINY_FRAME_ROOT=str(r/'data/raw/VT-Tiny-MOT'),VTTINY_GT_FOLDER=str(r/'data/official_gt/GT/GT_00/test'))
for key in ('VTTINY_FRAME_ROOT','VTTINY_GT_FOLDER','TRACKEVAL_ROOT'):
 if key in os.environ: env[key]=os.environ[key]
env['VTTINY_TEST_TRACKEVAL_ROOT']=env['TRACKEVAL_ROOT']
env['PATH']=str(Path(py).parent)+os.pathsep+env['PATH']
import concurrent.futures,traceback,time
seq3=['DJI_0030_2','DJI_0129_2','DJI_0327_3']
ids=[12,13,25,26]
cfg={int(f.name[:2]):f for f in (repo/'configs/botsort_plus').glob('*.json') if f.name[:2].isdigit()}
nm={k:json.loads(f.read_text())['name'] for k,f in cfg.items()}
def call(args,log):
 with (s/'logs'/(log+'.log')).open('w') as f:subprocess.run(list(map(str,[py,*args])),cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
def one(i,phase):
 root=exp/phase
 call([repo/'tools/run_botsort_plus.py','--config',cfg[i],'--runtime-root',old,'--features-dir',features,'--images-root',env['VTTINY_FRAME_ROOT'],'--output-dir',root,'--trace',*(['--sequences',*seq3] if phase=='three_sequences' else [])],phase+'_'+nm[i])
 print('done',phase,nm[i],flush=True)
def batch(nums,phase):
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  list(pool.map(lambda i: one(i,phase),nums))
def audit(phase,nums):
 root=exp/phase
 call([repo/'tools/audit_botsort_plus_a2.py','--run-dir',root,'--trackers',*[nm[i] for i in nums],'--compare-pair','baseline_ref','A2_shadow','--compare-pair','B_ref','A2B_shadow','--output',root/'structure_audit.json'],phase+'_audit')
 hashes={}
 for a,b in [(12,25),(13,26)]:
  A=root/'trackers'/nm[a]/'data';B=root/'trackers'/nm[b]/'data'
  files=sorted(A.glob('*.txt'))
  assert {f.name for f in files}=={f.name for f in B.glob('*.txt')}
  assert len(files)==(3 if phase=='three_sequences' else 30)
  H={}
  for f in files:
   raw=f.read_bytes(); assert raw==(B/f.name).read_bytes(); H[f.stem]=hashlib.sha256(raw).hexdigest()
  hashes[nm[a]+'='+nm[b]]=H
 (root/'paired_bytes.json').write_text(json.dumps(hashes,indent=2))
 print('paired byte equality',phase,flush=True)
if __name__=='__main__':
 batch(ids,'three_sequences')
 audit('three_sequences',ids)
 batch([12,13,25,26,21,17,18,19,20,14,15,16,22,23,24],'full30')
 root=exp/'full30'
 call([repo/'tools/evaluate_botsort_plus.py','--run-dir',root,'--trackers',*[nm[i] for i in range(12,27)],'--gt-root',env['VTTINY_GT_FOLDER'],'--trackeval-root',env['TRACKEVAL_ROOT'],'--reference-tracker','B_ref'],'full30_evaluation')
 audit('full30',list(range(12,27)))
 print('COMPLETE',flush=True)