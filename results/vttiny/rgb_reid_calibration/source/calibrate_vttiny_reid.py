"""Explicit TEST-set association-threshold calibration; preserve all prior runs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,time,json,shutil,traceback,copy,csv,zipfile
from pathlib import Path
from vttiny_tracking_common import ROOT,read,dump,sha
BASE=ROOT/'outputs/rgb_test_7trackers_v1';FT=ROOT/'outputs/rgb_test_4trackers_reid_testselect_v2';DIAG=ROOT/'outputs/reid_drop_diagnosis_v1';OUT=ROOT/'outputs/rgb_reid_calibration_test_v1'
GRIDS={
 'deepsort':('max_cos_dist',[.2,.3,.4,.5,.6,.7,.8]),
 'strongsort':('max_cos_dist',[.4,.5,.6,.7,.8,.9,1.0]),
 'deepocsort':('iou_threshold',[.05,.1,.15,.2,.25,.3,.4,.5]),
 'imprassoc':('match_thresh',[.45,.55,.65,.75,.85,.95])}
NAMES={'deepsort':'DeepSORT','strongsort':'StrongSORT (online)','deepocsort':'Deep OC-SORT','imprassoc':'ImprAsso','bytetrack':'ByteTrack','ocsort':'OC-SORT','botsort_no_reid':'BoT-SORT (no ReID)'}
def trial_name(method,t):return method+'_'+GRIDS[method][0]+'_'+format(t,'.2f').replace('.','p')
def make_plan():
 OUT.mkdir(exist_ok=False);p=read(FT/'protocol.json');cases=[]
 for method,(parameter,values) in GRIDS.items():
  for value in values:
   case=dict(name=trial_name(method,value),method=method,parameter=parameter,value=value,reused_from=None,created=time.time())
   if value==p['trackers'][method][parameter]:case['reused_from']=str(FT)
   elif method in ['deepsort','strongsort']:
    source=DIAG/'variants'/(method+'_t'+str(value).replace('.',''))
    if (source/'protocol.json').exists():case['reused_from']=str(source)
   cases.append(case)
 plan=dict(status='FROZEN_BEFORE_THIS_GRID_RUN',created=time.time(),evaluation_status='TEST_SET_SELECTED_AND_TEST_SET_THRESHOLD_TUNED_NOT_HELD_OUT',history='Search ranges informed by prior test-set diagnosis; not preregistered before observing test results.',data='Official RGB test, all30 sequences12590frames',selection_rule=['highest COMBINED_SEQ HOTA','then highest IDF1','then lowest IDSW','then nearest original threshold','then lowest threshold'],grid=GRIDS,cases=cases,unchanged=['detector checkpoint','detections','fine-tuned ReID checkpoint','embeddings','all other tracker parameters','official GT and evaluation'],caveats=['Deep OC-SORT tunes IoU acceptance gate, NOT a nonexistent cosine appearance gate. Embedding weight .75 stays fixed.','ImprAsso tunes active combined-cost match_thresh. appearance_thresh is unused in this implementation. lambda=.05, proximity=.1, second_match=.19 stay fixed; internal low-score cost scaling match_thresh/second_match_thresh follows native code.','This compares joint ReID adaptation + threshold calibration against original untuned baseline; it does not isolate a pure ReID-only gain.','No claims that grid maximum is global optimum. No unrecorded adaptive expansion.'])
 dump(OUT/'search_plan.json',plan)
 inputs={}
 for root in [BASE,FT]:
  for pattern in ['protocol.json','weights/*.pt','detections/*.json','embeddings/*.npz','embeddings/*.json','trackers/*/data/*.txt','metrics/summary.json']:
   for f in root.glob(pattern):inputs[str(f)]=sha(f)
 for seq in p['sequences']:
  f=ROOT/'data/processed_v1/mot/rgb/test'/seq/'gt/gt.txt';assert f.read_bytes()==(ROOT/'data/official_gt/GT/GT_00/test'/seq/'gt/gt.txt').read_bytes();inputs[str(f)]=sha(f)
 dump(OUT/'input_sha256.json',inputs)
 (OUT/'source').mkdir()
 for name in ['calibrate_vttiny_reid.py','run_vttiny_rgb_tracking.py','vttiny_reid_adapter.py','vttiny_sort_adapters.py','vttiny_tracking_common.py','evaluate_vttiny.py']:
  shutil.copy2(ROOT/'tools'/name,OUT/'source'/name)
 return plan
def trial(case):
 started=time.time();method=case['method'];out=OUT/'trials'/case['name'];out.mkdir(parents=True,exist_ok=True)
 p=read(FT/'protocol.json');expected=copy.deepcopy(p['trackers'][method]);expected[case['parameter']]=case['value']
 p['trackers'][method]=expected;p['calibration_trial']=case;p['evaluation_status']='TEST_SET_THRESHOLD_TUNED_NOT_HELD_OUT'
 # Replace inherited baseline notes which no longer describe this calibrated run.
 p['notes']=['Official test set used for ReID checkpoint selection and association-threshold calibration; not held-out evaluation.','No GT supplied to tracker prediction; same original RGB detections and same fine-tuned embeddings.',*read(OUT/'search_plan.json')['caveats']]
 dump(out/'protocol.json',p)
 source=Path(case['reused_from']) if case['reused_from'] else None
 if source:
  sourcep=read(source/'protocol.json');assert sourcep['trackers'][method]==expected
  assert sourcep['reid_sha256']==p['reid_sha256'] and sourcep['weight_sha256']==p['weight_sha256']
  # Copy trajectories into this run so delivery is self-contained.
  shutil.copytree(source/'trackers'/method/'data',out/'trackers'/method/'data',dirs_exist_ok=True)
  shutil.copytree(source/'timing'/method,out/'timing'/method,dirs_exist_ok=True)
 else:
  for directory in ['weights','detections','embeddings']:
   link=out/directory
   if not link.exists():link.symlink_to(FT/directory,target_is_directory=True)
  import run_vttiny_rgb_tracking as base
  base.OUT=out;reid,sorts=base.adapters()
  for n in reid.CONFIGS:reid.CONFIGS[n]=copy.deepcopy(p['trackers'][n])
  for n in sorts.CONFIGS:sorts.CONFIGS[n]=copy.deepcopy(p['trackers'][n])
  actual=(reid.CONFIGS if method in reid.CONFIGS else sorts.CONFIGS)[method]
  assert actual==expected
  base.track_method(method)
 link=OUT/'trackers'/case['name'];link.mkdir(parents=True,exist_ok=True)
 if not (link/'data').exists():(link/'data').symlink_to(out/'trackers'/method/'data',target_is_directory=True)
 files=list((out/'trackers'/method/'data').glob('*.txt'));assert len(files)==30
 for f in files:
  info=read(out/'timing'/method/(f.stem+'.json'));assert sha(f)==info['prediction_sha256']
 result=dict(**case,status='TRACKS_READY',elapsed_seconds=time.time()-started,finished=time.time(),trajectory_files=len(files),protocol_sha256=sha(out/'protocol.json'))
 dump(out/'trial_record.json',result);return result
def summarize(plan):
 metrics=read(OUT/'metrics/summary.json')['results'];pooled={r['tracker']:r for r in metrics if r['sequence']=='COMBINED_SEQ'}
 baseline={r['tracker']:r for r in read(BASE/'baseline_results.json')};ft={r['tracker']:r for r in read(FT/'metrics/summary.json')['results'] if r['sequence']=='COMBINED_SEQ'}
 p=read(FT/'protocol.json');allrows=[];selected={}
 for method in GRIDS:
  cases=[c for c in plan['cases'] if c['method']==method];original=p['trackers'][method][GRIDS[method][0]]
  ranked=sorted(cases,key=lambda c:(-pooled[c['name']]['HOTA'],-pooled[c['name']]['IDF1'],pooled[c['name']]['IDSW'],abs(c['value']-original),c['value']))
  best=ranked[0];selected[method]=dict(case=best,metrics=pooled[best['name']],original_threshold=original,at_grid_boundary=best['value'] in [min(GRIDS[method][1]),max(GRIDS[method][1])],tracker_config=read(OUT/'trials'/best['name']/'protocol.json')['trackers'][method])
  for case in cases:
   r=pooled[case['name']];allrows.append(dict(**case,metrics=r,selected=case['name']==best['name']))
  # Reused defaults should reproduce their already published numerical metrics.
  default=pooled[trial_name(method,original)]
  for key in ['HOTA','IDF1','MOTA','FP','FN','IDSW']:assert abs(default[key]-ft[method][key])<1e-8,(method,key)
 dump(OUT/'all_trials.json',allrows);dump(OUT/'selection.json',dict(evaluation_status=plan['evaluation_status'],selection_rule=plan['selection_rule'],selected=selected))
 for method,v in selected.items():
  shutil.copytree(OUT/'trials'/v['case']['name']/'trackers'/method/'data',OUT/'selected_trackers'/method/'data',dirs_exist_ok=True)
 finalp=copy.deepcopy(p);finalp['trackers']={m:v['tracker_config'] for m,v in selected.items()};finalp['evaluation_status']=plan['evaluation_status'];finalp['search_plan_sha256']=sha(OUT/'search_plan.json');finalp['selection_rule']=plan['selection_rule'];finalp['notes']=read(OUT/'trials'/next(iter(selected.values()))['case']['name']/'protocol.json')['notes'];dump(OUT/'selected_protocol.json',finalp)
 comparison=[]
 for method,v in selected.items():
  r=v['metrics'];b=baseline[method];f=ft[method]
  comparison.append(dict(method=method,parameter=v['case']['parameter'],old_threshold=v['original_threshold'],selected_threshold=v['case']['value'],baseline=b,finetuned_default=f,calibrated=r,delta_vs_baseline={k:r[k]-b[k] for k in ['HOTA','DetA','AssA','IDF1','MOTA','FP','FN','IDSW']},delta_vs_finetuned_default={k:r[k]-f[k] for k in ['HOTA','IDF1','MOTA','IDSW']}))
 dump(OUT/'comparison.json',comparison)
 lines=['# RGB 外观跟踪门限校准与原始基线对比','','**评估口径：官方测试集同时用于 ReID 权重选择、关联门限调参和最终汇报。不是独立留出测试。**','', '检测权重与原始基线一致；四种方法共享同一套微调 ReID 和同一份外观缓存。使用全部30段序列、12,590帧，含空GT序列。单个算法仅改变下列一项生效参数。','','## 选出的门限','','|方法|参数|原值|选中值|是否网格边界|','|---|---|---:|---:|---|']
 for method,v in selected.items():lines.append(f"|{NAMES[method]}|{v['case']['parameter']}|{v['original_threshold']}|{v['case']['value']}|{'是' if v['at_grid_boundary'] else '否'}|")
 lines+=['','Deep OC-SORT 校准的是实际 IoU 接受门限，不是余弦距离门限；外观权重保持0.75。ImprAsso 的 appearance_thresh 在当前实现未生效，校准实际组合代价门限 match_thresh，其内部低分支缩放随原代码联动。','','## 三阶段结果','','|方法|配置|HOTA↑|IDF1↑|MOTA↑|IDSW↓|FP↓|FN↓|','|---|---|---:|---:|---:|---:|---:|---:|']
 for row in comparison:
  for label,key in [('原始预训练基线','baseline'),('微调ReID＋原门限','finetuned_default'),('微调ReID＋校准门限','calibrated')]:
   r=row[key];lines.append('|'+ '|'.join([NAMES[row['method']],label]+[f'{r[k]:.2f}' for k in ['HOTA','IDF1','MOTA']]+[str(r[k]) for k in ['IDSW','FP','FN']])+'|')
 lines+=['','## 相对原始基线变化','','|方法|ΔHOTA|ΔIDF1|ΔMOTA|ΔIDSW|','|---|---:|---:|---:|---:|']
 for row in comparison:
  d=row['delta_vs_baseline'];lines.append('|'+ '|'.join([NAMES[row['method']]]+[f'{d[k]:+.2f}' for k in ['HOTA','IDF1','MOTA']]+[f"{d['IDSW']:+d}"])+'|')
 lines+=['','## 完整调参记录','','搜索网格根据前次测试集诊断设定，在本轮运行前冻结。主指标为汇总HOTA；依次以IDF1、较少IDSW、更接近原门限、较小门限打破并列。未根据单个序列挑选参数，不混用不同权重。所有候选均使用完整测试集。','',f"共{len(plan['cases'])}组配置，其中{sum(c['reused_from'] is not None for c in plan['cases'])}组复用已核验的相同配置轨迹，其余新运行。见 search_plan.json、trial_ledger.jsonl、all_trials.json 与 metrics/summary.json。网格内最优不代表全局最优；比较的收益包含ReID适配与测试集门限调参两部分。",'', '|方法|参数值|HOTA|IDF1|MOTA|IDSW|选中|来源|','|---|---:|---:|---:|---:|---:|---|---|']
 for row in allrows:
  r=row['metrics'];lines.append('|'+ '|'.join([NAMES[row['method']],str(row['value'])]+[f'{r[k]:.2f}' for k in ['HOTA','IDF1','MOTA']]+[str(r['IDSW']),'是' if row['selected'] else '', '复用' if row['reused_from'] else '新运行'])+'|')
 (OUT/'comparison.md').write_text('\n'.join(lines),encoding='utf8')
 combined=[]
 for method,b in baseline.items():
  row=copy.deepcopy(selected[method]['metrics'] if method in selected else b);row['tracker']=method;row['configuration']='fine-tuned ReID + test-calibrated threshold' if method in selected else 'unchanged original baseline';combined.append(row)
 dump(OUT/'seven_tracker_summary.json',dict(evaluation_status='MIXED: four test-calibrated appearance methods; other three original unchanged',results=combined))
 all_lines=['# 七种跟踪算法汇总','','四种外观方法使用测试集选模与门限校准；其余三种保持原始配置。应标注此调参差异，不视为对所有算法进行了等预算调参。','','|方法|配置|HOTA|IDF1|MOTA|IDSW|','|---|---|---:|---:|---:|---:|']
 for r in sorted(combined,key=lambda x:-x['HOTA']):all_lines.append('|'+ '|'.join([NAMES[r['tracker']],'微调＋测试集校准' if r['tracker'] in selected else '原始配置']+[f'{r[k]:.2f}' for k in ['HOTA','IDF1','MOTA']]+[str(r['IDSW'])])+'|')
 (OUT/'seven_tracker_summary.md').write_text('\n'.join(all_lines),encoding='utf8')
 return comparison
def audit_and_package():
 import numpy as np
 hashes=read(OUT/'input_sha256.json');assert all(sha(k)==v for k,v in hashes.items())
 p=read(OUT/'selected_protocol.json');checked=0
 for method in GRIDS:
  for seq,frames in p['sequences'].items():
   file=OUT/'selected_trackers'/method/'data'/(seq+'.txt');seen=set()
   for row in csv.reader(file.open()):
    assert len(row)==10;x=list(map(float,row));f,tid=x[:2];assert np.isfinite(x).all() and f==int(f) and 1<=f<=len(frames) and tid==int(tid) and tid>0 and x[4]>0 and x[5]>0 and (f,tid) not in seen;seen.add((f,tid))
   checked+=1
 dump(OUT/'integrity.json',dict(status='PASS',inputs_unchanged=True,selected_trajectory_files=checked,frames_per_method=12590,original_and_finetuned_baselines_preserved=True))
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 rows=read(OUT/'all_trials.json');fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
 for ax,method in zip(axes.flat,GRIDS):
  selected=[r for r in rows if r['method']==method];ax.plot([r['value'] for r in selected],[r['metrics']['HOTA'] for r in selected],marker='o',label='HOTA');ax.plot([r['value'] for r in selected],[r['metrics']['IDF1'] for r in selected],marker='s',label='IDF1');winner=next(r for r in selected if r['selected']);ax.axvline(winner['value'],ls='--',color='gray');ax.set(title=NAMES[method],xlabel=GRIDS[method][0],ylabel='%');ax.grid(alpha=.2);ax.legend()
 fig.suptitle('Official TEST-set calibration: same fine-tuned ReID and detections');fig.savefig(OUT/'calibration_curves.png',dpi=170);plt.close(fig)
 archive=OUT/'rgb_reid_calibration_results.zip'
 include=['search_plan.json','trial_ledger.jsonl','all_trials.json','selection.json','comparison.json','comparison.md','selected_protocol.json','seven_tracker_summary.json','seven_tracker_summary.md','input_sha256.json','integrity.json','calibration_curves.png']
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
  for name in include:z.write(OUT/name,name)
  for directory in ['selected_trackers','metrics','source']:
   for f in (OUT/directory).rglob('*'):
    if f.is_file():z.write(f,f.relative_to(OUT))
  for f in (OUT/'trials').glob('*/trial_record.json'):z.write(f,f.relative_to(OUT))
  for f in (OUT/'trials').glob('*/protocol.json'):z.write(f,f.relative_to(OUT))
 dump(OUT/'delivery_manifest.json',dict(archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size))
def main():
 plan=make_plan();dump(OUT/'state.json',dict(status='RUNNING',stage='tracking',total_trials=len(plan['cases']),completed=0,started=time.time()))
 from concurrent.futures import ProcessPoolExecutor,as_completed
 import multiprocessing
 with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
  futures=[pool.submit(trial,c) for c in plan['cases']]
  for completed,f in enumerate(as_completed(futures),1):
   record=f.result()
   with (OUT/'trial_ledger.jsonl').open('a') as stream:stream.write(json.dumps(record,ensure_ascii=False)+'\n')
   dump(OUT/'state.json',dict(status='RUNNING',stage='tracking',total_trials=len(futures),completed=completed,updated=time.time()));print('COMPLETE_TRIAL',record['name'],completed,flush=True)
 dump(OUT/'state.json',dict(status='RUNNING',stage='evaluation',updated=time.time()))
 from evaluate_vttiny import evaluate
 evaluate('rgb','test',OUT/'trackers',[c['name'] for c in plan['cases']],OUT/'metrics')
 result=summarize(plan);audit_and_package();dump(OUT/'state.json',dict(status='COMPLETE',total_trials=len(plan['cases']),updated=time.time()));print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  dump(OUT/'state.json',dict(status='FAILED',traceback=traceback.format_exc(),updated=time.time()));raise
