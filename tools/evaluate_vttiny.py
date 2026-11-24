"""Class-agnostic VT-Tiny-MOT evaluation, with explicit RGB/IR coordinate references."""
import argparse, hashlib, json, shutil, subprocess, sys
from pathlib import Path
import numpy as np
from vttiny_tracking_common import ROOT, OLD as RUNTIME, TRACK_EVAL, dump

OLD=RUNTIME/'external/TrackEval'
VENDOR=TRACK_EVAL
if not (VENDOR/'trackeval').exists():
 if not (OLD/'trackeval').is_dir():
  raise FileNotFoundError(f'TrackEval is required at {VENDOR}. Set TRACKEVAL_ROOT or see README.md.')
 shutil.copytree(OLD/'trackeval',VENDOR/'trackeval',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 for name in ['LICENSE','README.md']:
  if (OLD/name).exists():shutil.copyfile(OLD/name,VENDOR/name)
 dump(VENDOR/'source.json',{'repository':'https://github.com/JonathonLuiten/TrackEval','commit':subprocess.check_output(['git','-C',str(OLD),'rev-parse','HEAD'],text=True).strip(),'package_sha256':{str(p.relative_to(VENDOR)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (VENDOR/'trackeval').rglob('*.py')}})
sys.path.insert(0,str(VENDOR))
for name,value in [('int',int),('float',float),('bool',bool)]:
 if name not in np.__dict__:setattr(np,name,value)
import trackeval

class VTTinyObjects(trackeval.datasets.MotChallenge2DBox):
 def __init__(self,config):
  assert config.get('DO_PREPROC') is False
  super().__init__(dict(config,CLASSES_TO_EVAL=['pedestrian']))
  self.valid_classes=['object'];self.class_list=['object'];self.class_name_to_class_id['object']=1;self.config['CLASSES_TO_EVAL']=['object']

def evaluate(modality,split,folder,trackers,output,seqs=None,gt_folder=None):
 gt=gt_folder if gt_folder is not None else ROOT/'data/processed_v1/mot'/modality/split
 config={'GT_FOLDER':str(gt),'TRACKERS_FOLDER':str(folder),'OUTPUT_FOLDER':str(output),'TRACKERS_TO_EVAL':trackers,'BENCHMARK':'VT-Tiny-MOT','SPLIT_TO_EVAL':split,'SKIP_SPLIT_FOL':True,'SEQMAP_FILE':str(ROOT/'data/processed_v1/mot/seqmaps'/f'{modality}_{split}.txt'),'DO_PREPROC':False,'PRINT_CONFIG':False}
 if seqs:config['SEQ_INFO']=seqs if isinstance(seqs,dict) else {s:None for s in seqs}
 dataset=VTTinyObjects(config)
 evaluator=trackeval.Evaluator({'PRINT_CONFIG':False,'PRINT_RESULTS':False,'TIME_PROGRESS':False,'PLOT_CURVES':False,'BREAK_ON_ERROR':True,'LOG_ON_ERROR':str(output/'error.log')})
 results,messages=evaluator.evaluate([dataset],[trackeval.metrics.HOTA(),trackeval.metrics.CLEAR(),trackeval.metrics.Identity()])
 rows=[]
 for tracker,values in results['VTTinyObjects'].items():
  for seq,v in values.items():
   r=v['object'];c=r['CLEAR'];h=r['HOTA'];i=r['Identity']
   rows.append({'tracker':tracker,'sequence':seq,'HOTA':float(np.mean(h['HOTA'])*100),'DetA':float(np.mean(h['DetA'])*100),'AssA':float(np.mean(h['AssA'])*100),'IDF1':float(i['IDF1']*100),'MOTA':float(c['MOTA']*100),'FP':int(c['CLR_FP']),'FN':int(c['CLR_FN']),'IDSW':int(c['IDSW'])})
 payload={'status':'PASS','modality_reference':modality,'split':split,'evaluation':'all object classes pooled, native modality coordinates, official test GT unmodified','aggregation':'TrackEval COMBINED_SEQ, not unweighted averaging of sequence percentages','results':rows}
 dump(output/'summary.json',payload);return payload

def self_test():
 out=ROOT/'outputs/evaluation_self_test';seqs=['DJI_0030_2','DJI_0073_3','DJI_0325_2'];results={}
 for mode in ['rgb','ir']:
  for index,seq in enumerate(seqs):
   raw=(ROOT/'data/processed_v1/mot'/mode/'test'/seq/'gt/gt.txt').read_text();gt=[line.split(',') for line in raw.splitlines() if line.strip()]
   pred=[','.join(r[:6]+['1','-1','-1','-1'])+'\n' for r in gt]
   for name in ['perfect_gt','one_miss_one_fp']:
    rows=list(pred)
    if name=='one_miss_one_fp':
     if index==0:rows=rows[1:]
     if index==2:rows.append('1,999999,-1000,-1000,10,10,1,-1,-1,-1\n')
    p=out/mode/'trackers'/name/'data'/(seq+'.txt');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(rows))
  result=evaluate(mode,'test',out/mode/'trackers',['perfect_gt','one_miss_one_fp'],out/mode/'metrics',seqs)
  pooled={r['tracker']:r for r in result['results'] if r['sequence']=='COMBINED_SEQ'};perfect=pooled['perfect_gt'];negative=pooled['one_miss_one_fp']
  assert all(abs(perfect[k]-100)<1e-8 for k in ['HOTA','MOTA','IDF1']) and perfect['FP']==perfect['FN']==perfect['IDSW']==0
  assert negative['FP']==negative['FN']==1 and negative['IDSW']==0 and negative['HOTA']<100 and negative['IDF1']<100
  results[mode]=pooled
 dump(ROOT/'reports/evaluation_self_test.json',{'status':'PASS','purpose':'SOFTWARE_DIAGNOSTIC_ONLY_NOT_MODEL_RESULTS','sequences':seqs,'checks':'Perfect GT yields 100; one removed detection + one unmatched false positive yields FN=1, FP=1. Includes empty RGB GT and the official-vs-COCO discrepancy sequence.','results':results})
 print(json.dumps(results),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');p.add_argument('--modality',choices=['rgb','ir']);p.add_argument('--split',choices=['val','test'],default='test');p.add_argument('--trackers-folder',type=Path);p.add_argument('--trackers',nargs='+');p.add_argument('--output',type=Path);args=p.parse_args()
 if args.self_test:self_test()
 else:
  assert args.modality and args.trackers_folder and args.trackers and args.output
  evaluate(args.modality,args.split,args.trackers_folder,args.trackers,args.output)
