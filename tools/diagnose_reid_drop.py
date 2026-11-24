"""Post-hoc diagnosis only: feature distances and fixed-threshold interventions."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,csv,collections,time,argparse
from pathlib import Path
from vttiny_tracking_common import ROOT,read,dump,sha
import numpy as np
from scipy.optimize import linear_sum_assignment
BASE=ROOT/'outputs/rgb_test_7trackers_v1';FT=ROOT/'outputs/rgb_test_4trackers_reid_testselect_v2';OUT=ROOT/'outputs/reid_drop_diagnosis_v1'
def ious(a,b):
 lt=np.maximum(a[:,None,:2],b[None,:,:2]);rb=np.minimum(a[:,None,2:],b[None,:,2:]);intersection=np.maximum(rb-lt,0).prod(-1)
 aa=np.maximum(a[:,2:]-a[:,:2],0).prod(-1);bb=np.maximum(b[:,2:]-b[:,:2],0).prod(-1)
 return intersection/np.maximum(aa[:,None]+bb[None,:]-intersection,1e-9)
def describe(x):
 x=np.asarray(x);return dict(count=len(x),mean=float(x.mean()),quantiles=dict(zip(['p10','p50','p90','p95'],np.percentile(x,[10,50,90,95]).tolist())),exceeds={str(t):float(np.mean(x>t)*100) for t in [.2,.4,.6,.8,1.]}) if len(x) else dict(count=0)
def distances():
 OUT.mkdir(exist_ok=True);p=read(BASE/'protocol.json');values=collections.defaultdict(list);seqstats=[];samefeature=[]
 for seq in p['sequences']:
  assert sha(BASE/'detections'/(seq+'.json'))==sha(FT/'detections'/(seq+'.json'))
  cache=read(BASE/'detections'/(seq+'.json'));gt=collections.defaultdict(list)
  for line in (ROOT/'data/processed_v1/mot/rgb/test'/seq/'gt/gt.txt').read_text().splitlines():
   r=list(map(float,line.split(',')));gt[int(r[0])].append([int(r[1]),r[2],r[3],r[2]+r[4],r[3]+r[5]])
  with np.load(BASE/'embeddings'/(seq+'.npz')) as z:old={k:z[k] for k in z.files}
  with np.load(FT/'embeddings'/(seq+'.npz')) as z:new={k:z[k] for k in z.files}
  for k in ['indices','offsets','frame_ids']:assert np.array_equal(old[k],new[k])
  assert np.allclose(np.linalg.norm(old['features'],axis=1),1,atol=1e-4) and np.allclose(np.linalg.norm(new['features'],axis=1),1,atol=1e-4)
  samefeature.extend(np.einsum('ij,ij->i',old['features'],new['features']).tolist())
  history={};nmatched=0
  for index,frame in enumerate(cache['frames']):
   f=frame['frame'];a,b=old['offsets'][index:index+2];ids=old['indices'][a:b].astype(int);d=np.asarray(frame['detections'],np.float32).reshape(-1,6)[ids]
   if not len(d) or not gt[f]:continue
   truth=np.asarray(gt[f]);overlap=ious(d[:,:4],truth[:,1:]);ii,jj=linear_sum_assignment(-overlap)
   matched=[(i,j) for i,j in zip(ii,jj) if overlap[i,j]>=.5];nmatched+=len(matched);updates={}
   for i,j in matched:
    identity=int(truth[j,0]);ov=old['features'][a+i];nv=new['features'][a+i]
    if identity in history and 1<=f-history[identity][0]<=30:
     prev=history[identity];gap=f-prev[0];bucket='gap1' if gap==1 else ('gap2_5' if gap<=5 else 'gap6_30')
     for name,v,h in [('pretrained',ov,prev[1]),('finetuned',nv,prev[2])]:
      distance=float(1-v@h);values[name+'/same_id/all'].append(distance);values[name+'/same_id/'+bucket].append(distance)
     candidates=[h for tid,h in history.items() if tid!=identity and 1<=f-h[0]<=30]
     if candidates:
      for name,v,k in [('pretrained',ov,1),('finetuned',nv,2)]:
       dd=1-np.asarray([h[k] for h in candidates])@v
       values[name+'/nearest_other_id'].append(float(dd.min()))
    updates[identity]=(f,ov,nv)
   history.update(updates);history={tid:h for tid,h in history.items() if f-h[0]<=30}
  seqstats.append(dict(sequence=seq,matched_high_confidence_detections=nmatched));print('DISTANCE',seq,nmatched,flush=True)
 dump(OUT/'feature_distance_diagnosis.json',dict(purpose='POST_HOC_DIAGNOSTIC_NOT_FORMAL_TUNED_BASELINE',method='Same >0.5 detection caches; one-to-one Hungarian GT association IoU>=0.5; compare last observed same-GT feature within30frames; nearest other-ID feature in history ignores tracker motion gate and is only a proxy.',cache_alignment='PASS',unit_norm='PASS',statistics={k:describe(v) for k,v in values.items()},same_box_old_new_cosine_similarity=describe(samefeature),sequences=seqstats))
def variant(arg):
 method,threshold=arg;name=method+'_t'+str(threshold).replace('.','');out=OUT/'variants'/name
 import run_vttiny_rgb_tracking as base
 out.mkdir(parents=True,exist_ok=False)
 for directory in ['weights','detections','embeddings']:(out/directory).symlink_to(FT/directory,target_is_directory=True)
 p=read(FT/'protocol.json');p['purpose']='POST_HOC_THRESHOLD_INTERVENTION_DIAGNOSTIC';p['trackers'][method]['max_cos_dist']=threshold;dump(out/'protocol.json',p)
 base.OUT=out;_,sorts=base.adapters();sorts.CONFIGS[method]['max_cos_dist']=threshold
 base.track_method(method)
 link=OUT/'trackers'/name;link.mkdir(parents=True);(link/'data').symlink_to(out/'trackers'/method/'data',target_is_directory=True)
 return name
def interventions():
 OUT.mkdir(exist_ok=True)
 from concurrent.futures import ProcessPoolExecutor
 import multiprocessing
 cases=[('deepsort',t) for t in [.4,.6,.8]]+[('strongsort',t) for t in [.6,.8,1.]]
 dump(OUT/'intervention_protocol.json',dict(purpose='Post-hoc causal diagnostic, not independent benchmark or replacement baseline',cases=cases,only_changed_parameter='max_cos_dist',features=str(FT/'embeddings'),detections=str(FT/'detections')))
 with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:names=list(pool.map(variant,cases))
 from evaluate_vttiny import evaluate
 payload=evaluate('rgb','test',OUT/'trackers',names,OUT/'metrics')
 rows=[r for r in payload['results'] if r['sequence']=='COMBINED_SEQ'];dump(OUT/'intervention_summary.json',rows);print(json.dumps(rows),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['distances','interventions']);args=ap.parse_args();globals()[args.stage]()
