"""Frozen RGB detection -> seven trackers -> official pooled VT-Tiny-MOT evaluation."""
import os,sys,time,json,csv,shutil,subprocess,traceback,argparse
from pathlib import Path
from vttiny_tracking_common import ROOT,OLD,sha,read,dump,forbid_gt
sys.path[:0]=[str(ROOT/'formal_detection_v1/framework'),str(OLD/'tracking_runtime'),str(OLD/'tracking_extensions')]
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'formal_detection_v1/settings'))
os.environ.setdefault('CUDA_DEVICE_ORDER','PCI_BUS_ID')
os.environ.setdefault('CUDA_VISIBLE_DEVICES','1')
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import cv2
cv2.setNumThreads(1)
OUT=ROOT/'outputs/rgb_test_7trackers_v1'
BASE={
 'bytetrack':dict(det_thresh=.5,track_thresh=.5,min_conf=.1,match_thresh=.8,track_buffer=60,frame_rate=15),
 'ocsort':dict(det_thresh=.5,min_conf=.1,max_age=30,min_hits=3,iou_threshold=.3,use_byte=False),
 'botsort_no_reid':dict(det_thresh=.5,track_high_thresh=.5,track_low_thresh=.1,new_track_thresh=.6,match_thresh=.8,track_buffer=60,frame_rate=15,with_reid=False,cmc_method='sof')}
NAMES={'bytetrack':'ByteTrack','ocsort':'OC-SORT','botsort_no_reid':'BoT-SORT (no ReID)','deepocsort':'Deep OC-SORT','imprassoc':'ImprAsso','deepsort':'DeepSORT','strongsort':'StrongSORT (online)'}
def adapters():
 import vttiny_reid_adapter as reid
 import vttiny_sort_adapters as sorts
 reid.WEIGHTS=OUT/'weights/clip_veri.pt';sorts.WEIGHTS=reid.WEIGHTS
 reid.CONFIGS['imprassoc'].update(track_buffer=60,frame_rate=15)
 return reid,sorts
def construct(name):
 import torch
 torch.set_num_threads(1)
 if name in BASE:
  from boxmot import ByteTrack,OcSort,BotSort
  if name=='botsort_no_reid':return BotSort(reid_weights=Path('unused.pt'),device=torch.device('cpu'),half=False,**BASE[name])
  return {'bytetrack':ByteTrack,'ocsort':OcSort}[name](**BASE[name])
 reid,sorts=adapters()
 return (reid if name in reid.CONFIGS else sorts).tracker(name)
def init():
 OUT.mkdir(parents=True,exist_ok=False)
 (OUT/'weights').mkdir()
 source=ROOT/'formal_detection_v1/runs/rgb/weights/best.pt'
 state=read(ROOT/'formal_detection_v1/state/formal_rgb.json')
 assert state['status']=='COMPLETE' and state['epoch']==200
 shutil.copy2(source,OUT/'weights/rgb_best.pt')
 shutil.copy2(OLD/'reid_weights/clip_veri.pt',OUT/'weights/clip_veri.pt')
 seqs={}
 for line in (ROOT/'data/processed_v1/test_pairs.jsonl').read_text().splitlines():
  r=json.loads(line);seqs.setdefault(r['sequence'],[]).append({'frame':r['frame'],'image':r['rgb']})
 assert len(seqs)==30 and sum(map(len,seqs.values()))==12590
 for frames in seqs.values():assert [f['frame'] for f in frames]==list(range(1,len(frames)+1))
 reid,sorts=adapters()
 p=dict(dataset='VT-Tiny-MOT',split='official test',modality='RGB',frame_rate=15,frames=12590,sequences=seqs,
  source_weight=str(source),weight_sha256=sha(source),reid_sha256=sha(OUT/'weights/clip_veri.pt'),training_state=state,
  detector=dict(imgsz=640,conf=.05,iou=.7,max_det=300,half=False,batch=16,agnostic_nms=False,rect=True),
  trackers={**BASE,**reid.CONFIGS,**sorts.CONFIGS},
  notes=['All seven detector classes retained in cache; association pools them as object class 0; no additional cross-class NMS.',
   'Official RGB GT unmodified, including empty-GT DJI_0325_2. All 30 sequences included in TrackEval COMBINED_SEQ.',
   'No GT or first-frame boxes used for tracking; no test-set parameter tuning.',
   'Existing tracker thresholds retained. Native 15 FPS and buffer60 give 30 lost frames, preserving previous max-lost frame count.',
   'CLIP-VeRi vehicle-pretrained ReID is shared and NOT fine-tuned on VT-Tiny-MOT seven-class aerial objects.',
   'BoT-SORT uses no ReID. StrongSORT is online without AFLink/GSI. These are explicit baseline variants, not exact paper reproductions.'])
 dump(OUT/'protocol.json',p)
 shutil.copytree(OLD/'tracking_extensions/provenance',OUT/'implementation_provenance')
 for name in ['run_vttiny_rgb_tracking.py','vttiny_tracking_common.py','vttiny_reid_adapter.py','vttiny_sort_adapters.py','evaluate_vttiny.py']:
  d=OUT/'source'/name;d.parent.mkdir(exist_ok=True);shutil.copy2(ROOT/'tools'/name,d)
 print('INIT',OUT,flush=True)
def smoke():
 sys.addaudithook(forbid_gt)
 rng=np.random.default_rng(0);img=rng.integers(0,255,(512,640,3),dtype=np.uint8)
 for name in NAMES:
  tr=construct(name);observed=[]
  for frame in range(18):
   empty=frame in (0,8,9)
   det=np.empty((0,6),np.float32) if empty else np.array([[10,10,20,20,.2,0],[50+frame,50,90+frame,90,.95,0]],np.float32)
   emb=np.zeros((len(det),1280),np.float32)
   if len(det):emb[1,0]=1
   pred=tr.update(det,img,**({'embs':emb} if name not in BASE else {}))
   assert np.isfinite(pred).all()
   if len(pred):observed.extend(pred[:,4].tolist())
  assert observed,name
  print('SMOKE PASS',name,flush=True)
 dump(OUT/'smoke.json',{'status':'PASS','checks':'All seven: leading empty frame, low/high confidence mixed feature alignment, missed frames and reappearance, finite outputs.'})
def detect():
 sys.addaudithook(forbid_gt)
 import torch
 torch.set_num_threads(4)
 from ultralytics import YOLO
 p=read(OUT/'protocol.json');assert sha(OUT/'weights/rgb_best.pt')==p['weight_sha256']
 model=YOLO(str(OUT/'weights/rgb_best.pt'))
 for seq,frames in p['sequences'].items():
  dest=OUT/'detections'/(seq+'.json')
  if dest.exists():continue
  records=[];seconds=0
  for start in range(0,len(frames),16):
   batch=frames[start:start+16];t=time.perf_counter()
   results=model.predict(source=[f['image'] for f in batch],device='1',verbose=False,save=False,**p['detector'])
   torch.cuda.synchronize();seconds+=time.perf_counter()-t
   assert len(results)==len(batch)
   for frame,res in zip(batch,results):
    assert res.orig_shape==(512,640) and Path(res.path).resolve()==Path(frame['image']).resolve()
    b=res.boxes;d=torch.cat([b.xyxy,b.conf[:,None],b.cls[:,None]],1).cpu().numpy()
    assert np.isfinite(d).all() and np.isin(d[:,5],np.arange(7)).all()
    assert ((d[:,2:4]-d[:,:2])>0).all()
    records.append({**frame,'detections':d.tolist()})
  dump(dest,dict(model_sha256=p['weight_sha256'],inference_wall_seconds=seconds,frames=records))
  print('DETECT',seq,len(records),'seconds',round(seconds,2),flush=True)
def embeddings():
 sys.addaudithook(forbid_gt)
 import torch
 torch.cuda.init();torch.set_num_threads(4)
 reid,_=adapters();backend,metadata=reid.encoder()
 metadata['training_domain']='VeRi visible-light vehicles; not adapted to VT-Tiny-MOT aerial seven-class targets'
 dump(OUT/'reid_encoder.json',metadata)
 p=read(OUT/'protocol.json');assert metadata['sha256']==p['reid_sha256']
 for seq in p['sequences']:
  dest=OUT/'embeddings'/(seq+'.npz')
  if dest.exists() and dest.with_suffix('.json').exists():continue
  cachepath=OUT/'detections'/(seq+'.json');cache=read(cachepath)
  features=[];indices=[];offsets=[0];seconds=0;wall=time.perf_counter()
  for frame in cache['frames']:
   det=np.asarray(frame['detections'],np.float32).reshape(-1,6);idx=np.flatnonzero(det[:,4]>.5)
   img=cv2.imread(frame['image']);assert img is not None
   if len(idx):
    t=time.perf_counter();v=backend.get_features(det[idx,:4],img);torch.cuda.synchronize();seconds+=time.perf_counter()-t
    assert np.isfinite(v).all() and np.allclose(np.linalg.norm(v,axis=1),1,atol=1e-4)
    features.append(v);indices.extend(idx.tolist())
   offsets.append(offsets[-1]+len(idx))
  dest.parent.mkdir(exist_ok=True)
  np.savez_compressed(dest,features=np.concatenate(features) if features else np.empty((0,1280),np.float32),indices=np.array(indices),offsets=np.array(offsets),frame_ids=np.array([f['frame'] for f in cache['frames']]))
  dump(dest.with_suffix('.json'),dict(detection_sha256=sha(cachepath),features_sha256=sha(dest),reid_sha256=metadata['sha256'],encoder_seconds=seconds,wall_seconds=time.perf_counter()-wall,high_detections=len(indices)))
  print('EMBED',seq,len(indices),'seconds',round(seconds,2),flush=True)
def track_method(name):
 sys.addaudithook(forbid_gt);p=read(OUT/'protocol.json');timings=[]
 for seq,expected in p['sequences'].items():
  dest=OUT/'trackers'/name/'data'/(seq+'.txt');timing=OUT/'timing'/name/(seq+'.json')
  if dest.exists() and timing.exists():timings.append(read(timing));continue
  tr=construct(name);cachepath=OUT/'detections'/(seq+'.json');cache=read(cachepath)
  assert cache['model_sha256']==p['weight_sha256'] and len(cache['frames'])==len(expected)
  emb=None;meta={}
  if name not in BASE:
   ep=OUT/'embeddings'/(seq+'.npz');meta=read(ep.with_suffix('.json'))
   assert meta['detection_sha256']==sha(cachepath) and meta['features_sha256']==sha(ep) and meta['reid_sha256']==p['reid_sha256']
   with np.load(ep) as z:emb={k:z[k] for k in z.files}
   assert list(emb['frame_ids'])==[f['frame'] for f in cache['frames']]
  seconds=0;rows=0;wall=time.perf_counter();dest.parent.mkdir(parents=True,exist_ok=True)
  with dest.with_suffix('.tmp').open('w') as stream:
   writer=csv.writer(stream,lineterminator='\n')
   for i,frame in enumerate(cache['frames']):
    img=cv2.imread(frame['image']);assert img is not None
    det=np.asarray(frame['detections'],np.float32).reshape(-1,6);det[:,5]=0
    kw={}
    if emb is not None:
     a,b=emb['offsets'][i:i+2];idx=emb['indices'][a:b].astype(int)
     assert np.array_equal(idx,np.flatnonzero(det[:,4]>.5))
     v=np.zeros((len(det),1280),np.float32);v[idx]=emb['features'][a:b];kw['embs']=v
    t=time.perf_counter();pred=tr.update(det,img,**kw);seconds+=time.perf_counter()-t;seen=set()
    for r in pred:
     assert len(r)==8 and np.isfinite(r).all()
     x,y,x2,y2,tid,score,cls,_=r
     assert tid==int(tid) and tid>0 and tid not in seen and cls==0 and x2>x and y2>y
     seen.add(tid);writer.writerow([frame['frame'],int(tid),x,y,x2-x,y2-y,score,-1,-1,-1]);rows+=1
  dest.with_suffix('.tmp').replace(dest)
  record=dict(tracker=name,sequence=seq,frames=len(expected),rows=rows,association_seconds=seconds,reid_seconds=meta.get('encoder_seconds',0),replay_seconds=time.perf_counter()-wall,detector_seconds=cache['inference_wall_seconds'],prediction_sha256=sha(dest))
  dump(timing,record);timings.append(record);print('TRACK',name,seq,rows,flush=True)
 return timings
def track():
 from concurrent.futures import ProcessPoolExecutor
 import multiprocessing
 with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
  results=list(pool.map(track_method,list(NAMES)))
 dump(OUT/'tracking_timing.json',[r for group in results for r in group])
def evaluate():
 from evaluate_vttiny import evaluate as evaluate_impl
 evaluate_impl('rgb','test',OUT/'trackers',list(NAMES),OUT/'metrics')
def report():
 p=read(OUT/'protocol.json');result=read(OUT/'metrics/summary.json');timing=read(OUT/'tracking_timing.json')
 rows=[r for r in result['results'] if r['sequence']=='COMBINED_SEQ'];assert len(rows)==7
 assert len(timing)==210 and sum(t['frames'] for t in timing)==12590*7
 for name in NAMES:
  assert len(list((OUT/'trackers'/name/'data').glob('*.txt')))==30
  assert sum(t['frames'] for t in timing if t['tracker']==name)==12590
 assert sha(OUT/'weights/rgb_best.pt')==p['weight_sha256']
 lines=['# VT-Tiny-MOT：RGB 七种跟踪算法测试结果','','已完成 200 轮训练的 RGB YOLO11s best.pt；官方测试集 30 段序列、12,590 帧。采用官方 RGB GT，所有目标类别联合评估；结果为 TrackEval COMBINED_SEQ，不是逐序列百分比平均。', '', '|方法|HOTA↑|DetA↑|AssA↑|IDF1↑|MOTA↑|FP↓|FN↓|IDSW↓|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for r in sorted(rows,key=lambda r:-r['HOTA']):
  lines.append('|'+ '|'.join([NAMES[r['tracker']]]+[f'{r[k]:.2f}' for k in ['HOTA','DetA','AssA','IDF1','MOTA']]+[str(r[k]) for k in ['FP','FN','IDSW']])+'|')
 lines+=['','## 实验口径','','- 所有方法共用固定权重和检测缓存，检测阈值 0.05，NMS IoU 0.7，max_det 300，FP32，imgsz 640。原始七类检测保留，关联阶段合并为 object；不额外执行跨类别 NMS。','- 沿用已有跟踪参数，未在测试集调参。15 FPS 下缓存参数 60 对应最大丢失 30 帧。空检测帧仍推进跟踪器；空 GT 序列不排除。','- Deep OC-SORT、ImprAsso、DeepSORT、StrongSORT 共享 CLIP-VeRi 外观特征；车辆预训练，未在本数据集七类航拍小目标上微调。','- BoT-SORT 为无 ReID 版本；StrongSORT 为在线版本，不包含 AFLink/GSI。结果应以这些具体配置列入论文，不能称为原论文的完整复现。','- 未接入 CenterTrack / PFTrack，本轮共七种跟踪方法。','', '权重 SHA256：`'+p['weight_sha256']+'`','', '可复现信息见 protocol.json / source / implementation_provenance；逐序列与汇总指标见 metrics/summary.json；完整 MOT 轨迹见 trackers/<method>/data/。','', 'GPU1 用于检测和外观特征；关联阶段多进程并行，当前计时不能作为严格的单任务端到端实时速度对比。']
 (OUT/'baseline_results.md').write_text('\n'.join(lines),encoding='utf8')
 dump(OUT/'baseline_results.json',rows)
 dump(OUT/'integrity.json',dict(status='PASS',sequences=30,frames_per_method=12590,methods=7,trajectory_files=210,weight_sha256=p['weight_sha256']))
 print(json.dumps(rows,ensure_ascii=False),flush=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['all','init','smoke','detect','embeddings','track','evaluate','report']);a=ap.parse_args()
 if a.stage!='all':globals()[a.stage]();return
 if not OUT.exists():init()
 for stage in ['smoke','detect','embeddings','track','evaluate','report']:
  dump(OUT/'state.json',dict(status='RUNNING',stage=stage,pid=os.getpid(),updated=time.time()))
  try:subprocess.run([sys.executable,__file__,stage],check=True)
  except Exception:
   dump(OUT/'state.json',dict(status='FAILED',stage=stage,traceback=traceback.format_exc(),updated=time.time()));raise
 dump(OUT/'state.json',dict(status='COMPLETE',updated=time.time()))
if __name__=='__main__':main()
