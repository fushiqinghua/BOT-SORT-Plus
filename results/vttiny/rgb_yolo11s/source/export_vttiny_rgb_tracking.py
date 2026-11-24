"""Audit and package RGB tracking results, without changing experiments."""
import sys,json,csv,zipfile,shutil,importlib.metadata
from collections import Counter,defaultdict
from pathlib import Path
from vttiny_tracking_common import ROOT,OLD,read,dump,sha
from run_vttiny_rgb_tracking import OUT,NAMES
import cv2,numpy as np
def main():
 assert read(OUT/'state.json')['status']=='COMPLETE'
 p=read(OUT/'protocol.json');detection_counts=Counter();high_counts=Counter();gt_counts={};integrity=[]
 for seq,frames in p['sequences'].items():
  cache=read(OUT/'detections'/(seq+'.json'))
  assert [f['frame'] for f in cache['frames']]==[f['frame'] for f in frames]
  for f in cache['frames']:
   for d in f['detections']:
    detection_counts[int(d[5])]+=1
    if d[4]>.5:high_counts[int(d[5])]+=1
  gt=ROOT/'data/processed_v1/mot/rgb/test'/seq/'gt/gt.txt'
  original_gt=ROOT/'data/official_gt/GT/GT_00/test'/seq/'gt/gt.txt'
  assert gt.read_bytes()==original_gt.read_bytes()
  gt_counts[seq]=dict(rows=sum(bool(s.strip()) for s in gt.read_text().splitlines()),sha256=sha(gt))
  for name in NAMES:
   dest=OUT/'trackers'/name/'data'/(seq+'.txt');seen=set();rows=0
   for row in csv.reader(dest.open()):
    assert len(row)==10
    f,tid,x,y,w,h,score,*_=map(float,row)
    assert f==int(f) and 1<=f<=len(frames) and tid==int(tid) and tid>0 and (f,tid) not in seen
    assert np.isfinite(list(map(float,row))).all() and w>0 and h>0
    seen.add((f,tid));rows+=1
   t=read(OUT/'timing'/name/(seq+'.json'));assert t['rows']==rows and t['prediction_sha256']==sha(dest)
   integrity.append(dict(tracker=name,sequence=seq,rows=rows,sha256=sha(dest)))
 packages={}
 for name in ['torch','numpy','opencv-python','scipy','boxmot']:
  try:packages[name]=importlib.metadata.version(name)
  except importlib.metadata.PackageNotFoundError:packages[name]='unknown'
 sources={}
 for directory in [ROOT/'tools',ROOT/'external/TrackEval/trackeval',OLD/'tracking_runtime/boxmot',OLD/'tracking_extensions']:
  sources[str(directory)]={str(f.relative_to(directory)):sha(f) for f in directory.rglob('*.py')}
 dump(OUT/'audit.json',dict(status='PASS',trajectory_validation=integrity,official_rgb_gt=gt_counts,all_detection_class_counts=dict(detection_counts),high_confidence_class_counts=dict(high_counts),packages=packages,source_sha256=sources))
 import torch
 checkpoint=torch.load(OUT/'weights/rgb_best.pt',map_location='cpu',weights_only=False)
 dump(OUT/'checkpoint_metadata.json',{k:checkpoint.get(k) for k in ['epoch','date','best_fitness','train_metrics','version']})
 previews=OUT/'previews';previews.mkdir(exist_ok=True)
 selected=[list(p['sequences'])[i] for i in (0,14,29)]
 for seq in selected:
  frames=p['sequences'][seq];fnum=min(100,len(frames));img=cv2.imread(frames[fnum-1]['image']);assert img is not None
  tiles=[]
  for name in ['rgb']+list(NAMES):
   tile=np.zeros((548,640,3),np.uint8);tile[36:]=img
   title='RGB input' if name=='rgb' else NAMES[name]
   if name!='rgb':
    for row in csv.reader((OUT/'trackers'/name/'data'/(seq+'.txt')).open()):
     f,tid,x,y,w,h,*_=map(float,row)
     if int(f)!=fnum:continue
     c=tuple(int(v) for v in np.random.default_rng(int(tid)).integers(70,256,3))
     cv2.rectangle(tile,(round(x),round(y)+36),(round(x+w),round(y+h)+36),c,1)
     cv2.putText(tile,str(int(tid)),(round(x),max(48,round(y)+33)),cv2.FONT_HERSHEY_SIMPLEX,.36,c,1,cv2.LINE_AA)
   cv2.putText(tile,f'{title} | frame {fnum}',(8,25),cv2.FONT_HERSHEY_SIMPLEX,.56,(255,255,255),1,cv2.LINE_AA);tiles.append(tile)
  contact=np.concatenate([np.concatenate(tiles[i:i+2],axis=1) for i in range(0,8,2)],axis=0)
  cv2.imwrite(str(previews/(seq+'.jpg')),contact,[cv2.IMWRITE_JPEG_QUALITY,92])
 dump(previews/'selection.json',dict(rule='Sequences at indices 0,14,29 of sorted official test list; frame min(100,sequence_length). Not selected by metric.',sequences=selected))
 export=OUT/'delivery';export.mkdir(exist_ok=True)
 for name in ['baseline_results.md','baseline_results.json','protocol.json','integrity.json','audit.json','smoke.json','reid_encoder.json','tracking_timing.json','state.json','checkpoint_metadata.json']:
  shutil.copy2(OUT/name,export/name)
 for name in ['trackers','metrics','source','implementation_provenance','previews']:
  shutil.copytree(OUT/name,export/name,dirs_exist_ok=True)
 shutil.copy2(__file__,export/'source'/Path(__file__).name)
 archive=OUT/'rgb_test_7trackers_results.zip'
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
  for f in export.rglob('*'):
   if f.is_file():z.write(f,f.relative_to(export))
 dump(OUT/'delivery_manifest.json',dict(archive=str(archive),sha256=sha(archive),size=archive.stat().st_size,files=sum(f.is_file() for f in export.rglob('*')),contents='Metrics, all 210 MOT trajectories, protocol, source, audit, three deterministic contact sheets. Detector/ReID weights and detection/embedding caches remain on server.'))
 print((OUT/'delivery_manifest.json').read_text(),flush=True)
if __name__=='__main__':main()
