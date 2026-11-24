"""DeepSORT official core plus BoxMOT online StrongSORT; cached CLIP-VeRi inputs."""
from pathlib import Path
from types import SimpleNamespace
from vttiny_reid_adapter import ROOT,WEIGHTS,CacheOnlyEncoder
import numpy as np
import torch
from deep_sort_official.detection import Detection
from deep_sort_official.nn_matching import NearestNeighborDistanceMetric
from deep_sort_official.tracker import Tracker
from deep_sort_official.track import Track
CONFIGS={
 'deepsort':dict(min_conf=.5,confidence_comparison='>',max_cos_dist=.2,max_iou_dist=.7,max_age=30,n_init=3,nn_budget=100,output_max_time_since_update=1,reid='CLIP-VeRi shared cache'),
 'strongsort':dict(min_conf=.6,max_cos_dist=.4,max_iou_dist=.7,max_age=30,n_init=3,nn_budget=100,mc_lambda=.98,ema_alpha=.9)
}

class ReportingTrack(Track):
 def update(self,kf,detection):
  super().update(kf,detection)
  self.output_confidence=detection.confidence;self.output_det_index=detection.source_index

class ReportingTracker(Tracker):
 def _initiate_track(self,detection):
  # Same initialization as official Tracker, with output-only metadata.
  mean,covariance=self.kf.initiate(detection.to_xyah())
  track=ReportingTrack(mean,covariance,self._next_id,self.n_init,self.max_age,detection.feature)
  track.output_confidence=detection.confidence;track.output_det_index=detection.source_index
  self.tracks.append(track);self._next_id+=1

class DeepSortAdapter:
 def __init__(self):
  cfg=CONFIGS['deepsort'];self.min_conf=cfg['min_conf']
  self.tracker=ReportingTracker(NearestNeighborDistanceMetric('cosine',cfg['max_cos_dist'],cfg['nn_budget']),max_iou_distance=cfg['max_iou_dist'],max_age=cfg['max_age'],n_init=cfg['n_init'])
 def update(self,dets,img,embs=None):
  assert embs is not None and len(dets)==len(embs)
  detections=[]
  for index in np.flatnonzero(dets[:,4]>self.min_conf):
   d=dets[index];assert d[5]==0 and np.isclose(np.linalg.norm(embs[index]),1,atol=1e-4)
   tlwh=d[:4].copy();tlwh[2:]-=tlwh[:2]
   detection=Detection(tlwh,d[4],embs[index].copy());detection.source_index=int(index);detections.append(detection)
  self.tracker.predict();self.tracker.update(detections)
  rows=[]
  for t in self.tracker.tracks:
   if not t.is_confirmed() or t.time_since_update>1:continue
   rows.append([*t.to_tlbr(),t.track_id,t.output_confidence,0,t.output_det_index if t.time_since_update==0 else -1])
  return np.asarray(rows,np.float64).reshape(-1,8)

def tracker(name):
 if name=='deepsort':return DeepSortAdapter()
 assert name=='strongsort'
 import boxmot.trackers.strongsort.strongsort as module
 original=module.ReidAutoBackend
 module.ReidAutoBackend=lambda **kw:SimpleNamespace(model=CacheOnlyEncoder())
 try:return module.StrongSort(reid_weights=WEIGHTS,device=torch.device('cpu'),half=False,**CONFIGS[name])
 finally:module.ReidAutoBackend=original

