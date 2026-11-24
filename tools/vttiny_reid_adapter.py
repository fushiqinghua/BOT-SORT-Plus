"""Shared verified CLIP-VeRi encoder and cache-only tracker construction."""
import sys
from types import SimpleNamespace
from vttiny_tracking_common import ROOT, OLD, FRAMEWORK, dump, sha
sys.path[:0]=[str(FRAMEWORK),str(OLD/'tracking_runtime'),str(OLD/'tracking_extensions')]
import numpy as np
import torch
WEIGHTS=ROOT/'reid_weights/clip_veri.pt'
CONFIGS={
 'deepocsort':dict(det_thresh=.5,max_age=30,min_hits=3,iou_threshold=.3,delta_t=3,inertia=.2,w_association_emb=.75,alpha_fixed_emb=.95,aw_param=.5,embedding_off=False,cmc_off=False,aw_off=False,Q_xy_scaling=.01,Q_s_scaling=.0001),
 'imprassoc':dict(track_high_thresh=.5,track_low_thresh=.1,new_track_thresh=.5,track_buffer=90,frame_rate=10,match_thresh=.65,second_match_thresh=.19,overlap_thresh=.55,proximity_thresh=.1,appearance_thresh=.25,lambda_=.05,with_reid=True)
}

def encoder():
 from boxmot.reid.backbones.clip.clip import clip
 original=clip._download
 clip._download=lambda url:original(url,root=str(OLD/'reid_weights'))
 from boxmot.reid.core.auto_backend import ReidAutoBackend
 backend=ReidAutoBackend(weights=WEIGHTS,device=torch.device('cuda:0'),half=False).model
 checkpoint=torch.load(WEIGHTS,map_location='cpu',weights_only=False)
 state=checkpoint.get('state_dict',checkpoint)
 state={k.removeprefix('module.'):v for k,v in state.items()}
 current=backend.model.state_dict()
 relevant=[k for k in current if k.startswith(('image_encoder.','bottleneck.','bottleneck_proj.'))]
 for k in relevant:
  assert k in state and current[k].shape==state[k].shape,(k,current[k].shape,state.get(k,torch.empty(0)).shape)
  assert torch.equal(current[k].cpu(),state[k].to(current[k].dtype)),k
 assert len(relevant)>100
 backend.warmup()
 return backend,{'name':'CLIP-ReID ViT-B/16 VeRi','weights':str(WEIGHTS),'sha256':sha(WEIGHTS),'source':'https://drive.google.com/uc?id=1RyfHdOBI2pan_wIGSim5-l6cM4S2WN8e','inference_state_keys_verified':len(relevant),'input_hw':list(backend.input_shape),'precision':'float32','training_domain':'VeRi visible-light vehicles; not adapted to M3OT aerial/IR','rgbt_embedding_reference':'RGB crop from fused detector box; not cross-modal ReID fusion'}

class CacheOnlyEncoder:
 def get_features(self,*args,**kwargs):raise AssertionError('Uncached embedding extraction forbidden')

def tracker(name):
 if name=='deepocsort':
  import boxmot.trackers.deepocsort.deepocsort as module
  cls=module.DeepOcSort
 else:
  import boxmot_legacy.trackers.imprassoc.imprassoctrack as module
  cls=module.ImprAssocTrack
 original=module.ReidAutoBackend
 module.ReidAutoBackend=lambda **kwargs:SimpleNamespace(model=CacheOnlyEncoder())
 try:return cls(reid_weights=WEIGHTS,device=torch.device('cpu'),half=False,**CONFIGS[name])
 finally:module.ReidAutoBackend=original

if __name__=='__main__':
 torch.cuda.init();torch.set_num_threads(4)
 backend,metadata=encoder();dump(ROOT/'reid_weights/verified_encoder.json',metadata)
 image=np.zeros((512,640,3),dtype=np.uint8);image[50:100,50:100]=127
 features=backend.get_features(np.array([[50,50,100,100]],dtype=np.float32),image)
 assert np.isfinite(features).all() and abs(np.linalg.norm(features)-1)<1e-4
 print(metadata,'feature_shape',features.shape,flush=True)
 for name in CONFIGS:
  tr=tracker(name)
  for f in range(12):
   det=np.empty((0,6),np.float32) if f in (5,6) else np.array([[50+f,50,100+f,100,.95,0]],np.float32)
   emb=np.empty((0,features.shape[1]),np.float32) if len(det)==0 else features.copy()
   out=tr.update(det,image,embs=emb);assert np.isfinite(out).all()
  print('SMOKE',name,'PASS',out.tolist(),flush=True)
