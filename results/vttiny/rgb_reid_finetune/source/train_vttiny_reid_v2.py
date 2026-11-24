"""Supervised visual-only adaptation with explicitly test-selected checkpoints (user-authorized)."""
import os,sys,time,math,random,collections,traceback,shutil,subprocess
from pathlib import Path
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']='4'
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='1'
from vttiny_tracking_common import ROOT,OLD,read,dump,sha
sys.path[:0]=[str(OLD/'tracking_runtime'),str(OLD/'tracking_extensions')]
import numpy as np,cv2,torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader,Sampler
cv2.setNumThreads(0)
WORK=ROOT/'reid_rgb_official85_testselect_v2'
class Crops(Dataset):
 def __init__(self,rows,augment):
  self.rows=rows;self.augment=augment
  self.cache=[cv2.cvtColor(cv2.imread(r['crop']),cv2.COLOR_BGR2RGB) for r in rows]
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  x=cv2.resize(self.cache[i],(256,256),interpolation=cv2.INTER_LINEAR)
  if self.augment:
   if random.random()<.5:x=x[:,::-1]
   x=np.clip(x.astype(np.float32)*random.uniform(.9,1.1)+random.uniform(-8,8),0,255)
  x=torch.from_numpy(np.ascontiguousarray(x.transpose(2,0,1))).float()/127.5-1
  return x,self.rows[i]['pid'],i
class IdentityBatches(Sampler):
 def __init__(self,rows,cfg):
  self.cfg=cfg;self.rows=rows;self.epoch=0;self.byid=collections.defaultdict(list);self.seqids=collections.defaultdict(list)
  for i,r in enumerate(rows):self.byid[r['pid']].append(i)
  for pid,indices in self.byid.items():self.seqids[rows[indices[0]]['sequence']].append(pid)
 def __len__(self):return self.cfg['steps_per_epoch']
 def __iter__(self):
  rng=random.Random(self.cfg['seed']+self.epoch);seqs=list(self.seqids)
  for _ in range(len(self)):
   chosen=[];parents=set();available=seqs.copy();rng.shuffle(available)
   for seq in available:
    par=seq.rsplit('_',1)[0]
    if par in parents:continue
    parents.add(par);ids=self.seqids[seq];chosen.extend(rng.sample(ids,min(4,len(ids))))
    if len(chosen)>=16:break
   assert len(chosen)>=16
   batch=[]
   for pid in chosen[:16]:batch+=rng.sample(self.byid[pid],4)
   yield batch
def worker_seed(i):
 seed=torch.initial_seed()%2**32;random.seed(seed);np.random.seed(seed)
def embedding(model,x):
 _,v,p=model.image_encoder(x)
 return torch.cat([model.bottleneck(v[:,0]),model.bottleneck_proj(p[:,0])],1)
@torch.no_grad()
def validate(model,loader,rows,gap):
 model.eval();features=[]
 for x,_,_ in loader:features.append(F.normalize(embedding(model,x.cuda(non_blocking=True)),dim=1).cpu())
 z=torch.cat(features).numpy();groups=collections.defaultdict(list)
 for i,r in enumerate(rows):groups[(r['sequence'],r['category'])].append(i)
 scores=collections.defaultdict(list);ranks=collections.defaultdict(list)
 for indices in groups.values():
  local=z[indices];sim=local@local.T
  for j,i in enumerate(indices):
   r=rows[i];valid=np.array([abs(r['frame']-rows[k]['frame'])>=gap for k in indices]);positive=np.array([rows[k]['pid']==r['pid'] for k in indices])
   if not (valid&positive).any() or not (valid&~positive).any():continue
   candidate=np.flatnonzero(valid);order=candidate[np.argsort(-sim[j,candidate],kind='stable')];hits=positive[order]
   ap=float(((np.cumsum(hits)/(np.arange(len(hits))+1))*hits).sum()/hits.sum())
   scores[r['pid']].append(ap);ranks[r['pid']].append(float(hits[0]))
 assert scores,'No valid temporal retrieval queries'
 return dict(mAP=100*float(np.mean([np.mean(v) for v in scores.values()])),rank1=100*float(np.mean([np.mean(v) for v in ranks.values()])),identities=len(scores),queries=sum(map(len,scores.values())))
def save(model,path,epoch,metric,cfg,head=None,optimizer=None):
 state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
 payload=dict(state_dict=state,epoch=epoch,validation=metric,config=cfg)
 if head is not None:payload['training_head']=head.state_dict();payload['optimizer']=optimizer.state_dict()
 temp=path.with_suffix('.tmp');torch.save(payload,temp);temp.replace(path)
def main():
 cfg=read(WORK/'config.json');torch.set_num_threads(4)
 random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert os.environ['CUDA_VISIBLE_DEVICES']=='4' and torch.cuda.device_count()==1
 torch.cuda.init()
 requested_visible_devices=os.environ['CUDA_VISIBLE_DEVICES']
 dump(WORK/'state.json',dict(status='INITIALIZING',physical_gpu=4,pid=os.getpid(),updated=time.time()))
 import vttiny_reid_adapter as adapter
 adapter.WEIGHTS=WORK/'weights/clip_veri_pretrained.pt';backend,metadata=adapter.encoder();model=backend.model
 gpu_rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True).splitlines()
 binding=[line for line in gpu_rows if line.split(',')[1].strip()==str(os.getpid())]
 assert len(binding)==1 and binding[0].startswith('GPU-fcb75e69-b298-9289-47e7-a4ffb0bd95b0'),binding
 dump(WORK/'gpu_binding_audit.json',dict(status='PASS',physical_gpu=4,pid=os.getpid(),requested_visible_devices=requested_visible_devices,nvidia_smi=binding[0],updated=time.time()))
 assert model.neck_feat=='after',model.neck_feat
 for p in model.parameters():p.requires_grad_(False)
 for name,p in model.named_parameters():
  if name.startswith(('image_encoder.transformer.resblocks.10.','image_encoder.transformer.resblocks.11.','image_encoder.ln_post.')) or name=='image_encoder.proj' or name in ['bottleneck.weight','bottleneck_proj.weight']:p.requires_grad_(True)
 train=read(WORK/'train.json');val=read(WORK/'val.json')
 assert all('/train2017/' in r['image'] for r in train['records'])
 assert all('/test2017/' in r['image'] for r in val['records'])
 assert not {r['image'] for r in train['records']}&{r['image'] for r in val['records']}
 assert sha(WORK/'train.json')==cfg['train_manifest_sha256'] and sha(WORK/'val.json')==cfg['val_manifest_sha256']
 head=nn.Linear(1280,len(train['identities']),bias=False).cuda();nn.init.normal_(head.weight,std=.001)
 sampler=IdentityBatches(train['records'],cfg)
 loader=DataLoader(Crops(train['records'],True),batch_sampler=sampler,num_workers=8,pin_memory=True,persistent_workers=True,worker_init_fn=worker_seed)
 vloader=DataLoader(Crops(val['records'],False),batch_size=64,shuffle=False,num_workers=8,pin_memory=True,persistent_workers=True,worker_init_fn=worker_seed)
 params=[p for p in model.parameters() if p.requires_grad]
 opt=torch.optim.AdamW([{'params':params,'lr':cfg['lr']},{'params':head.parameters(),'lr':cfg['head_lr']}],weight_decay=cfg['weight_decay'])
 # Inference-path parity: the train wrapper must equal the deployed 1280-D encoder.
 model.eval();x=next(iter(vloader))[0][:4].cuda()
 with torch.no_grad():assert torch.allclose(model(x),embedding(model,x),atol=1e-5,rtol=1e-5)
 initial=validate(model,vloader,val['records'],cfg['validation_min_frame_gap'])
 dump(WORK/'preflight.json',dict(status='PASS',requested_cuda_visible_devices=requested_visible_devices,backend_cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],evaluation_status='TEST_SET_SELECTED_NOT_HELD_OUT',device_name=torch.cuda.get_device_name(0),trainable_parameters=sum(p.numel() for p in params),training_head_parameters=sum(p.numel() for p in head.parameters()),inference_parity=True,baseline_validation=initial))
 history=[dict(epoch=0,validation=initial)];best=-1;started=time.time()
 print('BASELINE_VALIDATION',initial,flush=True)
 for epoch in range(1,cfg['epochs']+1):
  sampler.epoch=epoch;model.train();head.train();totals=[];start=time.time()
  factor=min(1,epoch/2)*(.1+.9*(1+math.cos(math.pi*(epoch-1)/cfg['epochs']))/2)
  for group,lr in zip(opt.param_groups,[cfg['lr'],cfg['head_lr']]):group['lr']=lr*factor
  for step,(x,y,_) in enumerate(loader,1):
   x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);opt.zero_grad(set_to_none=True)
   raw=embedding(model,x);f=F.normalize(raw,dim=1);distance=torch.cdist(f,f)
   same=y[:,None]==y[None,:];same.fill_diagonal_(False)
   positive=distance.masked_fill(~same,-1).max(1).values
   negative=distance.masked_fill(y[:,None]==y[None,:],float('inf')).min(1).values
   tri=F.relu(positive-negative+cfg['triplet_margin']).mean();ce=F.cross_entropy(head(raw),y,label_smoothing=cfg['label_smoothing']);loss=ce+tri
   assert torch.isfinite(loss),f'nonfinite loss epoch{epoch} step{step}'
   loss.backward();norm=nn.utils.clip_grad_norm_(params+list(head.parameters()),5);assert torch.isfinite(norm)
   opt.step();totals.append([loss.item(),ce.item(),tri.item()])
   if step%20==0:
    state=dict(status='TRAINING',physical_gpu=4,pid=os.getpid(),epoch=epoch,epochs=cfg['epochs'],step=step,steps=cfg['steps_per_epoch'],loss=loss.item(),updated=time.time(),elapsed_seconds=time.time()-started)
    dump(WORK/'state.json',state);print('TRAIN',epoch,step,round(loss.item(),4),flush=True)
  metric=validate(model,vloader,val['records'],cfg['validation_min_frame_gap']);row=dict(epoch=epoch,losses=np.mean(totals,axis=0).tolist(),validation=metric,seconds=time.time()-start);history.append(row)
  save(model,WORK/'weights/clip_veri_vttiny_last.pt',epoch,metric,cfg,head,opt)
  if metric['mAP']>best:
   best=metric['mAP'];save(model,WORK/'weights/clip_veri_vttiny_best.pt',epoch,metric,cfg)
  dump(WORK/'history.json',history);print('EPOCH',row,flush=True)
 dump(WORK/'state.json',dict(status='COMPLETE',physical_gpu=4,pid=os.getpid(),epochs=cfg['epochs'],best_validation_mAP=best,baseline_validation=initial,elapsed_seconds=time.time()-started,updated=time.time(),best_weight_sha256=sha(WORK/'weights/clip_veri_vttiny_best.pt')))
 print('TRAINING_COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  dump(WORK/'failure.json',dict(status='FAILED',traceback=traceback.format_exc(),updated=time.time()));raise
