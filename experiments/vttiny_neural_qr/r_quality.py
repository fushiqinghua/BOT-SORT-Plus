"""Fit observation variances from frozen train detections and aligned sensor patches."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from quality_features import FEATURES, NO_IMAGE, frame_features


AXES = ('cx','cy','w','h')
COLS = {'score': [0], 'no_image': list(range(len(NO_IMAGE))), 'full': list(range(len(FEATURES)))}
BOUND = float(np.log(64.))
FLOOR_VAR = .25**2
SEED = 4801


def scale_variance(x):
    wh=np.exp(x[:, [1,2,1,2]])
    return np.maximum(wh/20., .25)**2


class RPolicy:
    """NumPy runtime. variance(single_feature_row) is a length-4 pixel-variance vector."""
    def __init__(self, path, mode='full'):
        record = json.loads(Path(path).read_bytes())
        if (mode not in COLS or list(FEATURES)!=record['features']
                or record['models'][mode]['indices']!=COLS[mode]):
            raise ValueError('R observation model feature contract mismatch')
        self.model=record['models'][mode]
        self.indices=self.model['indices']
        self.params={k:np.asarray(self.model[k],dtype=float) for k in ('center','std','w1','b1','w2','b2')}
        self.bound=float(record['log_ratio_bound'])
        self.floor=float(record['floor_var'])

    def variance(self, features):
        x=np.asarray(features,dtype=float)
        single=x.ndim==1
        x=np.atleast_2d(x)
        if x.ndim!=2 or x.shape[1]!=len(FEATURES) or not np.isfinite(x).all():
            raise ValueError('Invalid current-frame R features')
        p=self.params
        z=np.clip((x[:,self.indices]-p['center'])/p['std'],-6.,6.)
        h=np.tanh(z@p['w1'].T+p['b1'])
        ratio=np.exp(self.bound*np.tanh((h@p['w2'].T+p['b2'])/self.bound))
        v=self.floor+scale_variance(x)*ratio
        return v[0] if single else v


def collect():
    import pandas as pd
    from qr_settings import PROJECT, KF, TRAIN, OUT, read, sha, write
    cv2.setNumThreads(1)
    root=PROJECT/'data/processed_v1'
    plan=read(TRAIN/'window_plan.json')
    inputs=read(KF/'inputs.json')
    train_det=TRAIN/'training_inputs/detections'
    plan_sha=sha(TRAIN/'window_plan.json')
    if plan_sha!=inputs['window_plan']:
        raise ValueError('Train windows differ from the fitted observation-pair archive')
    # Only use the frozen spatial transform. Quality labels, geometry bins and
    # any per-frame alignment fitted with annotations are never model inputs.
    registration_path=TRAIN.parent/'ir_raw_evidence_v1/registration.json'
    reg=read(registration_path)
    if (reg['model']!='identity' or reg['framewise'] or reg['frame_wh']!=[640,512]
            or not np.array_equal(reg['matrix'],np.eye(3)[:2]) or reg['extent']!=[1.,1.]):
        raise ValueError('This image sampler requires the verified static identity registration')
    official=set()
    for split in ('train','val'):
        with (root/(split+'_pairs.jsonl')).open(encoding='utf-8') as stream:
            for line in stream:
                v=json.loads(line)
                if v['official_split']!='train':
                    raise ValueError('Input manifest contains non-training imagery')
                official.add(v['sequence'])
    parents={s:s.rsplit('_',1)[0] for s in plan['sequences']}
    held={s.rsplit('_',1)[0] for s in read(root/'splits.json')['splits']['test']}
    if (not set(parents)<=official or set(parents.values())&held
            or set(plan['groups'])!=set(parents.values())):
        raise ValueError('Official test parent or unexpected train sequence in observation corpus')
    pair_path=KF/'pairs.pkl'
    expected_sha=read(KF/'policy.json')['training_pairs_sha']
    if sha(pair_path)!=expected_sha:
        raise ValueError('Observation-pair archive hash mismatch')
    df=pd.read_pickle(pair_path).query('iou >= .30 and gap >= .15 and score > .10').copy()
    if df.duplicated(['seq','frame','gid']).any() or set(df.split)!={'fit','check'}:
        raise ValueError('Ambiguous training observations')
    xparts=[]; yparts=[]; pparts=[]; splits=[]; sids=[]; fids=[]; gids=[]
    images_sha={}
    for seq, part in df.groupby('seq',sort=True):
        parent=parents[seq]; split=plan['groups'][parent]
        if set(part.parent)!={parent} or set(part.split)!={split}:
            raise ValueError('Window parent assignment changed')
        det_path=train_det/(seq+'.json.gz')
        if sha(det_path)!=inputs['detection_files'][seq]:
            raise ValueError('Detection file hash mismatch: '+seq)
        with gzip.open(det_path,'rt',encoding='utf-8') as stream: source=json.load(stream)
        if source['model_sha256']!=inputs['detector']:
            raise ValueError('Detector version mismatch')
        grouped={int(k):v for k,v in part.groupby('frame')}
        fingerprint=hashlib.sha256()
        for frame,(meta,det) in enumerate(zip(plan['sequences'][seq],source['frames'],strict=True),1):
            if frame not in grouped: continue
            if (meta['frame']!=frame or det['frame']!=frame or det['image']!=meta['image']
                    or det['thermal']!=meta['thermal']):
                raise ValueError('RGB/IR/detection frames do not agree')
            rgb,ir=Path(meta['image']),Path(meta['thermal'])
            if (rgb.name!=ir.name or rgb.parent.name!='00' or ir.parent.name!='01'
                    or rgb.parent.parent!=ir.parent.parent or rgb.parent.parent.name!=seq):
                raise ValueError('Mismatched sensor frame files')
            decoded=[]
            for image_path in (rgb,ir):
                encoded=image_path.read_bytes()
                fingerprint.update(hashlib.sha256(encoded).digest())
                image=cv2.imdecode(np.frombuffer(encoded,np.uint8),cv2.IMREAD_COLOR if image_path==rgb else cv2.IMREAD_GRAYSCALE)
                if image is None or image.shape[:2]!=(512,640):
                    raise ValueError(f'Unreadable or mis-sized sensor frame: {image_path}')
                decoded.append(image)
            z=np.asarray(det['detections'],float).reshape(-1,6)
            partf=grouped[frame]
            obs=partf[list(AXES)].to_numpy(dtype=float)
            truth=partf[['gt_'+k for k in AXES]].to_numpy(dtype=float)
            rows=np.column_stack(((z[:,:2]+z[:,2:4])/2,z[:,2:4]-z[:,:2],z[:,4]))
            observed=np.column_stack((obs,partf.score.to_numpy()))
            lookup={tuple(row):j for j,row in enumerate(rows)}
            idx=[lookup[tuple(row)] for row in observed]
            if len(set(idx))!=len(idx) or not np.array_equal(rows[idx],observed):
                raise ValueError('Cached detector-row lookup is ambiguous')
            xparts.append(frame_features(*decoded,z)[idx])
            yparts.append(obs-truth)
            count=len(idx)
            pparts.extend([parent]*count); splits.extend([split]*count)
            sids.extend([seq]*count); fids.extend([frame]*count); gids.extend(partf.gid.astype(int))
        images_sha[seq]=fingerprint.hexdigest()
        print('R images',seq,len(part),flush=True)
    record={'detector_sha256':inputs['detector'],'pairs_sha256':expected_sha,'plan_sha256':plan_sha,
            'registration_sha256':sha(registration_path),'image_streams_sha256':images_sha,
            'groups':plan['groups'],'test_parents_excluded':sorted(held),'selection':'IoU>=.30, row/column gap>=.15, frozen mutual same-class pairing',
            'feature_sources':'current matched score, geometry and contemporaneous detection rivals, raw RGB/IR pixels; annotation columns used only as targets/paired supervision'}
    write(OUT/'r_training_inputs.json',record)
    return (np.concatenate(xparts), np.concatenate(yparts), np.asarray(pparts),
            np.asarray(splits), np.asarray(sids),np.asarray(fids),np.asarray(gids))


def fit(x,y,parents,mode,epochs,device):
    import torch
    from torch import nn
    torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    idx=COLS[mode]
    center=x[:,idx].mean(0,dtype=np.float64); std=np.maximum(x[:,idx].std(0,dtype=np.float64),1.e-3)
    def tensor(a):return torch.as_tensor(a,dtype=torch.float32,device=device)
    z=tensor(np.clip((x[:,idx]-center)/std,-6.,6.)); squared=tensor(y*y)
    base=tensor(scale_variance(x))
    _,p,n=np.unique(parents,return_inverse=True,return_counts=True)
    importance=1./n[p]; importance=tensor(importance/importance.mean())
    net=nn.Sequential(nn.Linear(len(idx),16),nn.Tanh(),nn.Linear(16,4)).to(device)
    nn.init.zeros_(net[2].weight); nn.init.zeros_(net[2].bias)
    opt=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=.001)
    rng=np.random.default_rng(SEED); net.train()
    for _ in range(epochs):
        for raw in np.array_split(rng.permutation(len(x)),max(1,(len(x)+1023)//1024)):
            ids=torch.as_tensor(raw,device=device)
            ratio=BOUND*torch.tanh(net(z[ids])/BOUND)
            var=FLOOR_VAR+base[ids]*torch.exp(ratio)
            loss=(importance[ids]*(.5*(var.log()+squared[ids]/var)).mean(1)).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),5.); opt.step()
    def dump(t):return t.detach().cpu().numpy().tolist()
    return dict(indices=idx,center=center.tolist(),std=std.tolist(),w1=dump(net[0].weight),
                b1=dump(net[0].bias),w2=dump(net[2].weight),b2=dump(net[2].bias))


def summarize(x,y,parents,split,seq,frame,gid,policy_path):
    from qr_settings import OUT,table
    from scipy.stats import rankdata,spearmanr
    fit_low=(split=='fit') & (x[:,0]<.5)
    norm_error=np.linalg.norm(y[:,:2],axis=1)/np.maximum(1.,np.exp(x[:,1:3]).min(1))
    bias_cut=float(np.quantile(norm_error[fit_low],.75))
    eval_rows=[];bias_rows=[];sample_rows=[]
    methods={m:RPolicy(policy_path,m).variance(x) for m in COLS}
    methods['scale_proxy']=scale_variance(x)
    for m,var in methods.items():
        pos_nis=np.sum(y[:,:2]**2/var[:,:2],axis=1)
        nll=.5*(np.log(var)+y*y/var)
        conf=np.sqrt(var[:,0]+var[:,1])/np.maximum(1.,np.exp(x[:,1:3]).min(1))
        for group in ('fit','check'):
            for bucket,mask in [('all',split==group),('low',(split==group)&(x[:,0]<.5)),('high',(split==group)&(x[:,0]>=.5))]:
                if not mask.any():continue
                eval_rows.append(dict(method=m,split=group,score=bucket,n=int(mask.sum()),
                    parents=len(np.unique(parents[mask])),nll_axis_mean=float(nll[mask].mean()),
                    center_nll=float(nll[mask,:2].sum(1).mean()),
                    center_95_coverage=float(np.mean(pos_nis[mask]<=5.991465)),
                    center_95_area_median=float(np.median(np.pi*5.991465*np.sqrt(var[mask,0]*var[mask,1])))))
        low=(split=='check') & (x[:,0]<.5)
        for size,mask in [('all',low),('short_lt8',low & (x[:,1:3].min(1)<np.log(8))),
                          ('short_ge8',low & (x[:,1:3].min(1)>=np.log(8)))]:
            actual=norm_error[mask]; rounded=conf[mask]; high=actual>bias_cut
            pos=int(high.sum());neg=int((~high).sum())
            auc=float((rankdata(rounded)[high].sum()-pos*(pos+1)/2)/(pos*neg)) if pos and neg else None
            corr=spearmanr(rounded,actual).statistic if len(np.unique(rounded))>1 and len(np.unique(actual))>1 else None
            bias_rows.append(dict(method=m,split='check',score='0.10..0.50',size=size,n=len(actual),
                large_bias=pos,bias_cut_fit_norm_center=bias_cut,roc_auc=auc,
                rank_correlation=None if corr is None else float(corr),
                mean_predicted_norm_sigma_large=float(np.mean(rounded[high])) if pos else None,
                mean_predicted_norm_sigma_other=float(np.mean(rounded[~high])) if neg else None))
        if m=='full':
            for ix in np.flatnonzero(low):
                sample_rows.append(dict(seq=str(seq[ix]),frame=int(frame[ix]),gt=int(gid[ix]),score=float(x[ix,0]),
                    norm_center_error=float(norm_error[ix]),predicted_norm_sigma=float(conf[ix]),
                    ground_truth_large_bias=bool(norm_error[ix]>bias_cut)))
    table(OUT/'r_holdout_calibration.csv',eval_rows)
    table(OUT/'r_low_bias.csv',bias_rows)
    table(OUT/'r_low_bias_events.csv',sample_rows)


def main():
    from qr_settings import OUT,write
    import torch
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--epochs',type=int,default=32)
    ap.add_argument('--device',default='cpu')
    args=ap.parse_args()
    if args.epochs<1:raise ValueError('R needs a positive training epoch count')
    torch.set_num_threads(1)
    x,y,parents,split,seq,frame,gid=collect()
    fit_rows=split=='fit'
    if set(parents[fit_rows])&set(parents[~fit_rows]):raise ValueError('Training/held-out parent overlap')
    models={m:fit(x[fit_rows],y[fit_rows],parents[fit_rows],m,args.epochs,args.device) for m in COLS}
    policy_path=OUT/'r_policy.json'
    write(policy_path,dict(version=1,features=FEATURES,axes=AXES,models=models,log_ratio_bound=BOUND,floor_var=FLOOR_VAR,
          seed=SEED,training_epochs=args.epochs,fit_rows=int(fit_rows.sum()),check_rows=int((~fit_rows).sum()),
          fit_parents=sorted(set(parents[fit_rows])),check_parents=sorted(set(parents[~fit_rows])),
          convention='absolute xywh pixel variances; known detector dimensions set common scale for all score/no_image/full models'))
    summarize(x,y,parents,split,seq,frame,gid,policy_path)
    print('R POLICY COMPLETE',policy_path,flush=True)


if __name__=='__main__':
    main()
