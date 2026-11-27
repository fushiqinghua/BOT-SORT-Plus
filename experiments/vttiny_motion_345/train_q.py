"""Fit acceleration variance only on official training observations, with held-out parent videos."""
import gzip
import json
from collections import defaultdict
import cv2
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from boxmot.motion.cmc.sof import SOF
from boxmot.motion.kalman_filters.aabb.xywh_kf import KalmanFilterXYWH
from process_noise import features, variance, UNIT_Q
from settings import KF, TRAIN, RESULT, ROOT, PROJECT, read, save, sha, table

HORIZONS=(3,5,11,20,30)


def next_state(kf, mean, cov, warp, lost):
    m=mean.copy()
    if lost: m[6:]=0.
    m,p=kf.predict(m,cov)
    transform,offset=warp
    m=transform@m; m[:2]+=offset
    return m,transform@p@transform.T


def make_windows(kf, seq, rows, truth, warps, rng):
    lookup={(int(r.frame),int(r.gid)):(r.cx,r.cy) for r in truth.itertuples()}
    anchors=[]
    for gid,track in rows.groupby('gid'):
        last=-1; count=0
        for r in track.sort_values('frame').itertuples():
            t=int(r.frame); z=np.array([r.cx,r.cy,r.w,r.h])
            if t!=last+1:
                m,p=kf.initiate(z); count=1
            else:
                m,p=next_state(kf,m,p,warps[t],False)
                m,p=kf.update(m,p,z); count+=1
            if count>=3 and count%4==3:
                anchors.append((t,int(gid),m.copy(),p.copy()))
            last=t
    take=rng.choice(len(anchors),min(160,len(anchors)),replace=False)
    samples=[]
    for index in sorted(take):
        start,gid,m,p=anchors[index]; x=features(m)
        unit=np.zeros((8,8))
        for h in range(1,max(HORIZONS)+1):
            if start+h not in warps:break
            warp=warps[start+h]
            m,p=next_state(kf,m,p,warp,h>1)
            accel=UNIT_Q if h>1 else 0.
            f=kf._motion_mat; transform=warp[0]
            unit=transform@(f@unit@f.T+accel)@transform.T
            target=lookup.get((start+h,gid))
            if h in HORIZONS and target is not None:
                residual=np.asarray(target)-m[:2]
                samples.append((seq,h,*x,*residual,p[0,0],p[0,1],p[1,1],unit[0,0],unit[0,1],unit[1,1]))
    return samples


def bottom_probe(seq, part):
    groups={int(t):r for t,r in part.groupby('frame')}
    result=[]
    for t,r in groups.items():
        if t+1 not in groups or len(r)<2:continue
        next_rows={v.gid:v for v in groups[t+1].itertuples()}
        xy=r[['cx','cy']].to_numpy(); wh=r[['w','h']].to_numpy(); lo=xy-wh/2; hi=xy+wh/2
        ids=r.gid.to_numpy()
        overlaps=(np.minimum(hi[:,None],hi)-np.maximum(lo[:,None],lo)>0).all(2)
        for behind,front in np.argwhere(overlaps & (hi[None,:,1]>hi[:,None,1]+5.)):
            a,b=next_rows.get(ids[behind]),next_rows.get(ids[front])
            if a is None or b is None:continue
            def error(row):return np.hypot(row.cx-row.gt_cx,row.cy-row.gt_cy)/max(1,min(row.gt_w,row.gt_h))
            result.append((seq,error(a),error(b),a.score<.5 and b.score<.5))
    return result


def measures(samples,q):
    xx=samples.xx+q*samples.uxx; xy=samples.xy+q*samples.uxy; yy=samples.yy+q*samples.uyy
    det=(xx*yy-xy**2).to_numpy(); assert (det>0).all()
    d2=(yy*samples.dx**2+xx*samples.dy**2-2*xy*samples.dx*samples.dy).to_numpy()/det
    return .5*(np.log(det)+d2),d2,np.pi*5.991465*np.sqrt(det)


def main():
    cv2.setNumThreads(1); rng=np.random.default_rng(345)
    df=pd.read_pickle(KF/'pairs.pkl').query('iou>=.30 and gap>=.15'); truth=pd.read_pickle(KF/'truth.pkl')
    plan=read(TRAIN/'window_plan.json')
    allowed=set()
    for split in ('train','val'):
        with (PROJECT/'data/processed_v1'/(split+'_pairs.jsonl')).open() as file:
            for line in file:
                entry=json.loads(line); assert entry['official_split']=='train'
                allowed.add(entry['sequence'])
    assert len(allowed)==plan['official_train'] and set(df.seq)<=set(plan['sequences'])
    info=read(KF/'inputs.json'); assert sha(TRAIN/'window_plan.json')==info['window_plan']
    pools={k:set(g.parent) for k,g in df.groupby('split')}
    assert set(df.seq)<=allowed and not pools['fit']&pools['check']
    test_names=read(PROJECT/'data/processed_v1/splits.json')['splits']['test']
    assert not ({s.rsplit('_',1)[0] for s in test_names} & (pools['fit']|pools['check']))
    cv2.setRNGSeed(0); kf=KalmanFilterXYWH(); all_samples=[]; probes=[]
    for seq,rows in df.groupby('seq',sort=True):
        path=TRAIN/'training_inputs/detections'/(seq+'.json.gz')
        assert sha(path)==info['detection_files'][seq]
        with gzip.open(path,'rt') as f: frames=json.load(f)['frames']
        sof=SOF(); warps={}; cv2.setRNGSeed(0)
        for frame,det in zip(plan['sequences'][seq],frames,strict=True):
            t=frame['frame']; assert det['frame']==t and frame['image']==det['image']
            image=cv2.imread(frame['image']); assert image is not None
            boxes=np.asarray(det['detections'],float).reshape(-1,6)
            H=sof.apply(image,boxes); warps[t]=(np.kron(np.eye(4),H[:,:2]),H[:,2])
        split=rows.split.iloc[0]
        for row in make_windows(kf,seq,rows,truth[truth.seq==seq],warps,rng):all_samples.append((split,*row))
        for row in bottom_probe(seq,rows):probes.append((split,*row))
        print(seq,len(all_samples),flush=True)
    samples=pd.DataFrame(all_samples,columns='split seq h bias speed size dx dy xx xy yy uxx uxy uyy'.split())
    samples.to_pickle(ROOT/'training_windows.pkl'); fit=samples.query("split=='fit'")
    weights=1/fit.groupby(['seq','h']).h.transform('size').to_numpy()
    params={}
    for name,cols in [('q_constant',1),('q_motion',3)]:
        x=fit[['bias','speed','size']].to_numpy()[:,:cols]
        def loss(w):
            nll,_,_=measures(fit,variance(x,w))
            return np.average(nll,weights=weights)+.005*np.sum(w[1:]**2)
        initial=np.zeros(cols);initial[0]=-3.
        sol=minimize(loss,initial,bounds=[(-12.,2.77)]+[(-4.,4.)]*(cols-1),method='L-BFGS-B')
        assert sol.success and np.isfinite(sol.fun),sol.message
        params[name]=np.pad(sol.x,(0,3-cols)).tolist()
    policy=dict(version=1,params=params,horizons=HORIZONS,seed=345,max_anchors_per_sequence=160,
                features=['1','clip(log1p(5*speed/size),0,3)','clip(log(size/12),-3,3)'],
                groups={k:sorted(v) for k,v in pools.items()},sequences={k:sorted(set(g.seq)) for k,g in df.groupby('split')},
                samples=dict(samples.groupby('split').size().astype(int).to_dict()),inputs={f:sha(KF/f) for f in ('pairs.pkl','truth.pkl','inputs.json')})
    save(RESULT/'q_policy.json',policy)
    scores=[]
    for (split,h),g in samples.groupby(['split','h']):
        for name in ('native',*params):
            q=0. if name=='native' else variance(g[['bias','speed','size']].to_numpy(),params[name])
            nll,d2,area=measures(g,q)
            scores.append(dict(split=split,h=int(h),method=name,n=len(g),coverage95=float(np.mean(d2<=5.991465)),
                               nll=float(np.mean(nll)),mean_area=float(np.mean(area))))
    table(RESULT/'train_gap_calibration.csv',scores)
    probes=pd.DataFrame(probes,columns=['split','seq','back','front','both_low'])
    rows=[]
    for split,sub in probes.groupby('split'):
        for group,g in [('all',sub),('both_low',sub[sub.both_low])]:
            delta=g.back-g.front
            rows.append(dict(split=split,group=group,n=len(g),sequences=g.seq.nunique(),
                             mean_back=g.back.mean(),mean_front=g.front.mean(),fraction_back_worse=(delta>0).mean(),
                             median_paired_delta=delta.median()))
    table(RESULT/'bottom_order_training_proxy.csv',rows)
    print('LOCKED',sha(RESULT/'q_policy.json'),flush=True)


if __name__=='__main__':main()
