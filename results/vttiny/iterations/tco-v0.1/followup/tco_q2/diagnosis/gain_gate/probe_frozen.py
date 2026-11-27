"""In-memory, first-selected-epoch loss replay and frozen quality-input interventions.

Caller supplies q2_root and old_root. No tracker, holdout, or test data is used.
"""
import copy
import csv
import gzip
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import torch
from torch.nn import functional as F


def cache(corpus, rows, net, device, batch_size):
    """Store the exact inputs at the shared head, detached from the P3 encoder."""
    saved = {}
    with torch.inference_mode():
        for start in range(0,len(rows),batch_size):
            ids = rows[start:start+batch_size]
            b = corpus.batch('P3',ids,device)
            captures = []
            handle = net.quality_head.register_forward_pre_hook(lambda module,args: captures.append(args[0]))
            try:
                out = net(b)
            finally:
                handle.remove()
            fr, fn = captures[0].chunk(2)
            data = dict(fr=fr,fn=fn,raw=b['raw_xyxy'],gt=b['gt_xyxy'],label=b['support_label'],
                        refined=out['refined_xyxy'],legal=out['box_valid'])
            for name,tensor in data.items():
                if name not in saved:
                    saved[name] = torch.empty((len(rows),*tensor.shape[1:]),device=device,dtype=tensor.dtype)
                saved[name][start:start+len(ids)] = tensor
    return {k:v.clone() for k,v in saved.items()}


def scores(head, fr, fn):
    return head(torch.cat((fr,fn))).flatten().sigmoid().chunk(2)


def loss_replay(name, base, feat, cal, policy, quality):
    head = copy.deepcopy(base.quality_head)
    opt = torch.optim.AdamW(head.parameters(),lr=policy['learning_rate'],weight_decay=policy['weight_decay'])
    obj = dict(policy['objective'])
    if 'unit_cross' in name:
        obj['crossing_weight']=1.
    if name=='paired_no_direction':
        obj['direction_weight']=0.
    if name=='paired_no_regression':
        obj['gain_regression_weight']=0.
    for start in range(0,len(feat['label']),policy['batch_size']):
        f={k:v[start:start+policy['batch_size']] for k,v in feat.items()}
        qr,qn=scores(head,f['fr'],f['fn'])
        out=dict(quality_raw=qr,quality_refined=qn,refined_xyxy=f['refined'],box_valid=f['legal'])
        val=quality.loss_terms(out,dict(raw_xyxy=f['raw']),f['label'],f['gt'],obj,
                               mode='absolute_only' if name=='absolute_only' else 'paired_gain')
        opt.zero_grad(set_to_none=True)
        val['loss'].backward()
        opt.step()
    with torch.no_grad():
        a,b=scores(head,cal['fr'],cal['fn'])
    return (b-a).cpu().numpy()


def quant(x):
    x=np.asarray(x)
    if len(x)==0:
        return None
    return dict(median=float(np.median(x)),mean=float(np.mean(x)),p10=float(np.quantile(x,.1)),
                p90=float(np.quantile(x,.9)))


def profile(feat, model, data, track, seq, id_to_seq):
    pos=((feat['label']==1)&feat['legal']).cpu().numpy()
    ir=model.paired_iou(feat['raw'],feat['gt']).cpu().numpy()
    ip=model.paired_iou(feat['refined'],feat['gt']).cpu().numpy()
    delta=ip-ir
    seq=seq[data]
    track=track[data]
    bins=dict(all=pos,raw_below=pos&(ir<.5),up=pos&(ir<.5)&(ip>=.5),down=pos&(ir>=.5)&(ip<.5),
              improve=pos&(delta>=.005),damage=pos&(delta<=-.005))
    summary={}
    for k,mask in bins.items():
        keys,counts=np.unique(seq[mask],return_counts=True)
        ranked=sorted(((id_to_seq[int(i)],int(n)) for i,n in zip(keys,counts)), key=lambda v:-v[1])
        summary[k]=dict(n=int(mask.sum()),tracks=len(np.unique(track[mask])),by_sequence=dict(ranked),
                        true_gain=quant(delta[mask]))
    return summary,bins


@torch.no_grad()
def training_neighbors(fit, cal, fit_mask, query_mask, tracks, model):
    """Describe signed visual changes in fit, with one neighbor per training track."""
    fi=np.flatnonzero(fit_mask)
    query=np.flatnonzero(query_mask)
    def directions(data,ids):
        return F.normalize(data['fn'][ids,:576]-data['fr'][ids,:576],dim=1)
    sim=(directions(cal,query) @ directions(fit,fi).T).cpu().numpy()
    gain=(model.paired_iou(fit['refined'],fit['gt'])-model.paired_iou(fit['raw'],fit['gt'])).cpu().numpy()
    result={}
    for index,score in zip(query,sim):
        chosen=[]
        seen=set()
        for j in np.argsort(-score,kind='stable'):
            row=fi[j]
            if tracks[row] in seen:
                continue
            chosen.append(j)
            seen.add(tracks[row])
            if len(chosen)==20:
                break
        record={}
        for n in (5,20):
            subset=chosen[:n]
            effects=gain[fi[subset]]
            record['neighbor_mean_gain_'+str(n)]=float(effects.mean())
            record['neighbor_improving_'+str(n)]=int((effects>=.005).sum())
            record['neighbor_damaging_'+str(n)]=int((effects<=-.005).sum())
            record['neighbor_mean_cosine_'+str(n)]=float(score[subset].mean())
        result[index]=record
    return result


def probe(root,old):
    sys.path[:0]=[str(old/'src'),str(root/'src/quality')]
    import train, model, quality_loss
    from fit_quality import _assert_fixed_proposer
    run=root/'quality_run'
    lock=json.loads((run/'calibration_lock.json').read_bytes())
    policy=json.loads((root/'src/quality/policy.json').read_bytes())
    manifest=json.loads((root/'corpus/crop_manifest.json').read_bytes())
    receipt=lock['model_and_policy']
    for file,key in [(root/'corpus/crop_manifest.json','crop_manifest_sha256'),
                     (root/'src/quality/policy.json','policy_sha256'),(Path(train.__file__),'original_trainer_sha256'),
                     (Path(model.__file__),'original_model_sha256'),(Path(quality_loss.__file__),'quality_source_sha256')]:
        assert train.sha256(file)==receipt[key]
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(2)
    modules={}
    for name,spec in dict(base=lock['base'],**lock['modes']).items():
        path=run/spec['checkpoint']
        assert train.sha256(path)==spec['checkpoint_sha256']
        net=model.TCO('P3').to(device).eval()
        net.load_state_dict(torch.load(path,map_location=device,weights_only=True)['model_state_dict'])
        if name!='base':
            assert spec['epoch']==1 and spec['thresholds'] is None
            _assert_fixed_proposer(net,modules['base'].state_dict())
        modules[name]=net
    sets={}
    for part in ('fit','calibration'):
        spec=manifest['partitions'][part]
        sets[part]=train.Corpus(root/'corpus'/part,spec['events'],spec['capacity'])
    ft,ca=sets.values()
    order=train.fit_order(ft,policy['seed'],policy['base_epochs']+1)
    order=order[ft.arrays['assigned'][order]]
    cids=np.flatnonzero(ca.arrays['assigned'] & (ca.arrays['support_label']==1))
    fit=cache(ft,order,modules['base'],device,policy['batch_size'])
    cal=cache(ca,cids,modules['base'],device,policy['batch_size'])
    seqnames={v:k for k,v in manifest['sequence_ids'].items()}
    profiles={}
    profiles['fit'],fmask=profile(fit,model,order,ft.arrays['track_key'],ft.arrays['sequence_id'],seqnames)
    profiles['calibration'],cmask=profile(cal,model,cids,ca.arrays['track_key'],ca.arrays['sequence_id'],seqnames)
    fit_batches=[slice(i,i+policy['batch_size']) for i in range(0,len(order),policy['batch_size'])]
    profiles['fit_batches']=dict(total=len(fit_batches),with_up=sum(bool(fmask['up'][s].any()) for s in fit_batches),
                                 with_down=sum(bool(fmask['down'][s].any()) for s in fit_batches))
    modes=('absolute_only','paired_gain','paired_unit_cross','paired_no_direction','paired_no_regression')
    control={m:loss_replay(m,modules['base'],fit,cal,policy,quality_loss) for m in modes}
    # Replace only the visual block or only the four box coordinates, keeping the
    # common history fixed. Average both replacement orders for an additive split.
    with torch.inference_mode():
        fr,fn=cal['fr'],cal['fn']
        assert torch.equal(fr[:,576:640],fn[:,576:640])
        vis=torch.cat((fn[:,:576],fr[:,576:]),dim=1)
        coord=torch.cat((fr[:,:640],fn[:,640:]),dim=1)
        obs=dict(cosine=F.cosine_similarity(fr[:,:576],fn[:,:576]).cpu().numpy(),
                 visual_relative_l2=((fn[:,:576]-fr[:,:576]).norm(dim=1)/fr[:,:576].norm(dim=1)).cpu().numpy())
        for mode in ('absolute_only','paired_gain'):
            h=modules[mode].quality_head
            ar,an=scores(h,fr,fn)
            av,ac=scores(h,vis,coord)
            dl=(av-ar+an-ac)/2
            dc=(ac-ar+an-av)/2
            assert torch.allclose(an-ar,dl+dc,atol=1e-6)
            for k,v in dict(gain=an-ar,visual=dl,coordinates=dc).items():
                obs[mode+'_'+k]=v.cpu().numpy()
    historical={}
    for mode in ('absolute_only','paired_gain'):
        with gzip.open(run/(mode+'_calibration.csv.gz'),'rt') as f:
            pred=list(csv.DictReader(f))
        historical[mode]=np.array([float(pred[i]['quality_refined'])-float(pred[i]['quality_raw']) for i in cids])
    down,up=cmask['down'],cmask['up']
    abs_pass=historical['absolute_only']>=.005
    paired_pass=historical['paired_gain']>=.005
    added=down & ~abs_pass & paired_pass
    assert down.sum()==72 and up.sum()==14 and added.sum()==29
    neighbors=training_neighbors(fit,cal,fmask['all'],up|down,ft.arrays['track_key'][order],model)
    records={}
    for mode,vals in control.items():
        mask=vals>=.005
        out=dict(up=int((mask&up).sum()),down=int((mask&down).sum()),of_original_added=int((mask&added).sum()))
        if mode in historical:
            truth=historical[mode]
            out['max_replay_error']=float(np.max(abs(vals-truth)))
            out['mean_replay_error']=float(np.mean(abs(vals-truth)))
            out['crossing_gate_mismatches']=int(((mask!=(truth>=.005))&(up|down)).sum())
            out['all_legal_gate_mismatches']=int(((mask!=(truth>=.005))&cmask['all']).sum())
            out['max_frozen_head_error']=float(np.max(abs(obs[mode+'_gain']-truth)))
        records[mode]=out
    fields=['row',*obs,*control,*next(iter(neighbors.values()))]
    csvbuf=io.StringIO()
    w=csv.writer(csvbuf)
    w.writerow(fields)
    for j,idx in enumerate(cids):
        if cmask['all'][j]:
            w.writerow([int(idx),*[float(v[j]) for v in obs.values()],*[float(v[j]) for v in control.values()],
                        *neighbors.get(j,{}).values()])
    summary=dict(profiles=profiles,first_epoch_controls=records,feature_groups={},device=str(device),
                 checkpoints={k:s['checkpoint_sha256'] for k,s in dict(base=lock['base'],**lock['modes']).items()},
                 scope='fit and calibration only; counterfactual heads discarded; no tracker replay')
    for g,mask in dict(up=up,down=down,added=added,both=down&abs_pass&paired_pass,
                       neither=down&~abs_pass&~paired_pass,removed=down&abs_pass&~paired_pass).items():
        summary['feature_groups'][g]={k:quant(vals[mask]) for k,vals in obs.items()}
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('diagnostics.json',json.dumps(summary,indent=2,allow_nan=False))
        archive.writestr('feature_and_loss_controls.csv',csvbuf.getvalue())
    return buf.getvalue()
