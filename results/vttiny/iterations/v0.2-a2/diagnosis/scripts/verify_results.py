"""Offline check on the contents of the unpacked delivery; requires NumPy and SciPy."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def mot(file, n, is_gt=False):
    frames=[[] for _ in range(n)]
    with file.open(encoding="utf-8-sig") as stream:
        for row in csv.reader(stream):
            if row and (not is_gt or float(row[6])!=0):
                frames[int(row[0])-1].append([float(x) for x in row[1:6]])
    return [np.asarray(a,dtype=float).reshape(-1,5) for a in frames]


def ious(gt,pred):
    a=gt[:,1:].copy(); b=pred[:,1:].copy()
    a[:,2:]+=a[:,:2]; b[:,2:]+=b[:,:2]
    lw=np.maximum(a[:,None,:2],b[None,:,:2])
    up=np.minimum(a[:,None,2:],b[None,:,2:])
    inter=np.maximum(up-lw,0).prod(axis=2)
    union=(a[:,2:]-a[:,:2]).prod(axis=1)[:,None]+(b[:,2:]-b[:,:2]).prod(axis=1)[None,:]-inter
    return np.divide(inter,union,out=np.zeros_like(inter),where=union>np.finfo(float).eps)


def match(gt, pred):
    prev,last={},{}
    fp=fn=sw=0
    covered=set()
    for f,(g,p) in enumerate(zip(gt,pred),1):
        if not len(g) or not len(p):
            fp+=len(p);fn+=len(g)
            continue
        iou=ious(g,p)
        preference=np.asarray([prev.get(int(i),-1) for i in g[:,0]])[:,None]==p[None,:,0]
        weight=iou+1000*preference
        weight[iou<.5-np.finfo(float).eps]=0.
        ii,jj=linear_sum_assignment(-weight)
        matches=[(int(g[i,0]),int(p[j,0])) for i,j in zip(ii,jj) if weight[i,j]>np.finfo(float).eps]
        fp+=len(p)-len(matches);fn+=len(g)-len(matches)
        prev=dict(matches)
        for gid,tid in matches:
            sw+=int(gid in last and last[gid]!=tid)
            last[gid]=tid;covered.add((f,gid))
    return (fp,fn,sw),covered


def run(root, full, gt_root):
    analysis=root/"analysis"
    with (analysis/"official_checks.csv").open(encoding="utf-8-sig") as stream:checks=list(csv.DictReader(stream))
    reference={}
    for m in ("B_ref","A2B","A2B_score020"):
        with (full/"protocols"/(m+".json")).open(encoding="utf-8") as stream:protocol=json.load(stream)
        by_seq={q["sequence"]:q for q in checks if q["method"]==m}
        summed=Counter()
        for seq,n in protocol["sequences"].items():
            gt=gt_root/seq/"gt/gt.txt"
            if not gt.exists():
                gt=gt_root/(seq+".txt")
            pred=full/"trackers"/m/"data"/(seq+".txt")
            assert hashlib.sha256(gt.read_bytes()).hexdigest()==by_seq[seq]["GT_sha256"]
            assert hashlib.sha256(pred.read_bytes()).hexdigest()==protocol["output_prediction_hashes"][seq]
            (fp,fn,ids),coverage=match(mot(gt,n,True),mot(pred,n))
            assert [fp,fn,ids]==[int(by_seq[seq][k]) for k in ("FP","FN","IDSW")],(seq,m)
            if m=="B_ref":reference[seq]=coverage
            summed.update(dict(FP=fp,FN=fn,IDSW=ids,gained_vs_B=len(coverage-reference[seq]),
                               lost_vs_B=len(reference[seq]-coverage)))
        print(m,json.dumps(summed,sort_keys=True))
    print("PASS: 90 prediction files and their CLEAR counts.")


if __name__=="__main__":
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("root",nargs="?",type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument("--run-dir",type=Path,default=Path(__file__).resolve().parents[2]/"full30")
    p.add_argument("--gt-root",type=Path,required=True)
    args=p.parse_args()
    run(args.root,args.run_dir,args.gt_root)
