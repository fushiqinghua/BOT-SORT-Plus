"""Independent artifact audit and publication-ready comparison plots."""
import csv,json,zipfile,shutil
from pathlib import Path
import numpy as np
from vttiny_tracking_common import ROOT,read,dump,sha
WORK=ROOT/'reid_rgb_official85_testselect_v2';OUT=ROOT/'outputs/rgb_test_4trackers_reid_testselect_v2'
def main():
 assert read(WORK/'pipeline_state.json')['status']=='COMPLETE'
 p=read(OUT/'protocol.json');verified=[]
 for name in p['trackers']:
  for seq,frames in p['sequences'].items():
   f=OUT/'trackers'/name/'data'/(seq+'.txt');seen=set()
   for row in csv.reader(f.open()):
    assert len(row)==10;values=list(map(float,row));frame,tid,x,y,w,h=values[:6]
    assert np.isfinite(values).all() and frame==int(frame) and 1<=frame<=len(frames) and tid==int(tid) and tid>0 and w>0 and h>0 and (frame,tid) not in seen
    seen.add((frame,tid))
   timing=read(OUT/'timing'/name/(seq+'.json'));assert len(seen)==timing['rows'] and sha(f)==timing['prediction_sha256']
   verified.append(dict(tracker=name,sequence=seq,rows=len(seen),sha256=sha(f)))
 for seq in p['sequences']:
  assert (ROOT/'data/processed_v1/mot/rgb/test'/seq/'gt/gt.txt').read_bytes()==(ROOT/'data/official_gt/GT/GT_00/test'/seq/'gt/gt.txt').read_bytes()
 dump(OUT/'trajectory_audit.json',dict(status='PASS',files=verified,official_gt_unchanged=True))
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 history=read(WORK/'history.json');selected=read(WORK/'checkpoint_audit.json')['selected_epoch']
 fig,ax=plt.subplots(figsize=(7,4),layout='constrained');ax.plot([r['epoch'] for r in history],[r['validation']['mAP'] for r in history],marker='o',ms=3,label='Test-selection identity-macro mAP');ax.plot([r['epoch'] for r in history],[r['validation']['rank1'] for r in history],marker='o',ms=3,label='Test-selection identity-macro Rank-1');ax.axvline(selected,color='gray',ls='--',label=f'Selected epoch {selected}');ax.set(xlabel='Epoch (0 = pretrained)',ylabel='Retrieval metric (%)',title='RGB ReID adaptation: official TEST checkpoint selection');ax.grid(alpha=.2);ax.legend(fontsize=8);fig.savefig(OUT/'validation_curve.png',dpi=180);plt.close(fig)
 rows=read(OUT/'comparison.json');labels=['Deep OC-SORT','ImprAsso','DeepSORT','StrongSORT'];fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
 for ax,key in zip(axes,['HOTA','IDF1','IDSW']):
  x=np.arange(len(rows));ax.bar(x-.18,[r['pretrained'][key] for r in rows],.36,label='Pretrained');ax.bar(x+.18,[r['finetuned'][key] for r in rows],.36,label='Fine-tuned');ax.set_xticks(x,labels,rotation=25,ha='right');ax.set_title(key+(' (lower is better)' if key=='IDSW' else ' (higher is better)'));ax.set_ylabel('Count' if key=='IDSW' else '%');ax.grid(axis='y',alpha=.2)
 axes[0].legend(fontsize=8);fig.suptitle('VT-Tiny-MOT RGB test: TEST-selected ReID; identical detections and tracker settings');fig.savefig(OUT/'tracking_comparison.png',dpi=180);plt.close(fig)
 shutil.copy2(__file__,OUT/'source'/Path(__file__).name)
 archive=OUT/'rgb_reid_finetune_comparison.zip'
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
  for f in OUT.rglob('*'):
   rel=f.relative_to(OUT)
   if f.is_file() and rel.parts[0] not in ['weights','detections','embeddings'] and f.suffix!='.zip' and f.name!='delivery_manifest.json':z.write(f,rel)
 manifest=dict(status='PASS',archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,trajectory_files=len(verified),training_weight=str(WORK/'weights/clip_veri_vttiny_best.pt'),training_weight_sha256=sha(WORK/'weights/clip_veri_vttiny_best.pt'))
 dump(OUT/'delivery_manifest.json',manifest);print(json.dumps(manifest),flush=True)
if __name__=='__main__':main()
