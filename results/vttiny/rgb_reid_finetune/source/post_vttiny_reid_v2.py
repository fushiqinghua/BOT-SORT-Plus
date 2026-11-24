"""Re-evaluate four appearance trackers with unchanged RGB detection caches."""
import os,sys,time,shutil,traceback,zipfile
from pathlib import Path
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']='4'
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
from vttiny_tracking_common import ROOT,read,dump,sha
WORK=ROOT/'reid_rgb_official85_testselect_v2';SOURCE=ROOT/'outputs/rgb_test_7trackers_v1';OUT=ROOT/'outputs/rgb_test_4trackers_reid_testselect_v2'
METHODS=['deepocsort','imprassoc','deepsort','strongsort']
def worker(name):
 import run_vttiny_rgb_tracking as base
 base.OUT=OUT
 return base.track_method(name)
def main():
 assert read(WORK/'state.json')['status']=='COMPLETE'
 import torch
 before=torch.load(WORK/'weights/clip_veri_pretrained.pt',map_location='cpu',weights_only=False)
 before=before.get('state_dict',before);before={k.removeprefix('module.'):v for k,v in before.items()}
 after=torch.load(WORK/'weights/clip_veri_vttiny_best.pt',map_location='cpu',weights_only=False)
 changed=[]
 for k,v in after['state_dict'].items():
  if k.startswith(('image_encoder.','bottleneck.','bottleneck_proj.')):
   assert k in before and v.shape==before[k].shape
   if not torch.equal(v,before[k].to(v.dtype)):
    assert k.startswith(('image_encoder.transformer.resblocks.10.','image_encoder.transformer.resblocks.11.','image_encoder.ln_post.','bottleneck.','bottleneck_proj.')) or k=='image_encoder.proj',k
    changed.append(k)
 assert any(k.startswith('image_encoder.transformer.resblocks.') for k in changed)
 dump(WORK/'checkpoint_audit.json',dict(status='PASS',selected_epoch=after['epoch'],selected_validation=after['validation'],changed_inference_keys=changed,frozen_early_visual_layers_unchanged=True))
 del before,after
 OUT.mkdir(exist_ok=False);(OUT/'weights').mkdir();(OUT/'detections').mkdir()
 original={str(p.relative_to(SOURCE)):sha(p) for pattern in ['protocol.json','baseline_results.json','weights/*.pt','detections/*.json','trackers/*/data/*.txt'] for p in SOURCE.glob(pattern)}
 dump(OUT/'baseline_sha256_before.json',original)
 p=read(SOURCE/'protocol.json');p['parent_run']=str(SOURCE);p['reid_training']=str(WORK);p['evaluation_status']='TEST_SET_SELECTED_NOT_HELD_OUT';p['trackers']={k:p['trackers'][k] for k in METHODS}
 p['notes']=[n for n in p['notes'] if 'NOT fine-tuned' not in n and 'All seven' not in n]
 p['notes']+=['Only change versus baseline: shared ReID weights selected on OFFICIAL TEST temporal retrieval (test-selected, not held-out). Detector weights/cache and all tracker parameters unchanged.','RGB ReID visual encoder adapted using cleaned train identities, using all official 85 training sequences before identity/crop screening; original parent-video overlap retained.']
 (OUT/'weights/rgb_best.pt').symlink_to(SOURCE/'weights/rgb_best.pt')
 shutil.copy2(WORK/'weights/clip_veri_vttiny_best.pt',OUT/'weights/clip_veri.pt');p['reid_sha256']=sha(OUT/'weights/clip_veri.pt')
 for f in (SOURCE/'detections').glob('*.json'):(OUT/'detections'/f.name).symlink_to(f)
 dump(OUT/'protocol.json',p)
 import run_vttiny_rgb_tracking as base
 base.OUT=OUT
 dump(OUT/'state.json',dict(status='RUNNING',stage='embeddings',updated=time.time()))
 base.embeddings()
 metadata=read(OUT/'reid_encoder.json');metadata['training_domain']='VeRi initialization + VT-Tiny-MOT cleaned train RGB identity adaptation';metadata['finetuning']=str(WORK);dump(OUT/'reid_encoder.json',metadata)
 from concurrent.futures import ProcessPoolExecutor
 import multiprocessing
 dump(OUT/'state.json',dict(status='RUNNING',stage='tracking',updated=time.time()))
 with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:timings=list(pool.map(worker,METHODS))
 dump(OUT/'tracking_timing.json',[r for group in timings for r in group])
 # Prediction audit hooks are process-local. Evaluation runs in a fresh process.
 import subprocess
 subprocess.run([sys.executable,str(ROOT/'tools/evaluate_vttiny.py'),'--modality','rgb','--split','test','--trackers-folder',str(OUT/'trackers'),'--trackers',*METHODS,'--output',str(OUT/'metrics')],check=True)
 baseline={r['tracker']:r for r in read(SOURCE/'baseline_results.json')}
 current={r['tracker']:r for r in read(OUT/'metrics/summary.json')['results'] if r['sequence']=='COMBINED_SEQ'}
 comparison=[]
 lines=['# RGB ReID 微调前后对比（官方测试集选模）','','本实验按用户确认使用官方测试集选择权重并在同一测试集汇报结果，属于测试集选模，不是独立留出测试。','相同 RGB 检测权重、相同检测缓存、相同跟踪参数，仅替换共享的 ReID 权重。官方测试集 30 段、12,590 帧，TrackEval 联合汇总。','','|方法|HOTA 原→微调|ΔHOTA|IDF1 原→微调|ΔIDF1|MOTA 原→微调|IDSW 原→微调|','|---|---:|---:|---:|---:|---:|---:|']
 for name in METHODS:
  b=baseline[name];c=current[name];delta={k:c[k]-b[k] for k in ['HOTA','DetA','AssA','IDF1','MOTA','FP','FN','IDSW']}
  comparison.append(dict(tracker=name,pretrained=b,finetuned=c,delta=delta))
  lines.append(f"|{base.NAMES[name]}|{b['HOTA']:.2f} → {c['HOTA']:.2f}|{delta['HOTA']:+.2f}|{b['IDF1']:.2f} → {c['IDF1']:.2f}|{delta['IDF1']:+.2f}|{b['MOTA']:.2f} → {c['MOTA']:.2f}|{b['IDSW']} → {c['IDSW']}|")
 assert original=={s:sha(SOURCE/s) for s in original},'Original baseline changed'
 for name in METHODS:assert len(list((OUT/'trackers'/name/'data').glob('*.txt')))==30
 history=read(WORK/'history.json');best=read(WORK/'state.json')
 lines+=['','训练配置与清理统计见 training/config.json、training/data_audit.json；逐轮验证见 training/history.json。微调权重按官方测试集身份检索 mAP 选择；下表跟踪指标也来自同一官方测试集。','',f"原始编码器测试集检索 mAP：{history[0]['validation']['mAP']:.2f}；微调最佳测试集检索 mAP：{best['best_validation_mAP']:.2f}。检索采用同序列所有目标、至少间隔15帧的身份宏平均检索，不是跨摄像头 ReID。",'', '训练20轮、每轮200个身份采样批次；初始权重为CLIP-VeRi，视觉编码器末两层与输出相关层适配。StrongSORT仍为在线版本，不含AFLink/GSI。完整训练与评估来源保留在服务器。']
 (OUT/'comparison.md').write_text('\n'.join(lines),encoding='utf8');dump(OUT/'comparison.json',comparison)
 (OUT/'training').mkdir()
 for f in ['config.json','data_audit.json','preflight.json','history.json','state.json','gpu_binding_audit.json','checkpoint_audit.json']:shutil.copy2(WORK/f,OUT/'training'/f)
 shutil.copytree(WORK/'source',OUT/'source')
 dump(OUT/'integrity.json',dict(status='PASS',evaluation_status='TEST_SET_SELECTED_NOT_HELD_OUT',original_baseline_unchanged=True,detector_weight_sha256=p['weight_sha256'],reid_sha256=p['reid_sha256'],frames_per_method=12590,sequences=30,trajectory_files=120))
 dump(OUT/'state.json',dict(status='COMPLETE',updated=time.time()))
 with zipfile.ZipFile(OUT/'rgb_reid_finetune_comparison.zip','w',zipfile.ZIP_DEFLATED) as z:
  for f in OUT.rglob('*'):
   rel=f.relative_to(OUT)
   if f.is_file() and rel.parts[0] not in ['weights','detections','embeddings'] and f.suffix!='.zip':z.write(f,rel)
 print((OUT/'comparison.md').read_text(),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  dump(WORK/'post_failure.json',dict(status='FAILED',traceback=traceback.format_exc(),updated=time.time()));raise
