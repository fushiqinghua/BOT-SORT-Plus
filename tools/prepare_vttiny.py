"""Build paired YOLO inputs and MOT evaluation data without altering official files."""
import collections, hashlib, json, random, shutil
from pathlib import Path
import yaml
from audit_vttiny import ROOT, RAW, NAMES, dump
from vttiny_tracking_common import FRAMEWORK

OUT=ROOT/'data/processed_v1'
def write(path,text):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text,encoding='utf8')
def link(path,target):
 path.parent.mkdir(parents=True,exist_ok=True)
 if path.is_symlink():assert path.resolve()==target.resolve()
 elif path.exists():raise FileExistsError(path)
 else:path.symlink_to(target.resolve())
def parent(name):return name.rsplit('_',1)[0]

def choose_split(sequences,checks):
 train=[s for s in sequences if s['official_split']=='train'];test=[s for s in sequences if s['official_split']=='test']
 bad={s for m in checks['train_conflicts'].values() for s in m}
 excluded={parent(s) for s in bad}|{parent(s['sequence']) for s in test}
 groups=collections.defaultdict(list)
 for seq in train:groups[parent(seq['sequence'])].append(seq)
 candidates=sorted(set(groups)-excluded);rng=random.Random(0);k=round(len(groups)*0.2);best=None
 totals={(m,c):sum(s[m]['classes'].get(c,0) for s in train) for m in ['rgb','ir'] for c in NAMES}
 totalframes=sum(s['frames'] for s in train)
 for _ in range(5000):
  selected=set(rng.sample(candidates,k));val=[s for g in selected for s in groups[g]]
  counts={(m,c):sum(s[m]['classes'].get(c,0) for s in val) for m,c in totals}
  if any(v==0 or v==totals[key] for key,v in counts.items()):continue
  score=sum(abs(counts[key]/totals[key]-.2) for key in totals)+2*abs(sum(s['frames'] for s in val)/totalframes-.2)
  if best is None or score<best[0]:best=score,sorted(selected)
 assert best is not None,'No valid grouped validation split'
 valparents=set(best[1]);split={s['sequence']:('test' if s['official_split']=='test' else ('val' if parent(s['sequence']) in valparents else 'train')) for s in sequences}
 info={'seed':0,'target_validation_fraction':.2,'selection':'5000 seeded parent-video group subsets, minimize class/frame fraction deviation using only training labels; require seven classes in train and val for both modalities','validation_parent_groups':best[1],'validation_ineligible_parent_groups':sorted(excluded),'reason':'Keep test-parent groups outside validation; reserve ambiguous training identity groups for detector training, not validation tracking GT. No test scores or model results used.','splits':{part:sorted(s for s,v in split.items() if v==part) for part in ['train','val','test']}}
 return split,info

def main():
 audit=json.loads((ROOT/'reports/annotation_audit.json').read_text());assert audit['status']=='AUDITED' and not audit['images']['invalid']
 sequences=json.loads((ROOT/'reports/sequences.json').read_text());checks=json.loads((ROOT/'reports/protocol_checks.json').read_text())
 split,splitinfo=choose_split(sequences,checks);dump(OUT/'splits.json',splitinfo)
 stats=collections.defaultdict(collections.Counter);issues=collections.Counter();motstats={};allseq={s['sequence']:s for s in sequences}
 edits=(ROOT/'reports/detection_label_changes.jsonl').open('w')
 for official in ['train','test']:
  for mode,folder in [('rgb','00'),('ir','01')]:
   d=json.loads((RAW/'annotations'/f'instances_{folder}_{official}2017.json').read_text());images={i['id']:i for i in d['images']};anns=collections.defaultdict(list)
   for a in d['annotations']:anns[a['image_id']].append(a)
   valgt=collections.defaultdict(list)
   for im in images.values():
    seq=im['file_name'].split('/')[0];part=split[seq];fid=im['frame_id']+1;stem=f'{seq}_{fid:06d}';rows=[];seen=set();source=RAW/(official+'2017')/im['file_name']
    for a in anns[im['id']]:
     x,y,w,h=a['bbox'];x1=max(0.,min(640.,x));y1=max(0.,min(512.,y));x2=max(0.,min(640.,x+w));y2=max(0.,min(512.,y+h));box=[x1,y1,x2-x1,y2-y1];key=(a['category_id'],*box)
     action=[]
     if box!=a['bbox']:action.append('clip_to_image');issues[official+'/'+mode+'/clipped']+=1
     if box[2]<=0 or box[3]<=0:action.append('drop_empty_after_clip');issues[official+'/'+mode+'/empty']+=1
     elif key in seen:action.append('deduplicate_identical_detection');issues[official+'/'+mode+'/identical']+=1
     else:
      seen.add(key);rows.append(f'{key[0]} {(x1+x2)/1280:.10f} {(y1+y2)/1024:.10f} {(x2-x1)/640:.10f} {(y2-y1)/512:.10f}');stats[mode+'/'+part][NAMES[key[0]]]+=1
     if action:edits.write(json.dumps({'official_split':official,'modality':mode,'image':im['file_name'],'annotation_id':a['id'],'old_bbox':a['bbox'],'new_bbox':box,'actions':action})+'\n')
     if part=='val':valgt[seq].append((fid,int(a['track_id']),*a['bbox'],1,1,1))
    label=OUT/'detection'/mode/'labels'/part/(stem+'.txt');write(label,'\n'.join(rows)+ ('\n' if rows else ''))
    link(OUT/'detection'/mode/'images'/part/(stem+'.jpg'),source)
    link(OUT/'detection/rgbt'/('images' if mode=='rgb' else 'images_ir')/part/(stem+'.jpg'),source)
    if mode=='rgb':link(OUT/'detection/rgbt/labels'/part/(stem+'.txt'),label)
    stats[mode+'/'+part]['images']+=1;stats[mode+'/'+part]['boxes']+=len(rows)
   for seq in [s for s in allseq if split[s]=='test' and official=='test']+sorted(valgt):
    part=split[seq];base=OUT/'mot'/mode/part/seq;meta=allseq[seq]
    if part=='test':
     src=ROOT/'data/official_gt/GT'/('GT_'+folder)/'test'/seq/'gt/gt.txt';data=src.read_text();write(base/'gt/gt.txt',data)
     assert hashlib.sha256((base/'gt/gt.txt').read_bytes()).hexdigest()==hashlib.sha256(src.read_bytes()).hexdigest()
     rows=[list(map(float,line.split(','))) for line in data.splitlines() if line.strip()]
    else:
     rows=sorted(valgt[seq]);write(base/'gt/gt.txt',''.join(','.join(map(str,row))+'\n' for row in rows))
    assert len({(r[0],r[1]) for r in rows})==len(rows),(mode,seq)
    assert all(1<=r[0]<=meta['frames'] and r[4]>0 and r[5]>0 for r in rows)
    for idx,name in enumerate(meta[mode]['filenames'],1):link(base/'img1'/f'{idx:06d}.jpg',Path(meta[mode]['directory'])/name)
    write(base/'seqinfo.ini',f'[Sequence]\nname={seq}\nimDir=img1\nframeRate=15\nseqLength={meta["frames"]}\nimWidth=640\nimHeight=512\nimExt=.jpg\n')
    motstats[mode+'/'+part+'/'+seq]={'frames':meta['frames'],'gt_rows':len(rows),'source':'official GT.zip byte-for-byte' if part=='test' else 'COCO raw bbox and track_id; no clipping; collapse classes only for class-agnostic MOT metrics'}
   print('PREPARED',official,mode,flush=True)
 edits.close()
 for part in ['train','val','test']:
  with (OUT/(part+'_pairs.jsonl')).open('w') as out:
   for s in sorted(sequences,key=lambda s:s['sequence']):
    if split[s['sequence']]!=part:continue
    for idx,(rgb,ir) in enumerate(zip(s['rgb']['filenames'],s['ir']['filenames']),1):
     out.write(json.dumps({'sequence':s['sequence'],'split':part,'official_split':s['official_split'],'frame':idx,'source_filename':rgb,'source_frame':s['rgb']['source_frame_ids'][idx-1],'rgb':str(Path(s['rgb']['directory'])/rgb),'ir':str(Path(s['ir']['directory'])/ir)},ensure_ascii=False)+'\n')
  for mode in ['rgb','ir']:write(OUT/'mot/seqmaps'/f'{mode}_{part}.txt','name\n'+'\n'.join(splitinfo['splits'][part])+'\n') if part!='train' else None
 for mode in ['rgb','ir','rgbt']:
  config={'path':str(OUT/'detection'/mode),'train':'images/train','val':'images/val','test':'images/test','channels':1 if mode=='ir' else 3,'names':dict(enumerate(NAMES))}
  if mode=='rgbt':config.update(modality={'rgb':'images','x':'images_ir'},modality_used=['rgb','x'],modality_ch={'rgb':3,'x':1},Xch=1)
  write(ROOT/'configs'/(mode+'_data.yaml'),yaml.safe_dump(config,allow_unicode=True,sort_keys=False))
 fw=FRAMEWORK
 for name,source in [('yolo11s_7class.yaml','ultralytics/cfg/models/11/yolo11.yaml'),('yolo11s_rgbt_p345_7class.yaml','ultralytics/cfg/models/mm/rsf/yolo11s-mm-rsf-baseline-p345.yaml')]:
  cfg=yaml.safe_load((fw/source).read_text());cfg.update(nc=7,scale='s');write(ROOT/'configs'/name,yaml.safe_dump(cfg,sort_keys=False))
 summary={'status':'PREPARED_PENDING_VERIFICATION','dataset':'VT-Tiny-MOT','classes':NAMES,'official_counts':{p:{'sequences':sum(s['official_split']==p for s in sequences),'pairs':sum(s['frames'] for s in sequences if s['official_split']==p)} for p in ['train','test']},'split_counts':{p:{'sequences':len(splitinfo['splits'][p]),'pairs':sum(s['frames'] for s in sequences if split[s['sequence']]==p)} for p in ['train','val','test']},'detection':{k:dict(v) for k,v in stats.items()},'label_changes':dict(issues),'mot':motstats,'policy':{'raw':'unchanged','detection':'seven classes; RGB own labels, IR own labels; RGBT RGB reference. Clip to image for YOLO only; drop zero-intersection and identical detections with explicit log; do not choose between conflicting identity annotations.','tracking':'class-agnostic official test GT; COMBINED_SEQ over all sequences separately for RGB and IR coordinate references. All seven detector classes contribute; GT category=1 means object, not pedestrian.','test':'Original 30-sequence test fixed; never used as validation; official GT preserved byte-for-byte including empty RGB sequence.','validation':'Internal grouped holdout from original train; same split for RGB, IR and RGBT. See splits.json for fully specified construction.','train_mot':'No MOT training GT exported: ambiguous raw IDs require review before supervised ReID/association training. Detector training uses identity-free boxes.','class_aware_mot':'Not enabled: official GT has no semantic classes and DJI_0073_3 boxes differ from COCO; cannot assign labels by exact matching for all rows.','weights':'Existing M3OT one-class weights are not final seven-class models; new training required.','parent_overlap':'Official train/test share 8 parent clip names; official split preserved and overlap disclosed. Internal train/val parent groups are disjoint.'}}
 dump(ROOT/'reports/preparation_summary.json',summary);print(json.dumps({'splits':summary['split_counts'],'changes':summary['label_changes']}),flush=True)

if __name__=='__main__':main()
