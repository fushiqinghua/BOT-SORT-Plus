"""Audit unmodified official VT-Tiny-MOT; never rewrite raw annotations."""
import collections, concurrent.futures, hashlib, json, math, zipfile
from pathlib import Path
import cv2

from vttiny_tracking_common import ROOT
RAW=ROOT/'data/raw/VT-Tiny-MOT'
NAMES=['ship','car','cyclist','pedestrian','bus','drone','plane']

def dump(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')

def check_image(path):
 im=cv2.imread(str(path)); h=hashlib.sha256(path.read_bytes()).hexdigest()
 return str(path.relative_to(RAW)), None if im is None else list(im.shape), h

def main():
 report={'datasets':{},'errors':[],'warnings':[]}; records={}; checks={}; all_images=[]; names_by_split={}
 for split in ('train','test'):
  for mod,folder in [('rgb','00'),('ir','01')]:
   p=RAW/'annotations'/f'instances_{folder}_{split}2017.json'; d=json.loads(p.read_text())
   assert [c['name'] for c in d['categories']]==NAMES and [c['id'] for c in d['categories']]==list(range(7))
   images={im['id']:im for im in d['images']}; assert len(images)==len(d['images'])
   vids={v['id']:v for v in d['videos']}; seqs=collections.defaultdict(list); gt=collections.defaultdict(list)
   annids=set(); seen=set(); classes=collections.Counter(); flags=collections.Counter(); anomalies=collections.Counter(); examples=collections.defaultdict(list); trackcats=collections.defaultdict(set)
   for im in images.values():
    seq,modal,name=im['file_name'].split('/'); assert modal==folder
    assert vids[im['video_id']]['name']==seq+'/'+folder
    assert im['width']==640 and im['height']==512
    assert int(Path(name).stem)==im['mot_frame_id']
    path=RAW/(split+'2017')/im['file_name']; assert path.is_file(),path
    seqs[seq].append(im); all_images.append(path)
   for ann in d['annotations']:
    assert ann['id'] not in annids; annids.add(ann['id'])
    im=images[ann['image_id']]; seq=im['file_name'].split('/')[0]
    frame=im['frame_id']+1; tid=int(ann['track_id']); cls=ann['category_id']; box=ann['bbox']; x,y,w,h=box
    assert cls in range(7) and all(math.isfinite(v) for v in box)
    classes[NAMES[cls]]+=1; flags[str((ann.get('iscrowd',0),ann.get('ignore',0)))]+=1
    for issue,condition in [('nonpositive_box',w<=0 or h<=0),('out_of_bounds',x<0 or y<0 or x+w>640 or y+h>512),('duplicate_frame_id',(seq,frame,tid) in seen)]:
     if condition:
      anomalies[issue]+=1
      if len(examples[issue])<20:examples[issue].append({'sequence':seq,'image':im['file_name'],'annotation':ann})
    seen.add((seq,frame,tid)); trackcats[(seq,tid)].add(cls)
    gt[seq].append((frame,tid,*box,cls))
   changes=[(seq,tid,sorted(cats)) for (seq,tid),cats in trackcats.items() if len(cats)>1]
   if changes:anomalies['track_class_changes']=len(changes);examples['track_class_changes']=changes[:20]
   for seq,ims in seqs.items():
    ims.sort(key=lambda im:im['frame_id']); assert [im['frame_id'] for im in ims]==list(range(len(ims))),seq
    actual=sorted(p.name for p in (RAW/(split+'2017')/seq/folder).glob('*.jpg'))
    indexed=sorted(Path(im['file_name']).name for im in ims); assert actual==indexed,(seq,folder,'unindexed images')
    key=split+'/'+seq
    item=records.setdefault(key,{'sequence':seq,'official_split':split,'frames':len(ims),'fps':15,'width':640,'height':512})
    item[mod]={'directory':str(RAW/(split+'2017')/seq/folder),'filenames':[Path(im['file_name']).name for im in ims],'source_frame_ids':[im['mot_frame_id'] for im in ims],'annotations':len(gt[seq]),'classes':dict(collections.Counter(NAMES[r[-1]] for r in gt[seq]))}
    if mod=='ir':assert item['rgb']['filenames']==item['ir']['filenames']
   names_by_split[split]=set(seqs)
   report['datasets'][split+'/'+mod]={'sequences':len(seqs),'images':len(images),'annotations':len(d['annotations']),'classes':dict(classes),'flags':dict(flags),'anomalies':dict(anomalies),'examples':dict(examples),'tracks':len(trackcats),'annotation_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
   if split=='test':checks[folder]=gt
   print('ANNOTATIONS',split,mod,report['datasets'][split+'/'+mod]['anomalies'],flush=True)
 assert not names_by_split['train']&names_by_split['test']
 report['official_split_sequence_overlap']=[]
 # Assess parent clips without changing official test membership.
 parent=lambda s:s.rsplit('_',1)[0]
 report['parent_clip_overlap']=sorted({parent(s) for s in names_by_split['train']}&{parent(s) for s in names_by_split['test']})
 gtzip=next((ROOT/'external').glob('HGTMT-*/GT.zip')); reference={}
 with zipfile.ZipFile(gtzip) as z:
  for folder,seqs in checks.items():
   for seq,rows in seqs.items():
    name=f'GT/GT_{folder}/test/{seq}/gt/gt.txt'; raw=z.read(name)
    out=ROOT/'data/official_gt'/name;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(raw)
    official=[tuple(float(v) for v in line.split(',')) for line in raw.decode().splitlines() if line.strip()]
    compare=lambda rs:collections.Counter((int(r[0]),*r[2:6]) for r in rs)
    match=compare(official)==compare(rows)
    reference[folder+'/'+seq]={'official_rows':len(official),'coco_rows':len(rows),'frame_bbox_multiset_matches':match,'sha256':hashlib.sha256(raw).hexdigest(),'class_column':sorted({r[7] for r in official}),'first_frame':min((r[0] for r in official),default=None),'last_frame':max((r[0] for r in official),default=None)}
 report['official_gt_comparison']=reference
 dump(ROOT/'reports/annotation_audit.json',report);dump(ROOT/'reports/sequences.json',list(records.values()))
 cv2.setNumThreads(1);shapes=collections.Counter();bad=[];hash_splits=collections.defaultdict(set);n=0
 with (ROOT/'reports/image_hashes.jsonl').open('w') as out, concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  for path,shape,sha in pool.map(check_image,all_images):
   n+=1;shapes[str(shape)]+=1;hash_splits[sha].add(path.split('/')[0]);out.write(json.dumps({'path':path,'shape':shape,'sha256':sha})+'\n')
   if shape!=[512,640,3]:bad.append({'path':path,'shape':shape})
   if n%10000==0:print('DECODED',n,len(all_images),flush=True)
 report['images']={'decoded':n,'shapes':dict(shapes),'invalid':bad,'cross_split_duplicate_hashes':sum(len(s)>1 for s in hash_splits.values())}
 report['status']='AUDITED';dump(ROOT/'reports/annotation_audit.json',report)
 print(json.dumps({'status':report['status'],'images':report['images'],'official_gt_mismatches':[k for k,v in reference.items() if not v['frame_bbox_multiset_matches']],'parent_clip_overlap':report['parent_clip_overlap']}),flush=True)

if __name__=='__main__':main()
