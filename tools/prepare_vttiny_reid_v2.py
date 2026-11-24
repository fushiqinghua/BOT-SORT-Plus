"""Build official85 training crops and official30 TEST checkpoint-selection crops."""
import json,math,collections,shutil
from pathlib import Path
import numpy as np,cv2
from vttiny_tracking_common import ROOT,OLD,read,dump,sha
WORK=ROOT/'reid_rgb_official85_testselect_v2'
def parent(seq):return seq.rsplit('_',1)[0]
def main():
 WORK.mkdir(exist_ok=False);(WORK/'source').mkdir()
 splits=read(ROOT/'data/processed_v1/splits.json')['splits']
 test_parents={parent(s) for s in splits['test']}
 train=set(splits['train'])|set(splits['val']);val=set(splits['test'])
 excluded=[]
 assert len(train)==85 and len(val)==30 and not train&val
 overlapping_parents=sorted({parent(s) for s in train}&test_parents)
 annotation=ROOT/'data/raw/VT-Tiny-MOT/annotations/instances_00_train2017.json'
 coco=read(annotation);images={x['id']:x for x in coco['images']};tracks=collections.defaultdict(list)
 counts=collections.Counter();bad=set();seen={};classes=collections.defaultdict(set)
 for a in coco['annotations']:
  im=images[a['image_id']];seq=im['file_name'].split('/')[0]
  if seq not in train|val:continue
  ident=(seq,str(a['track_id']));classes[ident].add(a['category_id'])
  key=(ident,im['frame_id']);signature=(tuple(a['bbox']),a['category_id'])
  if key in seen:
   if seen[key]==signature:counts['exact_duplicate_annotations']+=1;continue
   bad.add(ident);counts['ambiguous_frame_annotations']+=1
  seen[key]=signature
  if a.get('ignore',0) or a.get('iscrowd',0):bad.add(ident);counts['flagged_annotations']+=1
  tracks[ident].append(dict(sequence=seq,identity=seq+':'+str(a['track_id']),frame=im['frame_id']+1,category=a['category_id'],bbox=a['bbox'],image=str(ROOT/'data/raw/VT-Tiny-MOT/train2017'/im['file_name'])))
 bad|={k for k,v in classes.items() if len(v)!=1}
 # Test labels are explicitly authorized for checkpoint selection; never used in optimization.
 frame_map={}
 for line in (ROOT/'data/processed_v1/test_pairs.jsonl').read_text().splitlines():
  r=json.loads(line);frame_map[(r['sequence'],r['frame'])]=r['rgb']
 test_gt_sha={}
 for seq in sorted(val):
  gt=ROOT/'data/official_gt/GT/GT_00/test'/seq/'gt/gt.txt';test_gt_sha[seq]=sha(gt)
  assert gt.read_bytes()==(ROOT/'data/processed_v1/mot/rgb/test'/seq/'gt/gt.txt').read_bytes()
  unique=set()
  for line in gt.read_text().splitlines():
   if not line.strip():continue
   a=list(map(float,line.split(',')));frame,tid=int(a[0]),int(a[1]);assert (frame,tid) not in unique and a[6]>0
   unique.add((frame,tid));identity=(seq,str(tid))
   tracks[identity].append(dict(sequence=seq,identity=seq+':'+str(tid),frame=frame,category=-1,bbox=a[2:6],image=frame_map[(seq,frame)]))
 result={};excluded_ids=[]
 for split,seqs in [('train',train),('val',val)]:
  records=[];identities=[]
  for key,rows in sorted(tracks.items()):
   if key[0] not in seqs:continue
   if key in bad:excluded_ids.append(dict(identity=list(key),reason='ambiguous repeated ID / class change / ignored'));continue
   valid=[]
   for row in sorted(rows,key=lambda x:x['frame']):
    x,y,w,h=row['bbox'];x1,y1,x2,y2=np.rint([x,y,x+w,y+h]).astype(int)
    x1,y1,x2,y2=max(0,x1),max(0,y1),min(640,x2),min(512,y2)
    if min(x2-x1,y2-y1)<4:counts[split+'_crop_side_below_4px']+=1;continue
    row['xyxy']=[int(x1),int(y1),int(x2),int(y2)];valid.append(row)
   if len(valid)<4 or valid[-1]['frame']-valid[0]['frame']<15:
    counts[split+'_short_tracks']+=1;continue
   selected=np.unique(np.linspace(0,len(valid)-1,min(32 if split=='train' else 16,len(valid))).astype(int))
   pid=len(identities);identities.append(dict(identity=valid[0]['identity'],sequence=key[0],category=valid[0]['category'],available_frames=len(valid),sampled_frames=len(selected)))
   for idx in selected:records.append(dict(valid[idx],pid=pid))
  result[split]=(records,identities)
  assert identities and records
  folder=WORK/'crops'/split;folder.mkdir(parents=True)
  byimage=collections.defaultdict(list)
  for i,row in enumerate(records):row['crop']=str(folder/f'{i:07d}.png');byimage[row['image']].append(row)
  for path,rows in byimage.items():
   image=cv2.imread(path);assert image is not None
   for row in rows:
    x1,y1,x2,y2=row['xyxy'];assert cv2.imwrite(row['crop'],image[y1:y2,x1:x2])
  dump(WORK/(split+'.json'),dict(records=records,identities=identities))
  print('PREPARED',split,'sequences',len({r['sequence'] for r in records}),'identities',len(identities),'crops',len(records),flush=True)
 report=dict(annotation_sha256=sha(annotation),test_annotation_read=True,test_gt_sha256=test_gt_sha,input_train_sequences=85,input_selection_sequences=30,train_test_overlapping_parent_groups=overlapping_parents,evaluation_status="TEST_SET_SELECTED_NOT_HELD_OUT",excluded_test_parent_sequences=excluded,excluded_identities=excluded_ids,cleaning_counts=dict(counts),split_stats={})
 for split,(records,ids) in result.items():
  report['split_stats'][split]=dict(sequences=len({r['sequence'] for r in records}),identities=len(ids),crops=len(records),identities_per_class=dict(collections.Counter(str(r['category']) for r in ids)),crop_min_sides_percentiles=np.percentile([min(r['xyxy'][2]-r['xyxy'][0],r['xyxy'][3]-r['xyxy'][1]) for r in records],[0,25,50,75,100]).tolist())
 source=ROOT/'outputs/rgb_test_7trackers_v1/weights/clip_veri.pt';(WORK/'weights').mkdir();shutil.copy2(source,WORK/'weights/clip_veri_pretrained.pt')
 report['source_reid_sha256']=sha(source)
 dump(WORK/'data_audit.json',report)
 config=dict(seed=0,physical_gpu=4,epochs=20,steps_per_epoch=200,identities_per_batch=16,images_per_identity=4,input_hw=[256,256],feature_dimension=1280,lr=1e-5,head_lr=1e-3,weight_decay=1e-4,triplet_margin=.3,label_smoothing=.1,workers=8,precision='float32',freeze='ViT first 10 blocks and input embedding; train last 2 blocks, ln_post, projection, BN affine and new ID classifier',checkpoint_selection='highest OFFICIAL TEST macro-per-identity within-sequence all-object temporal retrieval mAP among epochs1..20; epoch0 reference only',evaluation_status='TEST_SET_SELECTED_NOT_HELD_OUT',internal_validation_split=False,selection_data='official 30 test sequences, RGB official GT; no optimizer updates on test crops',validation_min_frame_gap=15,train_crop_limit_per_identity=32,validation_crop_limit_per_identity=16,min_crop_side=4,augmentation='horizontal flip p0.5; mild brightness/contrast; no mosaic',training_method='supervised visual-only identity adaptation, not original CLIP-ReID two-stage prompt training',train_manifest_sha256=sha(WORK/'train.json'),val_manifest_sha256=sha(WORK/'val.json'),source_weight_sha256=sha(source))
 dump(WORK/'config.json',config)
 for name in ['prepare_vttiny_reid_v2.py','train_vttiny_reid_v2.py','post_vttiny_reid_v2.py','vttiny_tracking_common.py','vttiny_reid_adapter.py']:
  shutil.copy2(ROOT/'tools'/name,WORK/'source'/name)
 print(json.dumps(report,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
