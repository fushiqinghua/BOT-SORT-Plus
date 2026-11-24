import hashlib,json,os
from pathlib import Path
ROOT=Path('VTTINY_ROOT')
OLD=ROOT.parent/'M3OT_MOT_Baseline_v0_1'
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def dump(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(path)
def forbid_gt(event,args):
 if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
  s=os.fsdecode(args[0]).lower().replace('\\','/')
  if s.endswith('/gt.txt') or '/annotations/' in s or s.endswith('/instances_test2017.json'):
   raise PermissionError('Ground truth is forbidden during prediction: '+s)
