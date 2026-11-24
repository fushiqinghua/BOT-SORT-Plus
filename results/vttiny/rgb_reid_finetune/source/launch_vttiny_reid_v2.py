"""Run training then one frozen-cache evaluation; preserve both original and new runs."""
import os,sys,time,subprocess,traceback
from vttiny_tracking_common import ROOT,dump
WORK=ROOT/'reid_rgb_official85_testselect_v2'
if __name__=='__main__':
 try:
  for stage,script in [('training','train_vttiny_reid_v2.py'),('reevaluation','post_vttiny_reid_v2.py')]:
   dump(WORK/'pipeline_state.json',dict(status='RUNNING',stage=stage,pid=os.getpid(),updated=time.time()))
   subprocess.run([sys.executable,'-u',str(ROOT/'tools'/script)],check=True)
  dump(WORK/'pipeline_state.json',dict(status='COMPLETE',updated=time.time()))
 except Exception:
  dump(WORK/'pipeline_state.json',dict(status='FAILED',stage=stage,traceback=traceback.format_exc(),updated=time.time()));raise
