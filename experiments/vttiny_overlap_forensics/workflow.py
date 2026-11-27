"""Replay the frozen two versions, controlled interventions, and candidate repairs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
THIS=ROOT
JOINT=ROOT.with_name('b_gmc_evidence_overlap_v1')
os.environ['VT_INTEGRATIONS']=str(ROOT.parent)
sys.path.insert(0,str(JOINT))
from joint_settings import FIX,COMBINED,environment,read
FOCUS=('DJI_0028_3','DJI_0061_2','DJI_0169_3')


def replay(mode,gpu,full=False):
    name=FIX if mode=='fixed' else COMBINED
    root=ROOT/'runs'/mode
    p=root/'full'/'protocols'/(name+'.json')
    if p.exists() and read(p)['status']=='COMPLETE':return
    if root.exists():root.rename(root.with_name(root.name+'-partial-'+str(time.time_ns())))
    with (ROOT/(mode+'.log')).open('w') as log:
        env=dict(environment(name,gpu),VT_COMBO_ROOT=str(root/'experiment'),
                 PYTHONPATH=str(JOINT)+os.pathsep+environment(name,gpu)['PYTHONPATH'])
        args=[sys.executable,'-Bu',str(THIS/'trace_replay.py'),mode]+([] if full else list(FOCUS))
        subprocess.run(args,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('job')
    parser.add_argument('--gpu',type=int,default=5)
    args=parser.parse_args()
    if args.job=='audit':
        subprocess.run([sys.executable,'-Bu',str(ROOT/'audit.py')],env=environment('native',args.gpu),check=True)
    elif args.job.startswith('score_'):
        subprocess.run([sys.executable,'-Bu',str(ROOT/'fullscore.py'),args.job.removeprefix('score_')],
                       env=environment('native',args.gpu),check=True)
    elif args.job=='report':
        subprocess.run([sys.executable,'-Bu',str(ROOT/'report.py')],check=True)
    else:
        modes=('fixed','combo') if args.job=='replay' else tuple(args.job.split(','))
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(lambda m:replay(m,args.gpu,m.startswith('full_')),modes))
    print('COMPLETE',args.job,flush=True)


if __name__=='__main__':main()
