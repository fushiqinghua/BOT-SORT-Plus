"""Launch isolated native-BoT-SORT experiments and validate them with official inputs."""
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from settings import *


def run(script,*args):
    subprocess.run([sys.executable,'-Bu',str(script),*map(str,args)],check=True,env=environment(),cwd=ROOT)


def arm(name):
    with (ROOT/('arm_'+name+'.log')).open('a') as log:
        for stage in ('pilot','full'):
            prot=ROOT/stage/'protocols'/(name+'.json')
            if prot.exists() and read(prot)['status']=='COMPLETE':continue
            backup=ROOT/'attempts'/(stage+'-'+name+'-'+str(time.time_ns()))
            for part in [prot,*(ROOT/stage/f/name for f in ('trackers','timing','diagnostics345'))]:
                if part.exists():
                    dest=backup/part.relative_to(ROOT/stage);dest.parent.mkdir(parents=True,exist_ok=True);part.rename(dest)
            subprocess.run([sys.executable,'-Bu',str(SRC/'run_arm.py'),stage,name],check=True,env=environment(),cwd=ROOT,
                           stdout=log,stderr=subprocess.STDOUT)
    print('FINISHED',name,flush=True)


if __name__=='__main__':
    job=sys.argv[1]
    if job=='check':run(SRC/'selfcheck.py')
    elif job=='prep':run(SRC/'train_q.py')
    elif job in ('motion','filter'):
        names=('native',*CAMERA,*BAM) if job=='motion' else QARMS
        with ThreadPoolExecutor(max_workers=len(names)) as pool:list(pool.map(arm,names))
    elif job=='score':run(SRC/'evaluate.py')
    elif job=='probe':run(SRC/'projection_probe.py')
    elif job=='report':run(SRC/'report.py')
    else:raise ValueError(job)
    print('COMPLETE',job,flush=True)
