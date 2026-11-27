"""Run four independent full replays and then score them with the archived TrackEval evaluator."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import subprocess
import sys
import time
from joint_settings import COMBINED, FIX, METHODS, MOTION, ROOT, SOURCE, environment, protocol, run_dir


def replay(method,gpu):
    directory=run_dir(method)
    directory.parent.mkdir(parents=True,exist_ok=True)
    with (directory.parent/'replay.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        manifest=directory/'protocols'/(method+'.json')
        if manifest.exists() and protocol(directory,method)['status']=='COMPLETE':
            print('Complete output present',method,flush=True)
            return
        if directory.exists():
            directory.rename(directory.with_name('incomplete-'+str(time.time_ns())))
        cmd=([SOURCE/'run_fixed.py',method] if method in (FIX,COMBINED)
             else [MOTION/'run_arm.py','full',method])
        with (ROOT/(method+'.log')).open('w') as log:
            print('Starting',method,flush=True)
            subprocess.run([sys.executable,'-Bu',*map(str,cmd)],check=True,env=environment(method,gpu),
                           stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
        p=protocol(directory,method)
        assert p['status']=='COMPLETE' and len(p['sequences'])==30 and sum(p['sequences'].values())==12590
        print('Finished',method,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('job',choices=('replay','score'))
    parser.add_argument('--gpu',type=int,default=5)
    args=parser.parse_args()
    ROOT.mkdir(exist_ok=True,parents=True)
    if args.job=='replay':
        with ThreadPoolExecutor(max_workers=4) as pool:
            tasks=[pool.submit(replay,method,args.gpu) for method in METHODS]
            for future in tasks:
                future.result()
    else:
        subprocess.run([sys.executable,'-Bu',str(SOURCE/'score.py')],env=environment(METHODS[0],args.gpu),
                       check=True,cwd=ROOT)
    print('COMPLETE',args.job,flush=True)


if __name__=='__main__':
    main()
