"""Rebuild this four-arm study under a new output directory on the research server."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir', type=Path)
    args=parser.parse_args()
    src=Path(__file__).resolve().parent
    out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Select an empty output directory for reproduction')
    os.environ['VT_QR_ROOT']=str(out)
    os.environ.setdefault('VT_INTEGRATIONS',str(src.parent))
    from qr_settings import environment
    env=environment()
    steps=[('r_quality.py',),('q_motion.py',),('r_cache.py',),
           *(('replay.py',method) for method in ('native','learned_q','learned_r','learned_qr')),
           ('score.py',),('r_uncertainty.py',)]
    for script,*params in steps:
        subprocess.run([sys.executable,'-Bu',str(src/script),*params],cwd=out,env=env,check=True)

if __name__=='__main__':main()
