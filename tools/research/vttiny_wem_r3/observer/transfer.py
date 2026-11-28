"""Send and monitor the training-side W observer replay."""
import argparse
import base64
import hashlib
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from remote_exec import remote

PREFIX = """
import base64,hashlib
root=r.parent/'vttiny_wem_r3_v1/observer'
runner=root/'training_capture.py'
log=root/(stage+'.log')
pidfile=root/(stage+'.pid')
proof=root/'parity'/(stage+'.json')
"""
DEPLOY = """
root.mkdir(parents=True,exist_ok=True)
for path, encoded in ((runner, payload), (r.parent/'vttiny_wem_r01_v1/observer/capture.py', observer_payload)):
    data=base64.b64decode(encoded)
    if path.is_file():
        if path.read_bytes()!=data:
            raise ValueError(f'Source differs at {path}')
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as f:
            f.write(data)
print(runner)
"""
RUN = """
if pidfile.exists():
    proc=Path('/proc',pidfile.read_text().strip(),'cmdline')
    if proc.is_file() and os.fsencode(runner) in proc.read_bytes():
        raise ValueError('Stage already running')
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',
         OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
with log.open('ab') as stream:
    child=subprocess.Popen([sys.executable,'-Bu',str(runner),stage],cwd=root,env=env,
                           stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                           start_new_session=True)
pidfile.write_text(str(child.pid))
print(stage,child.pid)
"""
STATUS = """
if pidfile.is_file():
    proc=Path('/proc',pidfile.read_text().strip(),'cmdline')
    active=proc.is_file() and os.fsencode(runner) in proc.read_bytes()
    print(stage,'PASS' if proof.is_file() else ('RUNNING' if active else 'EXITED'))
if log.is_file():
    print(log.read_bytes()[-3600:].decode(errors='replace'))
for mode in ('off','on'):
    base=root/stage/'shards'/mode
    print(mode,len(list(base.glob('??/capture_receipts/*/*.json'))), 'clips complete')
    for child in sorted(base.glob('*.log')):
        rows=child.read_bytes()[-1600:].decode(errors='replace').splitlines()
        if any(x.startswith('Traceback') or 'Error:' in x for x in rows):
            print(child.name,'\\n'.join(rows[-17:]))
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("deploy","run","status","fetch"))
    parser.add_argument("stage", choices=("smoke","full"), nargs="?", default="smoke")
    args = parser.parse_args()
    start = f"stage={args.stage!r}\n" + PREFIX
    if args.action == "deploy":
        payload = base64.b64encode((HERE/"training_capture.py").read_bytes()).decode()
        observer_payload = base64.b64encode((HERE/"capture.py").read_bytes()).decode()
        content = remote(start + f"payload={payload!r}\nobserver_payload={observer_payload!r}\n" + DEPLOY)
    elif args.action == "fetch":
        result = remote(start + "sys.stdout.buffer.write(proof.read_bytes())")
        dest = HERE/"results"/(args.stage+".json")
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(result)
        print(dest,hashlib.sha256(result).hexdigest())
        content = b""
    else:
        content = remote(start + (RUN if args.action == "run" else STATUS))
    sys.stdout.buffer.write(content)
