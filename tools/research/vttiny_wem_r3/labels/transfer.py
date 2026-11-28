"""Deploy, run, and retrieve training-only W R3 label evidence."""
import argparse
import base64
import hashlib
import io
from pathlib import Path
import sys
import zipfile


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from remote_exec import remote


PREFIX = """
integrations = r.parent
work = integrations/'vttiny_wem_r3_v1'
labels = work/'labels_live'
tco = integrations/'botsort_plus_v0_4_motion_blur/tco_53faeccb/t23_v1'
repo = integrations/'botsort_plus_v0_4_motion_blur/BOT-SORT-Plus-followup-53faeccb'
source = work/'label_source'/version
"""

DEPLOY = """
import base64
source.mkdir(parents=True,exist_ok=True)
for name, encoded in files.items():
    data=base64.b64decode(encoded)
    path=source/name
    if path.is_file():
        if path.read_bytes()!=data:
            raise ValueError('Versioned R3 label artifact changed')
    else:
        with path.open('xb') as out:
            out.write(data)
print('staged',source)
"""

LAUNCH = """
import time

parity = work/'observer/parity'/(stage+'.json')
verified = json.loads(parity.read_bytes())
if verified['status']!='PASS' or verified['stage']!=stage or verified['gt_open_allowed'] or verified['online_w_updates']:
    raise ValueError('Training observer parity not verified')
if len(verified['sequences']) != {'smoke':3,'full':74}[stage]:
    raise ValueError('Observer proof has unexpected number of training sequences')
labels.mkdir(parents=True,exist_ok=True)
target = labels/stage
log = labels/(stage+'.log')
pid = labels/(stage+'.pid')
if target.exists() or log.exists() or pid.exists():
    if not retry or (target/'summary.json').is_file():
        raise ValueError('Prior R3 offline output exists')
    if pid.is_file():
        cmdline=Path('/proc')/pid.read_text().strip()/'cmdline'
        if cmdline.is_file() and b'label_train.py' in cmdline.read_bytes():
            raise ValueError('Prior R3 labeler is still running')
    old=labels/'superseded'/f'{stage}_{time.time_ns()}'
    old.mkdir(parents=True)
    for path in (target,log,pid):
        if path.exists():
            path.rename(old/path.name)
cmd=[sys.executable,'-X','utf8','-Bu',str(source/'label_train.py'),
     '--capture',str(parity),'--split',str(tco/'split/training_split.json'),
     '--gt',str(tco/'training_gt'),'--gt-manifest',str(tco/'training_gt_manifest.json'),
     '--detections',str(tco/'training_inputs/detections'),
     '--detection-manifest',str(tco/'training_inputs/verified.json'),
     '--trackers',str(tco/'training_runs/B/trackers'),
     '--trackeval',str(integrations.parent/'external/TrackEval'),
     '--repo',str(repo),'--reuse-r1b',str(source),'--output',str(target)]
if stage=='smoke':
    cmd += ['--smoke']
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',
         OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
with log.open('xb') as out:
    proc=subprocess.Popen(cmd,env=env,cwd=source,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,
                          start_new_session=True)
pid.write_text(str(proc.pid))
print(stage,proc.pid,target)
"""

STATUS = """
log = labels/(stage+'.log')
pid = labels/(stage+'.pid')
if log.is_file():
    print(log.read_bytes()[-2800:].decode(errors='replace'))
if pid.is_file():
    cmdline=Path('/proc')/pid.read_text().strip()/'cmdline'
    print('running',cmdline.is_file() and b'label_train.py' in cmdline.read_bytes())
summary=labels/stage/'summary.json'
if summary.is_file():
    data=json.loads(summary.read_bytes())
    print(data['status'],json.dumps({k:v['edge_counts'] for k,v in data['partitions'].items()},ensure_ascii=False))
else:
    print('R3 label summary not ready')
"""

FETCH = """
import io, zipfile
target=labels/stage
assert json.loads((target/'summary.json').read_bytes())['status']=='TRAINING_DIAGNOSTIC_LABELS_ONLY'
with io.BytesIO() as buf:
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        for path in target.rglob('*'):
            if path.is_file() and (path.suffix=='.json' or detail):
                z.write(path,path.relative_to(target).as_posix())
    sys.stdout.buffer.write(buf.getvalue())
"""


def assets():
    r1b = HERE / "r1b"
    files = {p.name: base64.b64encode(p.read_bytes()).decode("ascii") for p in
             (HERE / "label_train.py", r1b / "label_r1b.py", r1b / "label_rules.py")}
    return hashlib.sha256(repr(files).encode()).hexdigest()[:16], files


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("deploy", "launch", "status", "fetch"))
    p.add_argument("stage", choices=("smoke", "full"), nargs="?", default="full")
    p.add_argument("--retry", action="store_true")
    p.add_argument("--detail", action="store_true")
    a = p.parse_args()
    version, files = assets()
    script = dict(deploy=DEPLOY, launch=DEPLOY + LAUNCH, status=STATUS, fetch=FETCH)[a.action]
    values = f"version={version!r}\nstage={a.stage!r}\nretry={a.retry!r}\ndetail={a.detail!r}\n"
    if a.action in ("deploy", "launch"):
        values += f"files={files!r}\n"
    result = remote(values + PREFIX + script)
    if a.action != "fetch":
        print(result.decode())
        return
    target = HERE / "results" / a.stage
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(result)) as z:
        if z.testzip() is not None:
            raise ValueError("Bad R3 labels archive")
        for name in z.namelist():
            out = (target / name).resolve()
            if not out.is_relative_to(target.resolve()):
                raise ValueError("Unsafe R3 labels archive entry")
            out.parent.mkdir(parents=True, exist_ok=True)
            data = z.read(name)
            if out.exists() and out.read_bytes() != data:
                raise ValueError(f"Existing R3 evidence differs at {out}")
            out.write_bytes(data)
    print(target)


if __name__ == "__main__":
    main()
