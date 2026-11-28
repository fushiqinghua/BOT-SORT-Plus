"""Run the isolated training-side W R3 feature and diagnostic comparison."""

import argparse
import base64
import hashlib
import io
from pathlib import Path
import sys
import zipfile


HERE = Path(__file__).resolve().parent
from remote_exec import remote


PREFIX = """
integrations = r.parent
r = integrations.parent
work = integrations/'vttiny_wem_r3_v1'
obs = work/'observer'
suite = integrations/'botsort_plus_v0_4_motion_blur'
repo = suite/'BOT-SORT-Plus-followup-53faeccb'
train = suite/'tco_53faeccb/t23_v1'
split = train/'split/training_split.json'
assets = train/'training_inputs'
roots = work/'feature_analysis'
code = roots/'sources'/version
features = roots/'embeddings'/scope
labels = work/'labels_live'/scope
output = roots/'scores'/scope
"""

DEPLOY = """
import base64
code.mkdir(parents=True,exist_ok=True)
for name, text in sources.items():
    target = code/name
    data = base64.b64decode(text)
    if target.exists():
        assert target.read_bytes()==data
    else:
        with target.open('xb') as f:
            f.write(data)
"""

LAUNCH = """
import time
settings=json.loads((obs/'prediction_settings.json').read_bytes())
env=dict(os.environ,PYTHONUSERBASE=settings['python_user_base'],OMP_NUM_THREADS='1',
         OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
         PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='' if action=='score' or device=='cpu' else device,
         YOLO_CONFIG_DIR=str(roots/'runtime_config'))
if action=='features':
    parity=obs/'parity'/(scope+'.json')
    proof=json.loads(parity.read_bytes())
    assert proof['status']=='PASS'
    paths={Path(s['snapshot_file']).parent for s in proof['references']['B']['files'].values()}
    snapshots,=paths
    weights=r/'outputs/yolo11m_rgb_rgbt_mot_test_v1/rgbt/weights/clip_veri.pt'
    args=[str(code/'score_features.py'),'--repo',str(repo),'--split',str(split),'--assets',str(assets),
          '--parity',str(parity),'--snapshots',str(snapshots),'--rgb-weights',str(weights),
          '--reference-protocol',str(train/'training_runs/B/protocols/B_early_moderate.json'),
          '--runtime-root',settings['runtime_root'],'--output',str(features),'--scope',scope,
          '--device','cpu' if device=='cpu' else 'cuda:0']
    complete=features/'manifest.json'
else:
    assert (features/'manifest.json').is_file() and (labels/'summary.json').is_file()
    args=[str(code/'admission.py'),'--features',str(features),'--labels',str(labels),'--split',str(split),
          '--protocol',str(code/'R3_PROTOCOL.md'),'--output',str(output),'--scope',scope]
    complete=output/('smoke.json' if scope=='smoke' else 'calibration_lock.json')
logs=roots/'logs'
logs.mkdir(parents=True,exist_ok=True)
stem=action+'_'+scope
log=logs/(stem+'.log')
pid_file=logs/(stem+'.pid')
if complete.is_file():
    raise ValueError(f'Already complete: {complete}')
if log.exists() or pid_file.exists():
    if not retry:
        raise ValueError(f'Prior invocation exists: {stem}')
    if pid_file.is_file():
        cmd=Path('/proc')/pid_file.read_text().strip()/'cmdline'
        if cmd.is_file() and os.fsencode(str(code.parent)) in cmd.read_bytes():
            raise ValueError('Prior job is still active')
    old=roots/'superseded'/f'{stem}_{time.time_ns()}'
    old.mkdir(parents=True)
    for item in (log,pid_file,output if action=='score' else features):
        if item.exists(): item.rename(old/item.name)
with log.open('xb') as stream:
    proc=subprocess.Popen([sys.executable,'-X','utf8','-Bu',*args],cwd=repo,env=env,
                          stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
pid_file.write_text(str(proc.pid))
print(stem,proc.pid,version)
"""


STATUS = """
from pathlib import Path
for item,p in [('features',features/'manifest.json'),('score',output/('smoke.json' if scope=='smoke' else 'calibration_lock.json'))]:
    stem=item+'_'+scope
    log=roots/'logs'/(stem+'.log')
    pid=roots/'logs'/(stem+'.pid')
    active=pid.is_file() and (Path('/proc')/pid.read_text().strip()/'cmdline').is_file()
    print(item,'complete' if p.is_file() else ('running' if active else 'pending_or_failed'))
    if log.is_file(): print(log.read_bytes()[-1600:].decode('utf-8','replace'))
"""


FETCH = """
import io, zipfile
with io.BytesIO() as buf:
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as archive:
        archive.write(features/'manifest.json','feature_manifest.json')
        for file in output.iterdir():
            archive.write(file,file.name)
    sys.stdout.buffer.write(buf.getvalue())
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("features","score","status","fetch"))
    p.add_argument("scope", choices=("smoke","full"), nargs="?", default="full")
    p.add_argument("--device", default="4", choices=("cpu","0","1","2","3","4","5","6","7"))
    p.add_argument("--retry", action="store_true")
    a = p.parse_args()
    sources = {name:base64.b64encode((HERE/name).read_bytes()).decode()
               for name in ("score_features.py","admission.py","R3_PROTOCOL.md")}
    version = hashlib.sha256(repr(sources).encode()).hexdigest()[:16]
    prefix = f"scope={a.scope!r}\nversion={version!r}\n" + PREFIX
    if a.action in ("features","score"):
        script = (prefix+f"sources={sources!r}\naction={a.action!r}\ndevice={a.device!r}\nretry={a.retry!r}\n"
                  +DEPLOY+LAUNCH)
    else:
        script = prefix+(STATUS if a.action=="status" else FETCH)
    packet = remote(script)
    if a.action!="fetch":
        print(packet.decode())
        return
    output = HERE/"results"/a.scope
    output.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(packet)) as archive:
        if archive.testzip() or any(Path(n).name!=n for n in archive.namelist()):
            raise ValueError("Invalid R3 evidence archive")
        for name in archive.namelist():
            target=output/name
            content=archive.read(name)
            if target.is_file() and target.read_bytes()!=content:
                raise ValueError(f"Local result changed: {target}")
            target.write_bytes(content)
    print(output)


if __name__=="__main__":
    main()
