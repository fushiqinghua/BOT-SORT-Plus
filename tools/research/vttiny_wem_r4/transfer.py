"""Upload a versioned R4 runner and manage independent frozen-cache replays."""
import argparse
import base64
import hashlib
import io
from pathlib import Path
import sys
import zipfile


HERE = Path(__file__).resolve().parent
R3 = HERE.parent / "vttiny_wem_r3"
sys.path.insert(0, str(R3))
from remote_exec import remote


REMOTE = """
import base64, io, time, zipfile
root=r.parent/'vttiny_wem_r4_v1'
repo=r.parent/'botsort_plus_v0_4_motion_blur/BOT-SORT-Plus-followup-53faeccb'
sett=r.parent/'vttiny_wem_r01_v1/observer/prediction_settings.json'
job_root=root/stage
log=root/(stage+'.log')
pid=root/(stage+'.pid')
if action=='run':
    code=root/'source'/version
    code.mkdir(parents=True,exist_ok=True)
    for name, encoded in sources.items():
        f=code/name
        data=base64.b64decode(encoded)
        if f.exists():
            assert f.read_bytes()==data
        else:
            with f.open('xb') as out: out.write(data)
    if (job_root/'complete.json').exists():
        raise ValueError('Stage complete; use a clean isolated run to replicate it')
    if job_root.exists() or log.exists() or pid.exists():
        running=Path('/proc')/pid.read_text().strip()/'cmdline' if pid.is_file() else None
        if running is not None and running.is_file() and os.fsencode(str(root/'source')) in running.read_bytes():
            raise ValueError('Requested stage is already running')
        if not retry:
            raise ValueError('Existing partial stage; inspect logs and use --retry')
        backup=root/'superseded'/f'{stage}_{time.time_ns()}'
        backup.mkdir(parents=True)
        for entry in ((log,pid) if repair else (job_root,log,pid)):
            if entry.exists(): entry.rename(backup/entry.name)
    settings=json.loads(sett.read_bytes())
    env=dict(os.environ,PYTHONUSERBASE=settings['python_user_base'],PYTHONDONTWRITEBYTECODE='1',
             PYTHONWARNINGS='ignore',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    cmd=[sys.executable,'-Bu',str(code/'run.py'),stage,'--gpu',gpu]
    if repair: cmd+=['--repair-lower']
    with log.open('xb') as out:
        job=subprocess.Popen(cmd,cwd=repo,env=env,
                             stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
    pid.write_text(str(job.pid))
    print(stage,version,job.pid)
elif action=='status':
    running=Path('/proc')/pid.read_text().strip()/'cmdline' if pid.is_file() else None
    print(stage,'PASS' if (job_root/'complete.json').is_file() else 'RUNNING' if running and running.is_file()
          and os.fsencode(str(root/'source')) in running.read_bytes() else 'INCOMPLETE')
    if log.is_file():print(log.read_text(errors='replace')[-2400:])
    for folder in sorted(job_root.glob('*_*')):
        if folder.is_dir() and folder.name.startswith(('B_', 'BG_')):
            print(folder.name,len(list((folder/'capture_receipts').glob('*/*.json'))), 'sequences complete')
    for f in sorted(job_root.glob('*.log')):
        end=f.read_text(errors='replace')[-3600:]
        if 'Traceback' in end or f.name=='evaluate.log':print(f.name,end[-1650:])
elif action=='fetch':
    assert (job_root/'complete.json').is_file()
    with io.BytesIO() as buf:
        with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
            for f in (job_root/'verified.json', job_root/'plan.json', job_root/'complete.json'):z.write(f,f.name)
            for folder in ('metrics', 'attribution'):
                for f in sorted((job_root/folder).rglob('*')):
                    if f.is_file():z.write(f,f.relative_to(job_root).as_posix())
        sys.stdout.buffer.write(buf.getvalue())
elif action=='inspect':
    folders=list((r.parent/'botsort_plus_v0_4_motion_blur/tco_53faeccb/t01_v1').glob('*'))
    print([p.name for p in folders]); print((r.parent/'vttiny_wem_r01_v1/observer/parity/full.json').is_file())
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("run", "status", "fetch", "inspect"))
    p.add_argument("stage", choices=("smoke", "pilot", "full"), nargs="?", default="smoke")
    p.add_argument("--gpu", default="4", choices=tuple(map(str, range(8))))
    p.add_argument("--retry", action="store_true")
    p.add_argument("--repair-lower", action="store_true")
    a = p.parse_args()
    if a.repair_lower and not (a.action == "run" and a.stage == "pilot" and a.retry):
        p.error("--repair-lower requires run pilot --retry")
    files = [HERE / n for n in ("run.py", "online.py", "offline.py")]
    files += [R3 / "observer/capture.py"]
    files += [R3 / n for n in ("score_features.py", "admission.py")]
    sources = {path.name: base64.b64encode(path.read_bytes()).decode() for path in files}
    version = hashlib.sha256(repr(sources).encode()).hexdigest()[:16]
    packet = remote(f"action={a.action!r}\nstage={a.stage!r}\nversion={version!r}\ngpu={a.gpu!r}\n"
                    f"retry={a.retry!r}\nrepair={a.repair_lower!r}\nsources={sources if a.action == 'run' else {}!r}\n" + REMOTE)
    if a.action != "fetch":
        print(packet.decode())
        return
    dest = HERE / "results" / a.stage
    with zipfile.ZipFile(io.BytesIO(packet)) as z:
        if z.testzip():
            raise ValueError("Corrupt R4 report")
        for name in z.namelist():
            target = (dest / name).resolve()
            if not target.is_relative_to(dest.resolve()):
                raise ValueError("Unexpected zipped report path")
            target.parent.mkdir(parents=True, exist_ok=True)
            data = z.read(name)
            if target.exists() and target.read_bytes() != data:
                raise ValueError(f"Result differs at {target}")
            target.write_bytes(data)
    print(dest)


if __name__ == "__main__":
    main()
