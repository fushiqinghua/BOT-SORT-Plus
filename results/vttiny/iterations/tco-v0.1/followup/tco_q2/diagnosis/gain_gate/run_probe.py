"""Run the read-only forensic probe in redetr and return its numerical evidence."""
import io
import zipfile

from fetch_calibration import HERE, run_on_host


if __name__ == '__main__':
    setup = """
usage=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used',
                              '--format=csv,noheader,nounits'],text=True)
free=[i for i,u,m in (map(int,v.split(',')) for v in usage.splitlines()) if u<5 and m<1024]
os.environ.update(CUDA_VISIBLE_DEVICES=str(free[0]) if free else '',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                  OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
"""
    code=(HERE/'probe_frozen.py').read_text(encoding='utf-8')
    payload=run_on_host(setup+code+'\nsys.stdout.buffer.write(probe(root,old))\n')
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if archive.testzip() or set(archive.namelist())!={'diagnostics.json','feature_and_loss_controls.csv'}:
            raise ValueError('Incomplete audit transfer')
        for name in archive.namelist():
            (HERE/name).write_bytes(archive.read(name))
