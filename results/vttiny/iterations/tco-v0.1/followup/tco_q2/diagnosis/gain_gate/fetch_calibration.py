"""Read the two frozen calibration score files and the corresponding official training GT."""
import io
from pathlib import Path
import sys
import zipfile

HERE = Path(__file__).resolve().parent
from on_host import run_on_host


def fetch():
    archive = run_on_host("""
import io, zipfile
c=root/'corpus/calibration'
s=json.loads((root/'corpus/crop_manifest.json').read_bytes())
spl=json.loads((root/'split/training_split.json').read_bytes())
with __import__('gzip').open(c/'row_keys.jsonl.gz','rt') as stream:
    sequences=sorted({json.loads(line)['sequence'] for line in stream})
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
    for name in ('absolute_only','paired_gain'):
        z.write(root/'quality_run'/(name+'_calibration.csv.gz'),name+'.csv.gz')
    for name in ('row_keys.jsonl.gz','gt_xyxy.npy'):
        z.write(c/name,name)
    for sequence in sequences:
        z.write(old/'training_gt'/sequence/'gt/gt.txt','gt/'+sequence+'.txt')
sys.stdout.buffer.write(buf.getvalue())
""")
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        if z.testzip():
            raise ValueError('Calibration transfer failed integrity check')
        for name in z.namelist():
            dest = (HERE/'inputs'/name).resolve()
            if not dest.is_relative_to(HERE/'inputs'):
                raise ValueError(name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(z.read(name))


if __name__ == '__main__':
    fetch()
