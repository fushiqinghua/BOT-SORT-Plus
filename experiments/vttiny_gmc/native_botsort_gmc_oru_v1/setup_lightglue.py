"""Download official SuperPoint/LightGlue and keep dependencies in this experiment."""
from pathlib import Path
import hashlib
import io
import json
import os
import subprocess
import sys
import time
import urllib.request
import zipfile

root=Path(__file__).resolve().parent
vendor=root/"vendor"
vendor.mkdir(exist_ok=True)
headers={"User-Agent":"VT-Tiny-MOT-research"}


def get(url,accept=None):
    for attempt in range(5):
        try:
            request_headers=dict(headers,**({"Accept":accept} if accept else {}))
            with urllib.request.urlopen(urllib.request.Request(url,headers=request_headers),timeout=55) as response:
                return response.read()
        except OSError as exc:
            print("fetch attempt",attempt+1,type(exc).__name__,flush=True)
            if attempt==4:
                raise
            time.sleep(2+attempt*2)


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


pin=vendor/"lightglue_commit.txt"
if not pin.exists():
    pin.write_text("5a650c444718caffc2669dc0094cde908f42dd73")
commit=pin.read_text().strip()
assert len(commit)==40 and all(c in "0123456789abcdef" for c in commit)
source=vendor/("LightGlue-"+commit)
if not (source/"lightglue/lightglue.py").is_file():
    archive=zipfile.ZipFile(io.BytesIO(get("https://codeload.github.com/cvg/LightGlue/zip/"+commit)))
    for info in archive.infolist():
        parts=Path(info.filename).parts
        if len(parts)<2 or parts[0]!=source.name or '..' in parts or info.is_dir():
            continue
        if parts[1]=="lightglue" and info.filename.endswith(".py") or parts[1].startswith("LICENSE"):
            dest=vendor/info.filename
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(archive.read(info))
print("source",commit,flush=True)
deps=vendor/"dependencies"
if not (deps/"kornia/__init__.py").is_file():
    subprocess.run([sys.executable,"-m","pip","install","--disable-pip-version-check","--no-deps","--target",
                    str(deps),"kornia==0.7.4","kornia-rs==0.1.9"],check=True)
hub=vendor/"torch"
ckpt=hub/"hub/checkpoints"; ckpt.mkdir(parents=True,exist_ok=True)
files={}
official=[
    (114312652,"superpoint_lightglue_v0-1_arxiv.pth","6ff7040d0a497fc6639337946d7538dae07428c18f77a067a0b5a960e7cc551a"),
    (114312661,"superpoint_v1.pth","52b6708629640ca883673b5d5c097c4ddad37d8048b33f09c8ca0d69db12c40e")]
for asset,local,expected in official:
    target=ckpt/local
    if not target.is_file():
        raw=get("https://api.github.com/repos/cvg/LightGlue/releases/assets/"+str(asset)+"?download=1","application/octet-stream")
        assert hashlib.sha256(raw).hexdigest()==expected
        tmp=target.with_suffix(".part"); tmp.write_bytes(raw); tmp.replace(target)
    files[local]=sha(target)
    assert files[local]==expected
    print("model",local,files[local],flush=True)
os.environ["TORCH_HOME"]=str(hub)
sys.path[:0]=[str(source),str(deps)]
from lightglue import LightGlue,SuperPoint
import torch
m=LightGlue(features="superpoint",flash=False).eval()
e=SuperPoint(max_num_keypoints=512).eval()
(vendor/"provenance.json").write_text(json.dumps(dict(commit=commit,weights=files,torch=torch.__version__,
   kornia="0.7.4",kornia_rs="0.1.9",source_sha256={p.name:sha(p) for p in (source/"lightglue").glob("*.py")}),indent=2))
print("SETUP_COMPLETE",flush=True)
