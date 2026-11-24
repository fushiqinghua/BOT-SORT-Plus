"""Detection/feature contracts with checksums and exact row alignment."""
from __future__ import annotations
import gzip
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from .types import Feature


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    temp.replace(path)


def read_detection_cache(path):
    path=Path(path)
    opener=gzip.open if path.suffix==".gz" else open
    with opener(path,"rt",encoding="utf-8") as stream:
        data=json.load(stream)
    if not isinstance(data.get("model_sha256"),str) or not data.get("frames"):
        raise ValueError(f"Invalid detector cache: {path}")
    for expected,row in enumerate(data["frames"],1):
        if row.get("frame")!=expected or isinstance(row.get("frame"),bool):
            raise ValueError("Cached frames must be contiguous 1-based, including empty frames")
        det=np.asarray(row["detections"],np.float32)
        if det.size==0:
            det=np.empty((0,6),np.float32)
        if det.ndim!=2 or det.shape[1]!=6 or not np.isfinite(det).all():
            raise ValueError("Cache detections require finite Nx6 [xyxy,score,class]")
        if np.any(det[:,2:4]<=det[:,:2]) or np.any((det[:,4]<0)|(det[:,4]>1)):
            raise ValueError("Invalid detection size or score")
        if not isinstance(row.get("image"),str):
            raise ValueError("Every frame requires an RGB image path for native SOF GMC")
    return data


def detection_fingerprint(cache):
    """Path-independent fingerprint; .json/.json.gz and relocation are equivalent.

    Includes model identity, frame IDs, row order, ALL original categories and
    float32 detection values. No cached feature can be reused after reordering.
    """
    h=hashlib.sha256(b"botsort_plus_detection_v1\0")
    h.update(cache["model_sha256"].encode("utf-8"))
    for row in cache["frames"]:
        det=np.asarray(row["detections"],dtype="<f4").reshape(-1,6)
        h.update(np.asarray([row["frame"],len(det)],dtype="<i8").tobytes())
        h.update(det.tobytes(order="C"))
    return h.hexdigest()


def sequence_name(path):
    return Path(path).name.removesuffix(".gz").removesuffix(".json")


def detection_files(directory):
    paths=sorted(p for p in Path(directory).iterdir() if p.name.endswith((".json",".json.gz")))
    names=[sequence_name(p) for p in paths]
    if len(set(names))!=len(names):
        raise ValueError("Both compressed and uncompressed caches exist for a sequence; choose one")
    if not paths:
        raise FileNotFoundError(f"No detection caches in {directory}")
    return paths


def resolve_image(value, image_root):
    path=Path(value)
    if not path.is_absolute():
        path=Path(image_root)/path
    if not path.is_file():
        raise FileNotFoundError(f"Original image missing: {path}. Set --images-root to the parent "
                                "of test2017; dummy frames would invalidate SOF GMC.")
    return path


def forbid_ground_truth(event,args):
    if event=="open" and args and isinstance(args[0],(str,bytes,os.PathLike)):
        value=os.fsdecode(args[0]).lower().replace("\\","/")
        if value.endswith("/gt.txt") or "/annotations/" in value or value.endswith("instances_test2017.json"):
            raise PermissionError("Ground truth must not be opened during feature extraction/prediction")


class FeatureCache:
    def __init__(self,path,cache,expected_rgb_sha=None,require_thermal=False):
        path=Path(path)
        self.metadata=json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        meta=self.metadata
        if meta.get("schema")!="botsort_plus_features_v1":
            raise ValueError("Unsupported feature cache; export with export_botsort_plus_features.py")
        if meta.get("features_sha256")!=sha256(path):
            raise ValueError("Feature cache checksum mismatch")
        if meta.get("detection_fingerprint")!=detection_fingerprint(cache):
            raise ValueError("Features do not match these ordered detector outputs")
        if expected_rgb_sha is not None and meta.get("encoders",{}).get("rgb",{}).get("sha256")!=expected_rgb_sha:
            raise ValueError("RGB encoder checkpoint differs from the frozen baseline")
        with np.load(path,allow_pickle=False) as z:
            self.arrays={k:z[k].copy() for k in z.files}
        a=self.arrays
        expected_frames=np.asarray([r["frame"] for r in cache["frames"]])
        expected_offsets=np.r_[0,np.cumsum([len(r["detections"]) for r in cache["frames"]])]
        if not np.array_equal(a.get("frame_ids"),expected_frames) or not np.array_equal(a.get("offsets"),expected_offsets):
            raise ValueError("Feature frame/row offsets differ from detection cache")
        total=int(expected_offsets[-1]); self.modalities=[]
        for modality in ("rgb","thermal"):
            if modality not in a:
                if modality=="rgb":
                    raise ValueError("Real RGB features are required by the baseline")
                continue
            values=a[modality]; valid=a.get(modality+"_valid"); quality=a.get(modality+"_quality")
            if values.ndim!=2 or len(values)!=total or not np.isfinite(values).all():
                raise ValueError(f"Invalid {modality} feature matrix")
            if valid is None or valid.shape!=(total,) or valid.dtype!=np.bool_:
                raise ValueError(f"{modality} validity must be a boolean vector")
            if quality is None or quality.shape!=(total,) or not np.isfinite(quality).all() or np.any((quality<0)|(quality>1)):
                raise ValueError(f"Invalid {modality} observation quality")
            norms=np.linalg.norm(values[valid].astype(np.float64),axis=1)
            if np.any(norms<=1e-12):
                raise ValueError("Zero placeholder incorrectly marked as a real feature")
            if not np.allclose(norms,1.,rtol=1e-4,atol=1e-4):
                raise ValueError(f"Valid {modality} features must be unit normalized")
            declared_dim=meta.get("encoders",{}).get(modality,{}).get("output_dim")
            if declared_dim is not None and declared_dim!=values.shape[1]:
                raise ValueError(f"{modality} dimension differs from encoder metadata")
            self.modalities.append(modality)
        if require_thermal:
            mapping=meta.get("spatial_mapping")
            if "thermal" not in self.modalities or not a["thermal_valid"].any():
                raise ValueError("C requires real thermal features; no silent RGB-only substitution")
            if not meta.get("encoders",{}).get("thermal",{}).get("sha256"):
                raise ValueError("C requires thermal encoder provenance")
            if not isinstance(mapping,dict) or mapping.get("mode") not in ("aligned","homography") or mapping.get("declared_by_user") is not True:
                raise ValueError("C requires explicitly declared RGB-to-thermal spatial mapping")

    def frame(self,index):
        a=self.arrays; start,end=map(int,a["offsets"][index:index+2])
        packets=[]
        for row in range(start,end):
            packets.append({m:Feature(a[m][row].copy(),float(a[m+"_quality"][row]))
                            for m in self.modalities if a[m+"_valid"][row]})
        return packets
