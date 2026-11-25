#!/usr/bin/env python3
"""Replay frozen detections + real cached embeddings + original SOF images.

Prediction never reads GT. Evaluation is a separate process/CLI. Existing run
directories are not overwritten; choose a new output root to change parameters.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import sys
import time
import traceback

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import numpy as np
from botsort_plus import __version__ as plus_version
from botsort_plus.config import PlusConfig
from botsort_plus.io import (FeatureCache, detection_files, detection_fingerprint,
                            forbid_ground_truth, read_detection_cache, resolve_image,
                            sequence_name, sha256, write_json)


def prepare(args):
    cfg=PlusConfig.load(args.config)
    source_path=args.baseline_root/"protocol.json"
    source=json.loads(source_path.read_text(encoding="utf-8"))
    detector_sha=source["weight_sha256"]; rgb_sha=source["reid_sha256"]
    native=dict(source["trackers"]["botsort_reid"])
    native.update(cfg.native_overrides)
    frozen={sequence_name(p):p for p in detection_files(args.baseline_root/"detections")}
    paths=detection_files(args.detections_dir or args.baseline_root/"detections")
    expected=set(source["sequences"])
    if args.sequences:
        chosen=set(args.sequences)
        if not chosen <= expected:
            raise ValueError(f"Unknown sequences: {chosen-expected}")
    else:
        chosen=expected
    selected={sequence_name(p):p for p in paths if sequence_name(p) in chosen}
    if set(selected)!=chosen:
        raise ValueError(f"Missing detection sequences: {chosen-set(selected)}")
    sequences={}; fingerprints={}; missing=[]; coverage=Counter(); encoder_contracts={}; feature_hashes={}
    thermal_required=cfg.appearance.enabled and cfg.appearance.weight_thermal>0
    for seq,path in selected.items():
        cache=read_detection_cache(path)
        if cache["model_sha256"]!=detector_sha or len(cache["frames"])!=len(source["sequences"][seq]):
            raise ValueError(f"Frozen detector/protocol mismatch: {seq}")
        sequences[seq]=len(cache["frames"]); fingerprints[seq]=detection_fingerprint(cache)
        if seq not in frozen:
            raise ValueError(f"No frozen reference detection cache: {seq}")
        if path.resolve()!=frozen[seq].resolve() and fingerprints[seq]!=detection_fingerprint(read_detection_cache(frozen[seq])):
            raise ValueError(f"{seq}: --detections-dir content differs from frozen baseline detections")
        for row in cache["frames"]:
            try:
                resolve_image(row["image"],args.images_root)
            except FileNotFoundError as error:
                missing.append(str(error))
        feature_path=args.features_dir/(seq+".npz")
        if not feature_path.is_file() or not feature_path.with_suffix(".json").is_file():
            missing.append(f"Missing feature cache/metadata: {feature_path}")
            continue
        fc=FeatureCache(feature_path,cache,expected_rgb_sha=rgb_sha,require_thermal=thermal_required)
        feature_hashes[seq]={"features":sha256(feature_path),"metadata":sha256(feature_path.with_suffix(".json"))}
        for modality in fc.modalities:
            contract=fc.metadata.get("encoders",{}).get(modality)
            if not isinstance(contract,dict):
                raise ValueError(f"Missing {modality} encoder metadata: {seq}")
            if modality in encoder_contracts and contract!=encoder_contracts[modality]:
                raise ValueError(f"Mixed {modality} encoder or preprocessing across sequences")
            encoder_contracts[modality]=contract
        for i,row in enumerate(cache["frames"]):
            det=np.asarray(row["detections"],np.float32).reshape(-1,6)
            start,end=map(int,fc.arrays["offsets"][i:i+2])
            high=det[:,4].astype(np.float64)>native["track_high_thresh"]
            if np.any(~fc.arrays["rgb_valid"][start:end][high]):
                raise ValueError(f"{seq} frame {row['frame']}: missing high-score baseline RGB embeddings")
        for mode in fc.modalities:
            coverage[mode+"_valid"]+=int(fc.arrays[mode+"_valid"].sum())
            coverage[mode+"_quality_for_memory"]+=int((fc.arrays[mode+"_valid"] &
                (fc.arrays[mode+"_quality"]>=cfg.appearance.min_update_quality)).sum())
    return cfg,selected,{
        "schema":"botsort_plus_run_v1", "status":"READY" if not missing else "MISSING_INPUTS",
        "implementation_version":plus_version,
        "implementation_sha256":{str(p.relative_to(REPO)):sha256(p)
                                 for p in sorted((REPO/"botsort_plus").glob("*.py"))},
        "name":cfg.name,"config":cfg.to_dict(),"native_config":native,
        "sequences":sequences,"subset":chosen!=expected,
        "detector_sha256":detector_sha,"rgb_encoder_sha256":rgb_sha,
        "source_protocol_sha256":sha256(source_path),"detection_fingerprints":fingerprints,
        "encoder_contracts":encoder_contracts,"feature_hashes":feature_hashes,
        "input_roots":{"baseline":str(args.baseline_root.resolve()),
                       "images":str(args.images_root.resolve()),"features":str(args.features_dir.resolve())},
        "feature_coverage":dict(coverage),"missing_count":len(missing),"missing_examples":missing[:10],
        "evaluation_context":"Historical official test was previously used for checkpoint/gate selection; "
                             "treat new runs here as development experiments, not an unseen test.",
        "causal":True,"backfilled_frames":0,"interpolation":False,
    }


def run(args):
    sys.addaudithook(forbid_ground_truth)
    cfg,paths,protocol=prepare(args)
    if args.preflight:
        print(json.dumps(protocol,ensure_ascii=False,indent=2))
        return 0 if protocol["status"]=="READY" else 2
    if protocol["status"]!="READY":
        print(json.dumps(protocol,ensure_ascii=False,indent=2))
        raise FileNotFoundError("Inputs missing. Complete feature export/image paths before replay; no dummy GMC fallback.")
    runtime=args.runtime_root or os.environ.get("TRACKING_RUNTIME_ROOT")
    if runtime:
        sys.path.insert(0,str(Path(runtime)/"tracking_runtime"))
    import cv2
    import torch
    from botsort_plus.native_adapter import BoTSORTPlus
    cv2.setNumThreads(1); torch.set_num_threads(1)
    root=args.output_dir; name=cfg.name
    protocol_path=root/"protocols"/(name+".json")
    tracker_dir=root/"trackers"/name
    if protocol_path.exists() or tracker_dir.exists():
        raise FileExistsError(f"Run {name} already exists in {root}. Use a new output root; no silent resume/overwrite.")
    protocol.update(status="RUNNING",started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
                    runtime={"python":sys.version,"numpy":np.__version__,"torch":torch.__version__,
                             "opencv":cv2.__version__},output_prediction_hashes={})
    write_json(protocol_path,protocol)
    tracker_dir.joinpath("data").mkdir(parents=True)
    try:
        for seq,path in paths.items():
            cache=read_detection_cache(path)
            if detection_fingerprint(cache)!=protocol["detection_fingerprints"][seq]:
                raise ValueError(f"Detection inputs changed after preflight: {seq}")
            feature_path=args.features_dir/(seq+".npz")
            if {"features":sha256(feature_path),"metadata":sha256(feature_path.with_suffix(".json"))}!=protocol["feature_hashes"][seq]:
                raise ValueError(f"Feature inputs changed after preflight: {seq}")
            fc=FeatureCache(feature_path,cache,expected_rgb_sha=protocol["rgb_encoder_sha256"],
                            require_thermal=cfg.appearance.enabled and cfg.appearance.weight_thermal>0)
            cv2.setRNGSeed(0)
            tracker=BoTSORTPlus(plus_config=cfg,reid_weights=Path("cache-only.pt"),device=torch.device("cpu"),
                               half=False,**protocol["native_config"])
            protocol["boxmot_source_hashes"]=tracker.upstream_hashes
            dest=tracker_dir/"data"/(seq+".txt"); temp=dest.with_suffix(".tmp")
            event_stream=None; event_counts=Counter(); rows=0; seconds=0.; wall=time.perf_counter()
            try:
                if args.trace:
                    event_path=root/"events"/name/(seq+".jsonl")
                    event_path.parent.mkdir(parents=True,exist_ok=True)
                    event_stream=event_path.open("w",encoding="utf-8")
                with temp.open("w",newline="") as stream:
                    writer=csv.writer(stream,lineterminator="\n")
                    for index,row in enumerate(cache["frames"]):
                        image_path=resolve_image(row["image"],args.images_root)
                        image=cv2.imread(str(image_path))
                        if image is None or image.shape[:2]!=(512,640):
                            raise ValueError(f"Expected original 512x640 RGB image: {image_path}")
                        det=np.asarray(row["detections"],np.float32).reshape(-1,6); det[:,5]=0
                        packets=fc.frame(index)
                        t=time.perf_counter(); prediction=tracker.update(det,image,features=packets); seconds+=time.perf_counter()-t
                        seen=set()
                        for item in prediction:
                            x1,y1,x2,y2,tid,score,cls,det_idx=map(float,item)
                            if not np.isfinite(item).all() or tid!=int(tid) or tid<=0 or tid in seen or cls!=0 or x2<=x1 or y2<=y1:
                                raise RuntimeError("Invalid/duplicate MOT output")
                            seen.add(tid)
                            writer.writerow([row["frame"],int(tid),x1,y1,x2-x1,y2-y1,score,-1,-1,-1]); rows+=1
                        for event in tracker.last_events:
                            event_counts[event["event"]]+=1
                            if event_stream:
                                event_stream.write(json.dumps({"sequence":seq,"frame":row["frame"],**event},
                                                              ensure_ascii=False,allow_nan=False)+"\n")
                for event in tracker.finish_sequence_events():
                    event_counts[event["event"]]+=1
                    if event_stream:
                        event_stream.write(json.dumps({"sequence":seq, **event},
                                                      ensure_ascii=False,allow_nan=False)+"\n")
                temp.replace(dest)
            finally:
                if event_stream:
                    event_stream.close()
            timing={"sequence":seq,"frames":len(cache["frames"]),"output_rows":rows,
                    "association_and_gmc_seconds":seconds,"replay_wall_seconds":time.perf_counter()-wall,
                    "feature_extraction_seconds":fc.metadata.get("extraction_seconds"),
                    "feature_metadata_sha256":sha256(feature_path.with_suffix(".json")),
                    "prediction_sha256":sha256(dest),"event_counts":dict(event_counts),
                    "timing_note":"Replay includes cached features; this is not detector+ReID end-to-end FPS."}
            write_json(root/"timing"/name/(seq+".json"),timing)
            protocol["output_prediction_hashes"][seq]=timing["prediction_sha256"]
            write_json(protocol_path,protocol)
            print(f"{name}: {seq}, {timing['frames']} frames, {rows} rows",flush=True)
        protocol["status"]="COMPLETE"
    except Exception:
        protocol.update(status="FAILED",error=traceback.format_exc())
        raise
    finally:
        write_json(protocol_path,protocol)
    return 0


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-root",type=Path,default=REPO/"results/vttiny/yolo11m_rgb_rgbt/rgbt")
    p.add_argument("--detections-dir",type=Path)
    p.add_argument("--features-dir",type=Path,required=True)
    p.add_argument("--images-root",type=Path,required=True)
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,default=REPO/"outputs/botsort_plus_v1")
    p.add_argument("--runtime-root",type=Path)
    p.add_argument("--sequences",nargs="+")
    p.add_argument("--preflight",action="store_true")
    p.add_argument("--trace",action="store_true",help="Write association/candidate/recovery/memory JSONL logs")
    return p


if __name__=="__main__":
    raise SystemExit(run(parser().parse_args()))
