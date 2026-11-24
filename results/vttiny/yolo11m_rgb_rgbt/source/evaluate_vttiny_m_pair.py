"""Paired RGB and RGB–IR YOLO11-M tracking runs using frozen detector snapshots."""

import argparse
import csv
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

from vttiny_tracking_common import ROOT, dump, forbid_gt, read, sha

OUT = ROOT / "outputs/yolo11m_rgb_rgbt_mot_test_v1"
ORIGINAL = ROOT / "outputs/rgb_test_7trackers_v1"
ADAPTED = ROOT / "outputs/rgb_test_4trackers_reid_testselect_v2"
CALIBRATED = ROOT / "outputs/rgb_reid_calibration_test_v1"
BOT_RUN = ROOT / "outputs/rgb_botsort_reid_test_v1"
FRAMEWORK = ROOT / "formal_detection_v1/framework"
GPUS = {"rgb": 4, "rgbt": 5}
NAMES = {
    "bytetrack": "ByteTrack", "ocsort": "OC-SORT", "botsort_no_reid": "BoT-SORT（无 ReID）",
    "botsort_reid": "BoT-SORT（微调 ReID）", "deepocsort": "Deep OC-SORT",
    "imprassoc": "ImprAsso", "deepsort": "DeepSORT", "strongsort": "StrongSORT（在线版）",
}
METRICS = ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW", "FP", "FN")


def prepare():
    original = read(ORIGINAL / "protocol.json")
    snapshots = read(OUT / "snapshot_sources.json")
    seqs = {}
    for line in (ROOT / "data/processed_v1/test_pairs.jsonl").read_text().splitlines():
        row = json.loads(line)
        assert row["official_split"] == "test"
        seqs.setdefault(row["sequence"], []).append({"frame": row["frame"], "image": row["rgb"], "thermal": row["ir"]})
    rgb_only = {seq: [{"frame": f["frame"], "image": f["image"]} for f in fs] for seq, fs in seqs.items()}
    assert rgb_only == original["sequences"] and len(seqs) == 30 and sum(map(len, seqs.values())) == 12590
    configs = {name: original["trackers"][name] for name in ("bytetrack", "ocsort", "botsort_no_reid")}
    configs.update({name: row["tracker_config"] for name, row in read(CALIBRATED / "selection.json")["selected"].items()})
    bot = read(BOT_RUN / "comparison.json")["selected"]["finetuned"]
    configs["botsort_reid"] = read(BOT_RUN / "trials" / bot["name"] / "protocol.json")["trackers"]["botsort_reid"]
    reid_sha = read(ADAPTED / "protocol.json")["reid_sha256"]
    assert sha(ADAPTED / "weights/clip_veri.pt") == reid_sha
    for mode, weight in snapshots.items():
        target = OUT / mode
        assert sha(target / "weights/rgb_best.pt") == weight["sha256"]
        reid_path = target / "weights/clip_veri.pt"
        if not reid_path.exists():
            reid_path.symlink_to(ADAPTED / "weights/clip_veri.pt")
        dump(target / "protocol.json", {
            "detector_mode": mode, "frames": 12590, "fps": 15, "sequences": seqs, "trackers": configs,
            "detector": original["detector"], "weight_sha256": weight["sha256"], "reid_sha256": reid_sha,
            "source_weight": weight["source"], "reference": "RGB",
            "evaluation": "Official test 30 sequences. Both detectors used official test for checkpoint selection. ReID and gates also previously selected/calibrated on official test. No new gating search in this run.",
        })


def inference(mode):
    target = OUT / mode
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUS[mode])
    os.environ["YOLO_CONFIG_DIR"] = str(target / "settings")
    os.environ.update(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1")
    sys.path.insert(0, str(FRAMEWORK))
    import numpy as np
    import torch
    from ultralytics import YOLO, YOLOMM

    torch.set_num_threads(4)
    torch.cuda.init()
    sys.addaudithook(forbid_gt)
    p = read(target / "protocol.json")
    checkpoint = target / "weights/rgb_best.pt"
    trained = torch.load(checkpoint, map_location="cpu", weights_only=False)
    dump(target / "checkpoint.json", {
        "epoch": trained["epoch"] + 1, "detector_metrics": trained.get("train_metrics"),
        "sha256": p["weight_sha256"], "source": p["source_weight"],
    })
    del trained
    model = (YOLO if mode == "rgb" else YOLOMM)(str(checkpoint))
    assert model.model.yaml["scale"] == "m", model.model.yaml
    for seq, frames in p["sequences"].items():
        dest = target / "detections" / (seq + ".json")
        if dest.exists():
            assert read(dest)["model_sha256"] == p["weight_sha256"]
            continue
        records = []
        started = time.perf_counter()
        for start in range(0, len(frames), 16):
            batch = frames[start:start + 16]
            rgb = [f["image"] for f in batch]
            options = dict(p["detector"], device=str(GPUS[mode]), verbose=False, save=False)
            if mode == "rgb":
                results = model.predict(source=rgb, **options)
            else:
                results = model.predict(sources={"A": rgb, "B": [f["thermal"] for f in batch]}, **options)
            assert len(results) == len(batch)
            for frame, res in zip(batch, results):
                if mode == "rgb":
                    assert res.orig_shape == (512, 640) and Path(res.path).resolve() == Path(frame["image"]).resolve()
                    boxes = res.boxes
                    det = torch.cat([boxes.xyxy, boxes.conf[:, None], boxes.cls[:, None]], 1).cpu().numpy()
                else:
                    assert res.orig_imgs["A"].shape[:2] == (512, 640) and res.orig_imgs["B"].shape[:2] == (512, 640)
                    assert Path(res.paths["A"]).resolve() == Path(frame["image"]).resolve()
                    assert Path(res.paths["B"]).resolve() == Path(frame["thermal"]).resolve()
                    det = np.asarray(res.boxes, dtype=np.float32).reshape(-1, 6)
                assert np.isfinite(det).all() and np.isin(det[:, 5], np.arange(7)).all()
                assert ((det[:, 2:4] - det[:, :2]) > 0).all()
                records.append(dict(frame, detections=det.tolist()))
        torch.cuda.synchronize()
        dump(dest, dict(model_sha256=p["weight_sha256"], inference_wall_seconds=time.perf_counter()-started, frames=records))
        print("DETECT", mode, seq, len(records), flush=True)
    del model
    torch.cuda.empty_cache()
    import run_vttiny_rgb_tracking as runner

    runner.OUT = target
    runner.embeddings()
    metadata = read(target / "reid_encoder.json")
    metadata["training_domain"] = "VeRi initialization then VT-Tiny-MOT RGB identity fine-tuning"
    metadata["crop_source"] = "RGB detection coordinates, RGB image crops for both detector modes"
    dump(target / "reid_encoder.json", metadata)


def track(job):
    mode, name = job
    os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                      YOLO_CONFIG_DIR=str(OUT / mode / "tracking_settings" / name))
    import run_vttiny_rgb_tracking as runner

    runner.OUT = OUT / mode
    options = read(runner.OUT / "protocol.json")["trackers"]
    reid, sorts = runner.adapters()
    for configs in [runner.BASE, reid.CONFIGS, sorts.CONFIGS]:
        for key in configs:
            configs[key] = options[key]
    original = runner.construct

    def construct(method):
        if method != "botsort_reid":
            return original(method)
        import torch
        import boxmot.trackers.botsort.botsort as native
        torch.set_num_threads(1)
        backend = native.ReidAutoBackend
        native.ReidAutoBackend = lambda **kwargs: SimpleNamespace(model=reid.CacheOnlyEncoder())
        try:
            return native.BotSort(reid_weights=reid.WEIGHTS, device=torch.device("cpu"), half=False, **options[method])
        finally:
            native.ReidAutoBackend = backend

    runner.construct = construct
    timings = runner.track_method(name)
    assert len(timings) == 30 and sum(t["frames"] for t in timings) == 12590
    for item in timings:
        file = runner.OUT / "trackers" / name / "data" / (item["sequence"] + ".txt")
        assert sha(file) == item["prediction_sha256"]
    return mode, name


def summarize():
    from evaluate_vttiny import evaluate

    data = {}
    for mode in GPUS:
        target = OUT / mode
        eval_result = evaluate("rgb", "test", target / "trackers", list(NAMES), target / "metrics")
        data[mode] = {r["tracker"]: r for r in eval_result["results"] if r["sequence"] == "COMBINED_SEQ"}
        assert set(data[mode]) == set(NAMES) and len(eval_result["results"]) == 8 * 31
    d = {name: {key: data["rgbt"][name][key] - data["rgb"][name][key] for key in METRICS} for name in NAMES}
    checkpoints = {m: read(OUT / m / "checkpoint.json") for m in GPUS}
    dump(OUT / "comparison.json", dict(checkpoints=checkpoints, metrics=data, delta_rgbt_minus_rgb=d))
    lines = [
        "# VT-Tiny-MOT YOLO11-M 跟踪评估", "",
        f"RGB 的快照来自第 {checkpoints['rgb']['epoch']} 轮。RGB–IR P3/P4/P5 融合（开启 Mosaic）的快照来自第 {checkpoints['rgbt']['epoch']} 轮。"
        "官方 RGB 测试集包含 30 段序列和 12,590 帧。TrackEval 联合汇总所有目标。", "",
        "两组使用相同跟踪参数和 RGB 真值坐标。外观方法使用同一套微调 RGB ReID 权重，对各检测器产生的框分别提取 RGB 外观特征。"
        "沿用上一轮测试集校准后的参数。检测和 ReID 权重均曾用官方测试集选模。", "",
        "|方法|RGB HOTA|RGB–IR HOTA|ΔHOTA|RGB IDF1|RGB–IR IDF1|RGB MOTA|RGB–IR MOTA|RGB IDSW|RGB–IR IDSW|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in NAMES:
        a, b = data["rgb"][name], data["rgbt"][name]
        lines.append(f"|{NAMES[name]}|{a['HOTA']:.2f}|{b['HOTA']:.2f}|{d[name]['HOTA']:+.2f}|"
                     f"{a['IDF1']:.2f}|{b['IDF1']:.2f}|{a['MOTA']:.2f}|{b['MOTA']:.2f}|{a['IDSW']}|{b['IDSW']}|")
    lines += ["", "完整 DetA、AssA、FP、FN 指标和逐序列成绩见两个 metrics/summary.json。检测训练仍在进行，本次评测使用运行前固定的权重快照。"]
    (OUT / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    seqs = read(OUT / "rgb/protocol.json")["sequences"]
    for seq in seqs:
        gt = ROOT / "data/processed_v1/mot/rgb/test" / seq / "gt/gt.txt"
        assert gt.read_bytes() == (ROOT / "data/official_gt/GT/GT_00/test" / seq / "gt/gt.txt").read_bytes()
    for mode in GPUS:
        assert sha(OUT / mode / "weights/rgb_best.pt") == checkpoints[mode]["sha256"]
        assert len(list((OUT / mode / "trackers").glob("*/data/*.txt"))) == 240
    dump(OUT / "integrity.json", dict(status="PASS", sequences=30, frames=12590, trajectory_files=480))
    archive = OUT / "yolo11m_tracking_results.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for n in ["comparison.md", "comparison.json", "snapshot_sources.json", "integrity.json"]:
            bundle.write(OUT / n, n)
        bundle.write(__file__, "source/evaluate_vttiny_m_pair.py")
        for mode in GPUS:
            for n in ["protocol.json", "checkpoint.json", "reid_encoder.json"]:
                bundle.write(OUT / mode / n, f"{mode}/{n}")
            for folder in ["metrics", "trackers"]:
                for f in (OUT / mode / folder).rglob("*"):
                    if f.is_file():
                        bundle.write(f, f.relative_to(OUT))
    dump(OUT / "delivery_manifest.json", dict(sha256=sha(archive), bytes=archive.stat().st_size))
    print((OUT / "comparison.md").read_text(), flush=True)


def main():
    prepare()
    dump(OUT / "state.json", dict(status="RUNNING", stage="inference"))
    tasks = []
    for mode in GPUS:
        with (OUT / mode / "inference.log").open("ab") as log:
            tasks.append(subprocess.Popen([sys.executable, "-u", __file__, "--inference", mode], stdout=log, stderr=subprocess.STDOUT))
    statuses = [task.wait() for task in tasks]
    if any(statuses):
        raise RuntimeError(f"Inference exit codes {statuses}. See per-mode inference.log.")
    jobs = [(mode, name) for mode in GPUS for name in NAMES]
    with ProcessPoolExecutor(max_workers=6, max_tasks_per_child=1, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(track, job) for job in jobs]
        for count, future in enumerate(as_completed(futures), 1):
            print("TRACK COMPLETE", future.result(), count, flush=True)
            dump(OUT / "state.json", dict(status="RUNNING", stage="tracking", completed=count, total=len(jobs)))
    dump(OUT / "state.json", dict(status="RUNNING", stage="evaluation"))
    summarize()
    dump(OUT / "state.json", dict(status="COMPLETE"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference", choices=list(GPUS))
    args = parser.parse_args()
    if args.inference:
        inference(args.inference)
    else:
        try:
            main()
        except Exception:
            dump(OUT / "state.json", dict(status="FAILED", traceback=traceback.format_exc()))
            raise
