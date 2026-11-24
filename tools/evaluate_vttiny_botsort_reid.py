"""Evaluate native BoxMOT BoT-SORT with the two frozen RGB ReID caches."""

import os

os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")

import json
import multiprocessing
import shutil
import time
import traceback
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from types import SimpleNamespace

from vttiny_tracking_common import ROOT, OLD, dump, read, sha

BASELINE = ROOT / "outputs/rgb_test_7trackers_v1"
FINETUNED = ROOT / "outputs/rgb_test_4trackers_reid_testselect_v2"
OUT = ROOT / "outputs/rgb_botsort_reid_test_v1"
METHOD = "botsort_reid"
NO_REID = "botsort_no_reid"
ENCODERS = {"pretrained": BASELINE, "finetuned": FINETUNED}
THRESHOLDS = [0.10, 0.25, 0.40, 0.55, 0.70]
DEFAULT = 0.25
METRICS = ["HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW", "FP", "FN"]
LABELS = {"pretrained": "预训练 CLIP-VeRi", "finetuned": "VT-Tiny-MOT 微调 ReID"}


def make_plan():
    protocol = read(BASELINE / "protocol.json")
    adapted = read(FINETUNED / "protocol.json")
    assert protocol["weight_sha256"] == adapted["weight_sha256"]
    assert protocol["sequences"] == adapted["sequences"]
    assert len(protocol["sequences"]) == 30 and protocol["frames"] == 12590
    config = protocol["trackers"][NO_REID]
    assert config["track_high_thresh"] == 0.5 and config["with_reid"] is False
    assert sha(BASELINE / "weights/rgb_best.pt") == protocol["weight_sha256"]

    hashes = {}
    for seq in protocol["sequences"]:
        detection = BASELINE / "detections" / (seq + ".json")
        detection_sha = sha(detection)
        hashes[str(detection)] = detection_sha
        assert sha(FINETUNED / "detections" / detection.name) == detection_sha
        gt = ROOT / "data/processed_v1/mot/rgb/test" / seq / "gt/gt.txt"
        assert gt.read_bytes() == (ROOT / "data/official_gt/GT/GT_00/test" / seq / "gt/gt.txt").read_bytes()
        hashes[str(gt)] = sha(gt)
        old_track = BASELINE / "trackers" / NO_REID / "data" / (seq + ".txt")
        hashes[str(old_track)] = sha(old_track)
    for encoder, directory in ENCODERS.items():
        metadata = read(directory / "protocol.json")
        hashes[str(directory / "protocol.json")] = sha(directory / "protocol.json")
        checkpoint = directory / "weights/clip_veri.pt"
        assert sha(checkpoint) == metadata["reid_sha256"]
        hashes[str(checkpoint)] = metadata["reid_sha256"]
        for seq in protocol["sequences"]:
            cache = directory / "embeddings" / (seq + ".npz")
            record = read(cache.with_suffix(".json"))
            assert record["reid_sha256"] == metadata["reid_sha256"]
            assert record["detection_sha256"] == hashes[str(BASELINE / "detections" / (seq + ".json"))]
            assert sha(cache) == record["features_sha256"]
            hashes[str(cache)] = record["features_sha256"]
            hashes[str(cache.with_suffix(".json"))] = sha(cache.with_suffix(".json"))

    trials = [
        {"name": f"botsort_{encoder}_a{round(value * 100):03d}", "encoder": encoder, "appearance_thresh": value}
        for encoder in ENCODERS for value in THRESHOLDS
    ]
    plan = {
        "created": time.time(),
        "dataset": "VT-Tiny-MOT official RGB test, all 30 sequences and 12590 frames",
        "baseline": str(BASELINE),
        "fine_tuned_source": str(FINETUNED),
        "selection": ["highest unrounded HOTA", "highest IDF1", "fewest IDSW", "nearest 0.25", "lowest threshold"],
        "protocol_note": "Official test set used for threshold selection and reporting. Adapted checkpoint was also selected on official test retrieval mAP. Same grid for both encoders.",
        "fixed": "Same detector, detections, confidence thresholds, SoF camera motion compensation, IoU proximity gate 0.5, match_thresh 0.8 and track management. Native BoxMOT appearance gating and cost fusion.",
        "trials": trials,
    }
    dump(OUT / "search_plan.json", plan)
    dump(OUT / "input_sha256.json", hashes)
    return plan, protocol


def run_trial(trial):
    import run_vttiny_rgb_tracking as runner
    import boxmot.trackers.botsort.botsort as native
    import torch
    from vttiny_reid_adapter import CacheOnlyEncoder

    started = time.perf_counter()
    directory = OUT / "trials" / trial["name"]
    directory.mkdir(parents=True, exist_ok=False)
    source = ENCODERS[trial["encoder"]]
    for name, target in {
        "detections": BASELINE / "detections",
        "embeddings": source / "embeddings",
        "weights": source / "weights",
    }.items():
        (directory / name).symlink_to(target, target_is_directory=True)

    protocol = read(BASELINE / "protocol.json")
    protocol["reid_sha256"] = read(source / "protocol.json")["reid_sha256"]
    config = dict(
        protocol["trackers"][NO_REID],
        with_reid=True,
        proximity_thresh=0.5,
        appearance_thresh=trial["appearance_thresh"],
        fuse_first_associate=False,
    )
    protocol["trackers"] = {METHOD: config}
    protocol["notes"] = [
        "Native BoxMOT BoT-SORT with shared cached CLIP-ReID features.",
        "Official RGB test used for threshold calibration and final scoring.",
        "No ground truth supplied to tracking.",
    ]
    protocol["trial"] = trial
    dump(directory / "protocol.json", protocol)

    def construct(name):
        assert name == METHOD
        torch.set_num_threads(1)
        original_backend = native.ReidAutoBackend
        native.ReidAutoBackend = lambda **kwargs: SimpleNamespace(model=CacheOnlyEncoder())
        try:
            tracker = native.BotSort(
                reid_weights=directory / "weights/clip_veri.pt",
                device=torch.device("cpu"),
                half=False,
                **config,
            )
        finally:
            native.ReidAutoBackend = original_backend
        assert tracker.with_reid and tracker.appearance_thresh == config["appearance_thresh"]
        assert tracker.track_high_thresh == 0.5 and tracker.proximity_thresh == 0.5
        return tracker

    runner.OUT = directory
    runner.construct = construct
    timing = runner.track_method(METHOD)
    assert len(timing) == 30 and sum(row["frames"] for row in timing) == 12590
    linked = OUT / "trackers" / trial["name"]
    linked.mkdir(parents=True)
    (linked / "data").symlink_to(directory / "trackers" / METHOD / "data", target_is_directory=True)
    record = dict(trial, seconds=time.perf_counter() - started, sequences=30, frames=12590)
    dump(directory / "run_record.json", record)
    return record


def summarize(plan, protocol):
    payload = read(OUT / "metrics/summary.json")
    results = {row["tracker"]: row for row in payload["results"] if row["sequence"] == "COMBINED_SEQ"}
    reference = next(row for row in read(BASELINE / "baseline_results.json") if row["tracker"] == NO_REID)
    assert all(abs(results[NO_REID][key] - reference[key]) < 1e-8 for key in METRICS)
    rows = [dict(case, metrics=results[case["name"]]) for case in plan["trials"]]
    selected = {}
    for encoder in ENCODERS:
        candidates = [row for row in rows if row["encoder"] == encoder]
        selected[encoder] = min(
            candidates,
            key=lambda row: (
                -row["metrics"]["HOTA"], -row["metrics"]["IDF1"], row["metrics"]["IDSW"],
                abs(row["appearance_thresh"] - DEFAULT), row["appearance_thresh"],
            ),
        )
        best = selected[encoder]
        run_dir = OUT / "trials" / best["name"]
        dest = OUT / "selected_trackers" / encoder / "data"
        shutil.copytree(run_dir / "trackers" / METHOD / "data", dest)
        assert len(list(dest.glob("*.txt"))) == 30
        for seq, frames in protocol["sequences"].items():
            track_file = dest / (seq + ".txt")
            timing = read(run_dir / "timing" / METHOD / (seq + ".json"))
            assert timing["frames"] == len(frames)
            assert sha(track_file) == timing["prediction_sha256"]
        shutil.copy2(run_dir / "protocol.json", dest.parent / "protocol.json")
    dump(OUT / "all_trials.json", rows)
    dump(OUT / "comparison.json", {"without_reid": results[NO_REID], "selected": selected})

    def metric_line(label, value, result):
        fields = [label, value]
        fields += [f"{result[key]:.2f}" if key not in ["FP", "FN", "IDSW"] else str(result[key]) for key in METRICS]
        return "| " + " | ".join(fields) + " |"

    grid = ", ".join(f"{value:.2f}" for value in THRESHOLDS)
    lines = [
        "# VT-Tiny-MOT RGB BoT-SORT ReID 对照",
        "",
        "官方 RGB 测试集共 30 段、12,590 帧，使用 TrackEval 联合汇总。三种配置共用同一份 RGB 检测结果。"
        "开启 ReID 的两组复用前序实验的 1280 维特征缓存，BoT-SORT 对高置信度检测使用外观特征。",
        "",
        "两套编码器均搜索相同的 appearance_thresh 候选。原参数为 0.25，"
        f"本轮候选为 {grid}。按最高 HOTA、最高 IDF1、最少 IDSW、最接近原阈值和较小阈值依次选取。",
        "固定 IoU 距离接近门限 0.5、关联代价门限 0.8、相机运动补偿和轨迹管理参数。"
        "BoxMOT 首次关联使用余弦距离，未确认轨迹关联将该距离除以 2，保留实现原有计算方式。",
        "",
        "**启用 ReID 的两组均在官方测试集校准门限，微调权重亦在同一测试集选模。**"
        "无 ReID 基线使用此前固定配置。相同门限下两套编码器可直接对比，"
        "各自最佳配置与无 ReID 的差值同时包含外观分支和参数校准的影响。",
        "",
        "| 配置 | 外观门限 | HOTA↑ | DetA↑ | AssA↑ | IDF1↑ | MOTA↑ | IDSW↓ | FP↓ | FN↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        metric_line("无 ReID 原始基线", "—", results[NO_REID]),
    ]
    for encoder in ENCODERS:
        default_row = next(r for r in rows if r["encoder"] == encoder and r["appearance_thresh"] == DEFAULT)
        best_row = selected[encoder]
        displayed = [default_row] if default_row is best_row else [default_row, best_row]
        for row in displayed:
            label = LABELS[encoder] + ("（测试集选中）" if row is best_row else "（原门限）")
            lines.append(metric_line(label, f"{row['appearance_thresh']:.2f}", row["metrics"]))
    lines += ["", "全部候选及其逐序列指标见 all_trials.json、trial_ledger.jsonl 和 metrics/summary.json。"]
    (OUT / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return selected


def package():
    inputs = read(OUT / "input_sha256.json")
    assert all(sha(path) == digest for path, digest in inputs.items())
    dump(OUT / "integrity.json", {
        "status": "PASS", "inputs_unchanged": True, "no_reid_metrics_reproduced": True,
        "selected_trajectory_files": 60, "frames_per_variant": 12590, "sequences": 30,
    })
    source = OUT / "source"
    source.mkdir()
    for name in [
        "evaluate_vttiny_botsort_reid.py", "run_vttiny_rgb_tracking.py",
        "vttiny_tracking_common.py", "vttiny_reid_adapter.py", "evaluate_vttiny.py",
    ]:
        shutil.copy2(ROOT / "tools" / name, source / name)
    native = OLD / "tracking_runtime/boxmot/trackers/botsort/botsort.py"
    dump(source / "native_implementation.json", {"path": str(native), "sha256": sha(native)})
    archive = OUT / "botsort_reid_results.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name in ["comparison.md", "comparison.json", "all_trials.json", "trial_ledger.jsonl", "search_plan.json", "input_sha256.json", "integrity.json"]:
            bundle.write(OUT / name, name)
        for directory in ["selected_trackers", "metrics", "source"]:
            for entry in (OUT / directory).rglob("*"):
                if entry.is_file():
                    bundle.write(entry, entry.relative_to(OUT))
        for name in ["protocol.json", "run_record.json"]:
            for entry in (OUT / "trials").glob(f"*/{name}"):
                bundle.write(entry, entry.relative_to(OUT))
    dump(OUT / "delivery_manifest.json", {"archive": str(archive), "sha256": sha(archive), "bytes": archive.stat().st_size})


def main():
    OUT.mkdir(exist_ok=False)
    try:
        dump(OUT / "state.json", {"stage": "input_audit", "status": "RUNNING"})
        plan, protocol = make_plan()
        with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(run_trial, case) for case in plan["trials"]]
            for count, future in enumerate(as_completed(futures), 1):
                run = future.result()
                with (OUT / "trial_ledger.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(run) + "\n")
                dump(OUT / "state.json", {"stage": "tracking", "status": "RUNNING", "completed": count, "total": len(futures)})
                print("COMPLETED", run["name"], count, flush=True)
        no_reid = OUT / "trackers" / NO_REID
        no_reid.mkdir()
        (no_reid / "data").symlink_to(BASELINE / "trackers" / NO_REID / "data", target_is_directory=True)
        dump(OUT / "state.json", {"stage": "evaluation", "status": "RUNNING"})
        from evaluate_vttiny import evaluate

        evaluate("rgb", "test", OUT / "trackers", [NO_REID] + [row["name"] for row in plan["trials"]], OUT / "metrics")
        summarize(plan, protocol)
        package()
        dump(OUT / "state.json", {"status": "COMPLETE", "reid_trials": len(plan["trials"])})
    except Exception:
        dump(OUT / "state.json", {"status": "FAILED", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
