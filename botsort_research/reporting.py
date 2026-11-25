"""Compare completed runs without mixing detection caches or sequence subsets."""
import csv
from pathlib import Path

from .io import dump_json, load_json, sha256


def summarize_runs(run_dirs, output):
    if not run_dirs:
        raise ValueError("At least one run is required")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Use a new comparison directory: {output}")
    loaded, reference = [], None
    for run_dir in run_dirs:
        root = Path(run_dir).resolve()
        experiment = load_json(root / "experiment.json")
        if experiment.get("status") != "COMPLETE":
            raise ValueError(f"Incomplete run: {root}")
        summary = load_json(root / "metrics/summary.json")
        if summary.get("experiment_manifest_sha256") != sha256(root / "experiment.json"):
            raise ValueError(f"Metrics do not match the current experiment manifest: {root}")
        gt_hashes = summary.get("gt_sha256")
        image_hashes = experiment.get("image_manifest_sha256")
        if not isinstance(gt_hashes, dict) or set(gt_hashes) != set(experiment["frames"]) or not all(gt_hashes.values()):
            raise ValueError(f"Missing or incomplete GT fingerprints: {root}")
        if not isinstance(image_hashes, dict) or set(image_hashes) != set(experiment["frames"]) or not all(image_hashes.values()):
            raise ValueError(f"Missing or incomplete image manifest fingerprints: {root}")
        identity = (experiment["cache_sha256"], experiment["frames"], gt_hashes, image_hashes)
        if reference is None:
            reference = identity
        elif identity != reference:
            raise ValueError("Runs differ in selected sequences, detection caches, frame counts, GT or images; compare identical inputs")
        loaded.append((root, experiment, summary))
    # Prefer the first SOF native control; never subtract archived 30-sequence
    # scores from a 21-sequence development comparison.
    baseline = None
    for root, experiment, summary in loaded:
        for row in summary["results"]:
            if row["sequence"] == "COMBINED_SEQ" and row["tracker"] == "native_iou" and experiment["cmc"] == "sof":
                baseline = row
                break
        if baseline is not None:
            break
    rows = []
    metrics = ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "FP", "FN", "IDSW")
    for root, experiment, summary in loaded:
        for value in summary["results"]:
            if value["sequence"] != "COMBINED_SEQ":
                continue
            timing = [t for t in experiment["timing"] if t["variant"] == value["tracker"]]
            seconds = sum(t["association_seconds"] for t in timing)
            frames = sum(t["frames"] for t in timing)
            row = {"run": root.name, "variant": value["tracker"], "cmc": experiment["cmc"],
                   **{k: value[k] for k in metrics}, "association_fps": frames / seconds if seconds > 0 else None}
            row.update({"delta_"+k: value[k] - baseline[k] if baseline else None for k in ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "FP", "FN", "IDSW")})
            rows.append(row)
    if not rows:
        raise ValueError("No COMBINED_SEQ metrics found")
    output.mkdir(parents=True)
    with (output / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    columns = ("run", "variant", "cmc", "HOTA", "DetA", "AssA", "IDF1", "MOTA", "FP", "FN", "IDSW", "delta_HOTA", "delta_IDF1", "delta_MOTA")
    lines = ["# 实验汇总", "", "同一检测缓存、GT 和序列集合的 TrackEval COMBINED_SEQ。差值基准为第一个 SOF + native_iou 对照；无此对照时不计算差值。", "",
             "|"+"|".join(columns)+"|", "|"+"|".join(["---"]*len(columns))+"|"]
    def format_value(value):
        if value is None:
            return "—"
        return f"{value:.4f}" if isinstance(value, float) else str(value)
    lines += ["|"+"|".join(format_value(row[k]) for k in columns)+"|" for row in rows]
    lines += ["", "association_fps 仅包含跟踪 update，不含检测、特征提取、图像解码和指纹计算；不能当作端到端实时帧率。",
              "旧 official test 已参与历史模型选择，此处为回顾性开发比较。", ""]
    (output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    result = {"interpretation": "retrospective_development_only", "results": rows,
              "inputs": [{"run_dir": str(root), "experiment_sha256": sha256(root / "experiment.json"),
                          "metrics_sha256": sha256(root / "metrics/summary.json")} for root, _, _ in loaded]}
    dump_json(output / "comparison.json", result)
    return result
