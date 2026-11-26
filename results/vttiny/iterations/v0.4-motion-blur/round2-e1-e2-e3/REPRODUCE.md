# 第二批实验复现说明

运行依据为 Git 提交 `c3e142e6c11df0f2a7e8a84bede8556c7dabec1e`。使用 Python 3.11、BoxMOT 16.0.11、固定 TrackEval `12c8791b303e0a0b50f753af204249e622d0281a`，以及原实验的冻结检测、RGB ReID 缓存、解码 RGB 图像和官方 GT。原运行使用 `redetr` 环境与既有 `TRACKING_RUNTIME_ROOT`、`PYTHONUSERBASE` 设置。预测和只读旁路运行仅读取冻结预测输入，GT 由独立审核过程读取。

## 目录和输入

`SUITE` 为原 `botsort_plus_v0_4_motion_blur` 实验目录。原首轮在 `$SUITE/BOT-SORT-Plus/outputs/diagnostics_v0.4_full30_round1`。用独立工作副本在 `$SUITE/BOT-SORT-Plus-followup-c3e142e6` 检出上述提交，保留原 `configs/diagnostics/motion_blur_v0.4.local.json` 指向实际冻结输入。E2 同时使用 `$SUITE/round1/d3_validate_output_only_v1/variant_full` 和 `$SUITE/round1/d3_output_only_comparison_v1/full/trackeval` 的原实验产物与哈希。原始完整状态流包括 30 份 `.jsonl.gz` 和序列清单。

下列命令的 `RUN` 指向独立的新实验输出根。将本目录的 `run_p0.py` 和 `E1`、`E2`、`E3` 源文件复制到 `RUN` 下，保留相对位置。使用脚本生成新的输出目录。公开 JSON 和日志中的 `${TRACKING_ROOT}`、`${REDETR_ENV}` 是展示用路径占位符。用于哈希核验和复算的是原实验字节或重新生成的运行产物。机器路径遮盖和压缩记录见 [发布清单](EVIDENCE_MANIFEST.json)。

## P0 和 E1

`run_p0.py` 依次核验首轮等价性、重新核对当前输入与像素、复算基线指标，再调用分支的 `tools/prepare_v04_lost_cohort.py` 生成 E1-A。E1 的只读探针在新副本中安装。先过一个序列的重放与矩阵重算，才进入全量。

```bash
python "$RUN/run_p0.py" "$SUITE" "$RUN"
python "$RUN/E1/run_shadow.py" stage --repo "$REPO"
python "$RUN/E1/run_shadow.py" pilot --repo "$REPO"
python "$RUN/E1/audit_e1.py" pilot --repo "$REPO"
python "$RUN/E1/run_shadow.py" full --repo "$REPO"
python "$RUN/E1/audit_e1.py" audit --repo "$REPO"
python "$RUN/E1/audit_e1.py" compare --repo "$REPO"
python "$RUN/E1/check_e1_edges.py" --repo "$REPO"
python "$RUN/E1/summarize_e1.py" audit
```

`REPO` 为上述独立的执行提交副本。用 `summarize_e1.py compare --decision ... --detail ...` 写入本次决策和依据。实际回传见 [E1-B 报告](E1/audit_analysis/E1_failure_report.md) 和 [E1-C 决策](E1/compare_analysis/E1_decision.md)。

## E2

E2 的运行器按上述目录布局定位资料，冒烟通过后开始全量。底层 [坐标脚本](E2/e2_boxes.py) 和 [评测脚本](E2/e2_evaluate.py) 也可传入显式输入路径。运行器保留 191 个提名和所有原生跟踪行。

```bash
python "$RUN/E2/run_e2.py" smoke
python "$RUN/E2/run_e2.py" full
python "$RUN/E2/finalize_e2.py" "$RUN/E2"
```

## E3

`FROZEN` 为首轮完整实验根，`HISTORICAL_EVAL` 为上述 `round1/d3_output_only_comparison_v1/full/trackeval` 历史评测目录，`TRACKEVAL` 为固定源码目录。E3-A 使用实际通过的 `audit_native_fp_retry1.py`。复核样例和信号分析依赖 E3-A 重新生成的匹配及因果特征表。其逐文件参数可见已归档脚本。

```bash
python "$RUN/E3/audit_native_fp_retry1.py" --repo "$REPO" --frozen-root "$FROZEN" \
  --trackeval "$TRACKEVAL" --historical-eval "$HISTORICAL_EVAL" \
  --p0 "$RUN/P0/verification.json" --output "$RUN/E3/A"
python "$RUN/E3/stage_review.py" --repo "$REPO" --frozen-root "$FROZEN" \
  --input "$RUN/E3/A" --output "$RUN/E3/B"
python "$RUN/E3/analyze_signals.py" --e3a "$RUN/E3/A" --e3b "$RUN/E3/B" --output "$RUN/E3/C"
```

[模型辅助核对表](E3/B/E3_machine_assisted_review.csv) 保留此轮的样例判断，[正式复核表](E3/B/E3_manual_review.csv) 的标签均待人工填写。复核表中的图卡路径相对于生成的 `E3/B` 工作目录。重新跑 E3-C 复现代理检验，E3 的正式可分性仍取决于人工确认。

## 逐行证据

体积较大的逐行表以 `.csv.gz` 保存，各结果目录保留可直接查看的汇总。可以直接用支持 gzip 的 CSV 读取器研读。原始 E3 全行 CLEAR、因果特征缓存、原始图像和重放输入引用保留在实验环境。上述脚本可按原数据与固定状态重新生成这些中间产物。
