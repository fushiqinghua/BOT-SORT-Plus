# 第二轮 · A2 弱起轨和身份准入

实跑代码提交为 `39baaf2d676453d0877dc045a7b9f9d6b6836d7e`，在第一轮 `3698607` 上增加两阶段弱起轨确认 E 和旧身份竞争检查 I。两轮沿用冻结的 YOLO11-M RGB–红外融合检测器和微调 RGB ReID，评价使用官方 RGB 跟踪真值、七类合并、TrackEval 的 COMBINED_SEQ。官方 30 段曾参与选模和门限校准。

| 阶段 | 记录与结果 |
|---|---|
| 第一轮 | [A/B 四组消融、两项事件统计与 A 的错因](../v0.1-ab/README.md)。 |
| A2 的设计与实现 | [模块设计](../../../../docs/botsort_plus/A2_DESIGN_AND_REVIEW.md) · [配置 12–26](../../../../docs/botsort_plus/A2_CONFIGS.md) · [实跑测试记录](logs/tests_core.log)。 |
| 三段验收，再扩到 30 段 | baseline_ref 对 A2_shadow，B_ref 对 A2B_shadow，两对在各自阶段逐文件核对。对应 [3 段结果](three_sequences/paired_bytes.json) 与 [30 段结果](full30/paired_bytes.json)。 |
| 15 组全量消融 | [中文结果](全量消融.md) · [联合指标](full30/metrics/comparison.csv) · [逐序列指标](full30/metrics/summary.json) · [450 份轨迹](full30/trackers) · [逐帧事件](full30/events) · [事件原文 SHA-256](full30/event_hashes.json)。另有 12 份三段验收轨迹及对应日志。 |
| 对照官方 GT 的错误归因 | [三组诊断报告](diagnosis/错因分析.md) · [补回目标清单](diagnosis/analysis/new_targets.csv) · [误报来源](diagnosis/analysis/paired_fp_breakdown.csv) · [逐次换号](diagnosis/analysis/switches.csv)。 |
| 典型序列 | [八个案例](diagnosis/cases/案例索引.md)涵盖七段视频。每例提供关键帧框图、全段时间线、决策事件和候选证据，所选窗口共 167 张可直接浏览的逐帧框图。 |

### 跨版本比较

第一轮 B 的 FP/IDSW 为 15,750/380，第二轮 B_ref 为 15,753/381。只有 DJI_0061_2 的两版 B 输出不同。baseline_ref 与第一轮 baseline 在 30 段上完全一致。分析 A2 的收益与代价请用同轮 B_ref。[两轮轨迹比较](full30/historical_refs.json)

| 同轮方法 | HOTA ↑ | IDF1 ↑ | MOTA ↑ | FP ↓ | FN ↓ | IDSW ↓ |
|---|---:|---:|---:|---:|---:|---:|
| B_ref | 38.727 | 43.862 | 25.675 | 15,753 | 117,482 | 381 |
| A2B · 0.18 | 38.752 | 43.953 | 23.465 | 23,682 | 113,344 | 563 |
| A2B_score020 · 0.20 | 38.816 | 44.095 | 23.698 | 23,090 | 113,533 | 547 |

### 在仓库根目录复核

使用固定版本 TrackEval 复算 15 组共 465 行逐序列及汇总指标后，三项计数与服务器逐项一致。三组重点方法另外独立复算了 90 份 CLEAR 结果和新增覆盖帧数。[复算记录](reproduction_check.json)

官方 GT 目录使用 `<序列>/gt/gt.txt` 布局，TrackEval 版本及环境依赖见[基线说明](../../../../README.md)。以下首条命令独立复算三组的逐序列 FP、FN、IDSW 与覆盖得失。第二条用 TrackEval 复算全部 15 组。

~~~bash
R=results/vttiny/iterations/v0.2-a2
python "$R/diagnosis/scripts/verify_results.py" --gt-root "$VTTINY_GT_FOLDER"
python tools/evaluate_botsort_plus.py --run-dir "$R/full30" \
  --trackers baseline_ref B_ref A2_E A2_I A2 A2B_E A2B_I A2B A1_ref A1B_ref \
    A2B_score_window A2B_fixed_window A2B_score020 A2_shadow A2B_shadow \
  --reference-tracker B_ref --gt-root "$VTTINY_GT_FOLDER" \
  --trackeval-root "$TRACKEVAL_ROOT" --output-dir outputs/v02_trackeval
~~~

要生成完整的逐帧证据库、配对统计与案例图，`VTTINY_FRAME_ROOT` 指向官方数据根目录。使用新的输出目录运行。

~~~bash
python "$R/diagnosis/scripts/build_evidence.py" outputs/v02_evidence \
  --gt-root "$VTTINY_GT_FOLDER" --categories "$VTTINY_FRAME_ROOT/annotations/instances_00_test2017.json"
python "$R/diagnosis/scripts/summarize.py" outputs/v02_evidence
python "$R/diagnosis/scripts/render_timeline.py" --output outputs/v02_cases --all-frames
python "$R/diagnosis/scripts/render_timeline.py" --output outputs/v02_rgb --all-frames --images-root "$VTTINY_FRAME_ROOT"
~~~

归档中的关键帧图采用轨迹与真值框的裁剪坐标。逐帧可视化可在本地叠加官方 RGB 图像。完整标注、原图、权重和逐框外观嵌入由实验者在本地提供。轨迹与事件保留原始数值和字节指纹，运行协议中的服务器目录已替换为路径占位符。各序列事件原文可用标准 gzip 解压。[官方数据入口](https://github.com/xuqingyu26/HGTMT)

`run_ablation.py` 按三段验收、全量回放、联合评估的顺序运行。`a2_gt_ablation.py --gt-root ... --output ...` 可从归档轨迹重算全量逐目标表。`report_a2_ablation.py` 保存了原服务器目录布局下生成历史交接包的过程。如需从完整证据库重建中文 RGB 案例和事件，请将重建结果放在新的 `工作目录/analysis` 下，依次运行 `diagnosis/scripts/render_cases.py 工作目录 --images-root ...` 和 `diagnosis/scripts/case_events.py 工作目录`。绘图默认使用 Noto CJK 字体，也可用 `VTTINY_CJK_FONT` 指定中文字体文件。
