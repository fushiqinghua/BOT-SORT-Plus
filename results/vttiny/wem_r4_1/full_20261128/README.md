# W R4.1 · 2026-11-28 全量实验

十组各覆盖 VT-Tiny-MOT 官方 30 段、12,590 帧。技术校验、跟踪、TrackEval 和错因归因均已完成。关闭组复现既有 B 与 BG 归档。官方 30 段沿用开发评测口径。 [运行与输入凭据](run_provenance.json)

## 结果和结论

- W0 相对 B、BG 各自的关闭组，HOTA 分别增加 0.1724、0.1743 个百分点，IDF1 增加 0.1628、0.1544 个百分点。漏检分别减少 508、510，误报增加 554、578，MOTA 分别下降 0.0245、0.0378 个百分点。
- W1 与 W2 在各自底座上均复现 W0 的全部轨迹。每套底座的 769 条评分候选均有可计算的运动证据并通过门限，运动联合排序与软 RGB 排序保持相同指派。
- W3 相对 W2 提交更少的弱框，却在两套底座上分别净增 57 个误报与 27 个漏检，换号持平。W0 可作为这批实验的几何对照，W3 的结果支持保留当前关闭设置。
- `DJI_0073_3` 的净增 64 个误报主要来自旧 ID 18 的原生重连与后续延续。`DJI_0327_3` 的 16 个新增漏检中，5 个为门控当帧拒掉正确弱框，11 个来自后续原生低分续接和 B 恢复变化。

## 数据入口

| 文件 | 内容 |
| --- | --- |
| [完整报告](report/REPORT.md)与[聚合指标](report/comparison.csv) | 十组 TrackEval COMBINED 绝对指标和各自底座差值 |
| [正式 TrackEval 指标](trackeval/summary.json) | 300 项逐段记录和 10 项 COMBINED，另有每组原格式明细 |
| [逐序列对比](report/paired_per_sequence.csv)与[组件增量](report/component_per_sequence.csv) | 300 条配对、180 条逐序列组件增量 |
| [相机组合差](report/interactions.csv)、[门控诊断](report/exposure.csv)、[提交归因](report/attribution.csv) | 与自身关闭组相比的变化、候选曝光和状态闭环影响 |
| [全量逐序列归因](attribution/per_sequence.csv)、[弱框提交](attribution/committed_clear.csv)、[B 恢复变化](attribution/changed_B_events.csv) | CLEAR 匹配和已有事件键得到的派生记录 |
| [两段逐帧分析](frame_audit/ANALYSIS.md)、[逐帧计数](frame_audit/tables/frame_counts.csv) | 官方真值对照、误报来源、外观门与 B 恢复证据 |

归档包含派生的帧号、匹配 ID、重叠度和事件记录。原始图像、真值文件、特征缓存、权重、原始 MOT 轨迹和本机配置保留在实验环境。逐帧 CSV 省略框坐标。[发布文件清单](manifest.json)分别记录原始 SHA-256 与 GitHub 文件 SHA-256，已记录因 LF 归一或去除坐标而产生的差别。

## 复现

全量入口为 `tools/run_wem_r41.py`。在实验主机按手册准备冻结资产和本地配置，再执行 `init`、`check`、`full`。[运行手册](../../../../docs/botsort_plus/wem_r4_1/LOCAL_RUNBOOK.md)

两段逐帧归因在保存了 `full/` 运行结果和冻结资产的主机上离线重建。下列命令从仓库根目录运行，`<full运行目录>` 指向本地完整批次，两个输出目录选在本地实验空间。

```text
python -B results/vttiny/wem_r4_1/full_20261128/repro/fetch.py --batch <full运行目录> --output <本地中间目录>
python -B results/vttiny/wem_r4_1/full_20261128/repro/analyze.py --inputs <本地中间目录> --output <派生表目录> --omit-boxes
```

本地中间目录含用于对齐的坐标与事件。对外核验仅需当前归档，命令可直接复核文件哈希、绝对指标、配对行和两段逐帧计数。

```text
python -B results/vttiny/wem_r4_1/full_20261128/repro/verify_public.py
```
