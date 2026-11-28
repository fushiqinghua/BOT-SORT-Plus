# 原始 BoT-SORT 开关 W0 · 2026-11-28 全量补充消融

在官方开发评测的 30 段、12,590 帧上，重新运行原始 BoT-SORT 关闭和开启 W0 两组。W0 采用原锚点约束与正式状态反馈。固定检测、RGB ReID 和原生 SOF GMC 缓存。三段技术校验通过后执行全量，原始关闭组的 30 段 MOT 与历史原始 BoT-SORT 逐字节一致，开启组在每段首次弱更新前保持相同的状态摘要。B、BG 使用 R4.1 已有的四组完整轨迹，六组用同一份官方真值和冻结 TrackEval 统一重评。 [完整运行与输入凭据](run_provenance.json)

## 各底座启用 W0 的增量

每行是同一底座开启 W0 减关闭 W0。HOTA、IDF1 和 MOTA 的差值以百分点计。

| 底座 | ΔHOTA | ΔIDF1 | ΔMOTA | ΔFP | ΔFN | ΔIDSW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原始 BoT-SORT | +0.2402 | +0.2657 | +0.0367 | +664 | −738 | +8 |
| B | +0.1724 | +0.1628 | −0.0245 | +554 | −508 | −2 |
| BG | +0.1743 | +0.1544 | −0.0378 | +578 | −510 | 0 |

在原生 SOF GMC 和 RGB ReID 下，原始 BoT-SORT 直接启用 W0 也得到覆盖和 HOTA 收益。B、BG 上的覆盖增量较小，换号代价也较低。三套底座分别都有 18 段 HOTA 提升、5 段持平、7 段下降。BG 与 B 的增量差反映自适应 GMC、低分双重确认和关联证据的联合配置。本批继续沿用原锚点和正式状态反馈的 W0 作为定型候选。

[六组绝对指标和增量](REPORT.md) · [总体数据](comparison.csv) · [180 条逐序列记录](paired_per_sequence.csv) · [TrackEval 原格式指标](trackeval/summary.json)

## DJI_0061_2 原因与状态对照

- 原始开关组换号 **158→174**，实际是新增 34 个时点并消失 18 个。34 个新增当帧均为原生关联。28 个共用关闭组的检测行，其中 23 个改变当前轨迹归属，5 个当前轨迹的出生记录相同、前次真值关联历史改变。另有 6 个关闭组当帧漏检。
- 原始只输出诊断发布 **117 个弱框**，**0 次**正式弱状态更新。原生事件与正式状态摘要均与关闭组一致，换号仍为 158 次。仅 GT139 的换号从第 220 帧移到第 221 帧。34 个完整 W0 的新增时点中，另外 33 个随正式状态反馈及后续演化出现。
- B、BG 各自换号 **178→182**，新增 31 个时点、消失 27 个。原始新增的 34 个真值帧在两组开启时均有 16 个延续旧 ID、12 个换号、6 个漏检。本段两底座对应开关的轨迹与状态逐字节一致，240 帧均使用相同原生 SOF 矩阵，自适应 GMC 和低分双重确认没有额外动作。

[逐帧错因分析](dji_0061/ANALYSIS.md) · [逐帧换号对照](dji_0061/tables/switch_comparison.csv) · [只输出凭据](dji_0061/emit_only_receipt.json)

## 复现

使用与归档一致的 BoxMOT 16.0.11 环境和冻结资产。以下命令从仓库根目录执行。将 `/path/to/…` 替换为实验环境中的源码和私有运行目录。全量需要 R4.1 的完整 `run/full`，包括 `prediction_bundle.json`、源轨迹与评测凭据。冻结 Bundle 的路径在本机有效时，可直接复用。若已迁移，需先在实验环境恢复冻结输入和路径映射。

```bash
REPRO=results/vttiny/wem_r4_2/native_w0_20261128/repro
python -B "$REPRO/run_native.py" run --source /path/to/verified-r41-source \
  --archived /path/to/r41/run/full --output /path/to/private-native/run
python -B "$REPRO/summarize.py" /path/to/private-native/run --r41 /path/to/r41/run \
  --historical-native results/vttiny/iterations/v0.4-motion-blur/gmc/02-factorial
```

`run_native.py` 与原运行源码 SHA-256 一致。它先校验三段，再跑原始两组并重评三套底座。`summarize.py` 对账旧参考和新轨迹，再生成全量表。

已有完整运行时，可重建 `DJI_0061_2` 的 CLEAR 匹配和单段只输出诊断。`--r42-source` 指向包含 R4.2 输出隔离实现的冻结源码。已有单段诊断的完整输出时可传 `--emit-run` 复用。

```bash
python -B "$REPRO/reproduce_dji_0061.py" --native-run /path/to/private-native/run \
  --r41-run /path/to/r41/run/full --repo-root /path/to/verified-r41-source \
  --r42-source /path/to/verified-r42-source --output /path/to/private-case
python -B "$REPRO/analyze_dji_0061.py" --inputs /path/to/private-case/inputs \
  --output /path/to/private-case/tables
```

仅凭 GitHub 归档也能核验文件哈希、六组指标、所有逐序列差值、逐帧事件与只输出凭据。

```bash
python -B "$REPRO/verify_public.py"
```

发布内容由指标、按帧对齐的派生表、摘要、复现脚本组成。原始图像、真值、检测与 ReID 缓存、MOT 和完整事件流均保留在实验环境。清单记录公开文件实际 SHA-256 及可对应的原件 SHA-256。 [发布清单](manifest.json)
