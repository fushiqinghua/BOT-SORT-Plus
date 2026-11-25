# 第一轮 · 弱起轨 A 与短时恢复 B

服务器实跑提交为 `3698607beb10898c00484adc8c1dcb67dd180794`，直接继承基线 `70f28b1`。A 累积低分检测证据并建轨，B 将丢失轨迹与后续观测重新关联。本轮完成四组全量实验，另用六个控制组和两段定点复查诊断 A。C 在补丁中有实现，完整的可见光–红外外观融合评测仍需要红外编码器、预处理和空间映射。[代码与设计](../../../../README_BOTSORT_PLUS.md) · [接入验收记录](acceptance.json)

| 实验记录 | 可核对的内容 |
|---|---|
| [全量实验报告](full30/实验报告.md) · [联合指标](full30/metrics/comparison.csv) | baseline、A、B、AB 复用同一检测与 RGB ReID 缓存。覆盖 30 段、12,590 帧，七类合并。 |
| [弱起轨与旧 ID 恢复](full30/event_analysis) | 起轨当帧与全寿命归因、恢复两端的 GT 配对、间隔分组及 30 段基线字节核验。 |
| [A 的逐帧错因分析](module_a/错因分析.md) · [案例框图](module_a/analysis/box_comparison.png) | 对照 870 条 GT、3,393 次弱起轨与 1,549 次换号，追踪典型序列上的起轨收益、身份竞争与长时误报。 |
| [控制组配置](module_a/configs) · [六组联合指标](module_a/controls/metrics_all/comparison.csv) · [两段定点复查](module_a/subset_check) | 分别记录起轨提交、原生确认、候选窗口、速度及丢失轨迹列表对结果的影响。 |
| [输入与轨迹指纹](full30/protocols) · [120 份主实验轨迹](full30/trackers) · [逐帧事件](full30/events) | 每条轨迹按原始 MOT 字节归档。事件按序列无损压缩为 JSONL.GZ。控制组轨迹及日志使用同样的目录结构。 |

## 用官方真值复算

在仓库根目录设定 `VTTINY_GT_FOLDER` 为官方 RGB 跟踪标注目录，子目录格式为 `<序列>/gt/gt.txt`。`TRACKEVAL_ROOT` 指向冻结的 TrackEval 源码。评测器检查每条预测的 SHA-256，再联合汇总 HOTA、DetA、AssA、IDF1、MOTA、FP、FN 和 IDSW。版本和安装方法见[基线说明](../../../../README.md)。

~~~bash
R=results/vttiny/iterations/v0.1-ab/full30
python tools/evaluate_botsort_plus.py --run-dir "$R" --trackers baseline A B AB \
  --gt-root "$VTTINY_GT_FOLDER" --trackeval-root "$TRACKEVAL_ROOT" \
  --output-dir outputs/v01_trackeval --reference-tracker baseline
~~~

要复算两项事件统计，将 `full30` 复制到自己的输出目录，将以下 `--out` 指向该副本。该步骤会写入副本内的 `event_analysis/`。

~~~bash
R=results/vttiny/iterations/v0.1-ab/full30
mkdir -p outputs
cp -R "$R" outputs/v01_events
python "$R/analyze_botsort_ab_events.py" --out outputs/v01_events --repo "$PWD" \
  --original results/vttiny/yolo11m_rgb_rgbt/rgbt --gt-folder "$VTTINY_GT_FOLDER" \
  --low-file results/vttiny/botsort_error_analysis/gt_frames.csv.gz
~~~

所有数值均采用当轮官方测试集开发口径。运行报告中的服务器目录已替换为路径占位符。评估审计保存实跑时的协议指纹，各协议中的检测、ReID、代码和 MOT 输出指纹可用于逐一核对。数据原帧、完整官方标注和模型权重按原基线的方式由实验者本地提供。[官方数据入口](https://github.com/xuqingyu26/HGTMT)
