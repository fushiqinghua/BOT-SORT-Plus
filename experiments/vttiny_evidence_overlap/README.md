# 将上帧重叠面积降权接入长间隔重连修正版

四种方法从同一批冻结输入重新开始跟踪。修正版选用归档主版本 `evidence_both`，该版本结合 `B_early_moderate`、自适应 GMC、低分匹配双重确认，以及长间隔重连的观测证据与代价检查。单独④使用 `native_motion_345_v1` 的 `bam_overlap`。组合在同一修正版的低分更新阶段调用这份已验证的实现。

| 方法 | 运行名 |
|---|---|
| 原始 BoT-SORT | `native` |
| 单独④ | `bam_overlap` |
| 长间隔重连修正版 | `evidence_both` |
| 修正版＋④ | `evidence_both_bam_overlap` |

组合中，低分候选先通过原有门限和相机运动双重确认。若轨迹上一帧确有匹配观测，并且该观测框与其他活动轨迹观测框重叠，按当前预测 IoU 与上帧未覆盖面积比例的乘积，对原生卡尔曼的均值修正降权。用于计算上帧覆盖面积的是原始匹配检测框，其交集取并集。每次降权都会对照本帧实际接受的低分匹配记录。B 恢复使用独立的重新激活入口。

在服务器 `redetr` 环境运行，默认输出写入脚本所在目录。可创建一个新的输出目录并设置 `VT_COMBO_ROOT`，同时将 `VT_INTEGRATIONS` 指向既有 integrations 目录。依赖包括 `native_motion_345_v1`、`b_gmc_forensics_r1/live_repo`、LightGlue 固定权重、冻结检测与 ReID 缓存、原始 RGB 帧及 TrackEval。具体路径在 `joint_settings.py`。

```bash
python -Bu experiment.py replay --gpu 5
python -Bu experiment.py score
```

四个预测进程相互独立。两组自适应 GMC 使用同一张 GPU，另外两组使用 CPU。评分读取官方 GT，逐项核对检测指纹、特征数组、编码器、RGB 像素、原生配置、修正版逐帧相机变换和三个历史对照。原始轨迹位于 `runs/<运行名>/full/trackers/<运行名>/data`。压缩逐帧低分记录位于组合运行的 `full/diagnostics345/bam_overlap`。汇总、逐序列统计和已核验的换号事件见[完整实验报告](results/实验报告.md)。
