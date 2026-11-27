# 三段新增误报的逐帧诊断

以既有的 `evidence_both` 和 `evidence_both_bam_overlap` 为对照，在 DJI_0028_3、DJI_0061_2、DJI_0169_3 重新回放并记录输出和固定检测的行号。检查两组输出与其全量实验中的这三段逐字节相同。预测进程只读冻结检测、ReID 和图像。逐帧误报、漏检和换号由独立评估进程按官方 RGB 真值的 CLEAR 匹配计算。

在同一序列、帧、检测行上对齐输出，分别记录输出消失、新增和真值配对状态变化。同一帧的误报可以同时增加和消失，因此本轮的毛增减按输出行统计。连续输出片段从 B 恢复帧开始，直到该轨迹出现输出缺口或再次由 B 接回。B 片段归属描述输出的直接来源。反事实回放另外检查该恢复或此前的状态更新对整段指标的实际影响。直接由原生高分关联启动的片段可能沿用过去由 B 修改过的状态。

## 诊断性干预与通用方案

- `probe_weight` 在指定轨迹和帧撤回④的更新缩放，保留当帧已接受的低分匹配。正式改进不使用这些序列、帧和 ID。
- `probe_B` 从当帧 B 决定里移除指定的轨迹与检测配对。它只检查一次接回是否触发后续误报，下一帧可重新计算接回，且不重新分配当帧 B 配对。
- `output_only` 只把发生④的该帧输出框换回原生后验框。卡尔曼状态和在线关联仍用原④。逐帧比对确认路径、轨迹 ID 和检测行不变。
- `velocity` 用④降权更新目标中心和尺寸，速度及尺寸变化率使用卡尔曼原生后验。协方差沿用原版。
- `weight_floor` 把④的权重下限设为 0.5。
- `b_guard` 对最近五帧内受过④、末次输出为低分的 B 失联候选，当其新观测全是低分时，要求有效 RGB 外观代价不高于已有 moderate 上限 0.55，并要求至少三个观测。

后三项在给定冻结参数下逐段回放。它们的在线决策只使用当前和历史状态。焦点序列用于开发核验。全量对照仍是同一套曾用于选模的官方测试集。

## 复现与数据

服务器的实验根目录是 `${VT_PROJECT}/integrations/b_gmc_evidence_overlap_v1_forensics`。脚本使用同级 `b_gmc_evidence_overlap_v1` 的入口与冻结配置。依赖的既有实验、环境与资源仍按上轮约定布置。使用 `redetr` 依次执行以下命令。

```bash
python -Bu workflow.py replay --gpu 5
python -Bu workflow.py probe_weight,probe_B,velocity,weight_floor,b_guard --gpu 5
python -Bu workflow.py output_only --gpu 5
python -Bu workflow.py audit
python -Bu workflow.py full_velocity --gpu 5
python -Bu workflow.py score_full_velocity
python -Bu workflow.py report
```

- 完整逐检测行归因可由冻结输入重放生成。
- 连续误报片段 与 [典型轨迹时间线](results/typical_track_timelines.csv) 可按轨迹 ID 和帧查看。
- [焦点三段各方法指标](results/case_metrics.csv) 与 [干预前后逐行对照](results/intervention_effects.csv) 给出增减，并核对同一真值和帧上的换号位置是否重新出现。
- 官方真值配对、完整输出框和逐次换号均在 `results`。压缩的原始线上事件在各运行目录的 `trace`。
