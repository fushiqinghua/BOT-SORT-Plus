# W R4.2 结果记录

填写前阅读 [实验计划](EXPERIMENT_PLAN.md) 与 [执行手册](LOCAL_RUNBOOK.md)。真实运行前统一写 `NOT_RUN`，不要使用单元测试结果替代数据集指标。

## 1. 运行与复用记录

| 项目 | 实际记录 |
| --- | --- |
| 实验代码提交 | NOT_RUN |
| 完整运行输出目录（本地保存，公开时可去除路径） | NOT_RUN |
| 输入指纹／协议摘要 | NOT_RUN |
| BG_off、BG_W0 来源：复用或重新运行 | NOT_RUN |
| 复用源 R4.1 批次与验证凭据 | NOT_RUN |
| Python／NumPy／SciPy／Torch／CUDA | NOT_RUN |
| GPU、并发数 | NOT_RUN |
| 技术 smoke | NOT_RUN |
| 全量范围 | 目标为每组 30 段、12,590 输入帧；实测 NOT_RUN |
| 不完整组、重试及原因 | NOT_RUN |

## 2. 完整指标

填入 TrackEval COMBINED 结果；HOTA、DetA、AssA、IDF1、MOTA 使用百分制。保持本表五组，不只填入表现最好的组。

| 组别 | HOTA | DetA | AssA | IDF1 | MOTA | FP | FN | IDSW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BG_off | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| BG_W0 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| BG_lower_low | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| BG_W0_relaxed_anchor | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| BG_W0_emit_only | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |

## 3. 三项主比较

差值均为左组减右组。百分制指标差值使用百分点；计数为实际差值。

| 对比 | ΔHOTA | ΔIDF1 | ΔMOTA | ΔFP | ΔFN | ΔIDSW | 设计决定 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| W0 − lower_low | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| relaxed_anchor − W0 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| W0 − emit_only | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |

此外保留四个启用组相对 BG_off 的完整差值。不能将历史 B 底座的 lower_low 数值直接填成 BG_lower_low。

## 4. 干预是否真实发生

### 锚点约束包

- 因放宽成熟度而新开放的轨迹/候选数：NOT_RUN。
- 因放宽锚点与正式预测一致性而新开放的轨迹/候选数：NOT_RUN。
- 因放宽弱框与锚点一致性而新开放的候选数：NOT_RUN。
- 相同在线状态下，完整约束与放宽约束的候选或选择变化：NOT_RUN。
- 实际新增提交是否发生：NOT_RUN。

这些计数可能重叠，不应未经定义直接求和。若全量输出不变，应记录“当前数据未证实这组约束的独立增量”，不能写成必要性已证明。

### 正式状态反馈

- emit_only 原生事件与 BG_off 的一致性：NOT_RUN。
- emit_only 正式状态链与 BG_off 的一致性：NOT_RUN。
- emit_only 弱观测输出数：NOT_RUN。
- emit_only 正式弱状态更新数：应为 0；实测 NOT_RUN。
- W0 弱状态更新数：NOT_RUN。

emit_only 的状态链以原生输出作为摘要字段；其最终 MOT 可包含额外弱框。不要把状态链一致写成最终 MOT 一致，也不要将这一组称为只改变 KF 均值的消融。

报告字段分别为 `weak_outputs_emitted` 和 `weak_state_updates`。离线逐提交表中，emit_only 使用 `output_kind=emitted_measurement_no_state_feedback`、`state_committed=0`；不要将兼容事件名 `committed` 本身解读为正式 KF 更新。

## 5. 全程收益与代价

| 组别 | 弱观测输出数 | 当帧 CLEAR TP | 当帧 CLEAR FP | 当帧新增覆盖 | 全程新增覆盖 | 全程丢失覆盖 | B 恢复数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BG_W0 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| BG_W0_relaxed_anchor | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| BG_W0_emit_only | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |

检查全程“新增覆盖 − 丢失覆盖”与相对 BG_off 的 FN 减少一致。lower_low 没有 W 提交事件，其收益来自原生阈值改变，不要把 0 次 W 提交解释为没有算法作用。

## 6. 逐序列与案例

对三项主比较分别记录 HOTA、IDF1、MOTA 的提升／下降／不变段数；列出变化最大的正例和负例，并保留全部逐序列表。

案例优先从实际改变的接纳或状态分叉出发：

1. 锚点约束拒绝的候选，在放宽组中接纳后发生了什么；
2. 同一次弱观测输出后，正式反馈与 emit_only 如何影响后续原生关联和 B 恢复；
3. 降阈值多接纳的候选如何带来覆盖或误报代价。

记录序列、帧、原始检测行、旧轨迹 ID、匹配 GT（离线）、后续事件与证据路径。跨策略可能发生 ID 分叉，不能仅凭相同数字 ID 判断是同一物体。图像尚未人工核验时，应写“CLEAR 未匹配”，不要直接写“背景假目标”。

## 7. 最终判断

- 最终采用的规则与原因：NOT_RUN。
- 相对强对照的优势和明确代价：NOT_RUN。
- 已得到支持的机制解释：NOT_RUN。
- 未得到支持或当前冗余的规则：NOT_RUN。
- 失败序列、适用边界：NOT_RUN。
- 是否需要下一轮，以及只解决哪个剩余问题：NOT_RUN。

若现有 W0 仍是合适的折中，可以直接完成定型，不要求本批再创造新的最高指标。若更简单的变体更优，应接受简化证据；不要根据当前结果临时更改实验阈值后只上报最优版本。
