# 评分来源与核验

归档按检测档位、配置名和序列合并六份评分凭据。筛选 `sequence=COMBINED_SEQ` 可查看两段联合结果。字段 `FP`、`FN`、`IDSW` 为事件数，`TP` 为官方 CLEAR 匹配数。HOTA、DetA、AssA、IDF1、MOTA 单位为百分数，其差值用百分点。配置表的 `first_scored` 和 `score_receipts` 可定位原始数据，`both_tracks_equal_to_reference` 记录两段都与配对对照逐字节相同。

| 来源 | 原评分规模 | 本归档新增计分配置 |
| --- | ---: | ---: |
| [01 强弱检测](evidence/01-detector-transfer.json) | 8 | 8 |
| [02 C2 与④](evidence/02-c2-and-bam.json) | 6 | 4 |
| [03 跨分支复跑](evidence/03-branch-replay.json) | 169 | 161 |
| [04 低分 R、④和 M2](evidence/04-low-update-and-m2.json) | 16 | 8 |
| [05 M2 错因与辅助关联](evidence/05-m2-forensics-and-aux.json) | 15 | 5 |
| [06 仅低分 M2](evidence/06-m2-low-stage.json) | 12 | 2 |

共享配置在多份凭据中具有相同的逐段轨迹哈希、输出行数和得分。所有配置的 TP＋FN 均与同一官方真值行数一致，TP＋FP 与对应 MOT 输出行数一致。强组后五份凭据的检测权重和 RGB ReID 指纹完全一致。原始检测和特征指纹、运行范围及校验信息见 [评估协议](evidence/protocol.json)。原实验中固定相同参考的配对差值保留全精度，其他研究分支的共享参考可用 `detector + arm + sequence` 联合索引。

## 原分支与实际回放来源

分支快照记录归档来源，实跑还可能固定到较早提交。主要关系如下。

| 研究 | 实际源码、参数或归档依据 |
| --- | --- |
| 原始 BoT-SORT | 当前分支从项目默认 `baseline` 的 `70f28b1` 创建。近期使用强弱检测两档。历史 M3OT RGB-T 检测权重与本轮不同，历史档案与近期冻结复跑分别解释。 |
| A 与 A2 | A 回放载体为 `3698607`，A2 为 `39baaf2`。归档分支快照分别为 `0054121` 与 `6d9c258`。 |
| B、C2、红外 | 参考早版 `ba1276a` 与后续 `775eaad` 的恢复、确认记忆与红外逻辑。原坐标红外与位置回放载体为 `53faecc`，沿用历史参数。RGB-only 使用同一位置逻辑与 RGB 像素的受控适配。 |
| GMC、ORU、长间隔、D/E | 以原研究的 `c3e142e`、`0a7121e` 和 `7d549bc` 的诊断、相机补偿及 B 联合谱系为依据，正式运行凭据标记各配置采用的源码。 |
| M1/M2/M3 | `ec7b822` 的归档源码经过 35 个文件 Git Blob 指纹核对。M2/M3 训练校准使用互异的训练场景 `1-01`、`1-04`、`2-02`、`2-05`。 |
| ③、④、⑤、学习 Q/R | 归档 `235fe40`，沿用 `c3e142e` 的冻结 BoT-SORT。M3OT 训练侧重拟合按编号分组，01/02/04/06/09 用于拟合，05/07 为训练侧核验。 |
| TCO 与 W | 分别追踪 `176f2a8`、`db20871`。TCO 门槛沿用 VT 训练侧对应归档，W 使用分支现有 S0 库存工具。 |

[分支完整提交清单](evidence/branch_inventory.json) · [逐分支归档计数](evidence/chapter_index.json) · [配置与轨迹指纹](configurations.csv)

## 运行材料

读取结果只需各章 Markdown、CSV 和 JSON。[近期低分实验](recent_m2.md) 记录三批 M2 与低分 R 的回放入口和实验布局。[回放逻辑快照](repro/) 包含本轮低分更新、M2 叠加、辅助关联及低分 NLL 接入。按历史布局复跑需要原仓库的对应版本、BoxMOT 和 TrackEval、授权取得的 M3OT 数据、已有检测/ReID 缓存与训练权重。发布材料提供结果、参数核验和实现来源。
