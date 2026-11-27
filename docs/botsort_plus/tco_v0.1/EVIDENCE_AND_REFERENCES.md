# TCO v0.1 的证据依据与研究定位

整理日期：2026-11-27。本文件将已上传结果、对结果的解释和新模块假设分开。链接固定到对应提交；新模块的任何机制均尚未得到本项目实验验证。

## 1. 从已有结果得出的工作判断

### 1.1 B 保留为主贡献，自适应 GMC 保留为较小增量

第三轮统一 30 段评测结果如下。HOTA／IDF1 为报告中的百分数；表中没有 TCO 结果。

| 方法 | HOTA | IDF1 | FP | FN | IDSW |
|---|---:|---:|---:|---:|---:|
| 原生 BoT-SORT | 37.8608476 | 42.4507296 | 13,397 | 119,645 | 422 |
| `B_early_moderate` | 38.8449563 | 44.0595919 | 15,943 | 117,264 | 374 |
| 完整 BG：`B_early_moderate_dual_low` | 38.8654187 | 44.0621287 | 16,048 | 117,225 | 358 |
| `evidence_both` | 38.8596419 | 44.0827864 | 15,945 | 117,234 | 352 |

来源：[第三轮 B＋GMC 归档](https://github.com/fushiqinghua/BOT-SORT-Plus/blob/0a4b2273ebeac28134f937ff6c48749d4cde0901/results/vttiny/iterations/v0.4-motion-blur/round3-b-gmc/README.md)。

B 补回 2,657 个原生漏检目标帧，G 补回 72 个，交集 19 个；联合输出保留二者修复并集的 2,688/2,710，约 99.19%。因此，“两个大模块互相抵消”不是当前证据支持的主要解释。G 自身可新增的覆盖较小，同时存在引入额外 FP 的局部代价。以上是特定口径下的成对覆盖归因，不能把每项直接相加当作 HOTA 的因果分解。[同一归档](https://github.com/fushiqinghua/BOT-SORT-Plus/blob/0a4b2273ebeac28134f937ff6c48749d4cde0901/results/vttiny/iterations/v0.4-motion-blur/round3-b-gmc/README.md)

本轮用 `B_early_moderate` 与 `evidence_both` 作为两个工程参考，分别量化 TCO 增量。后者还包含长间隔证据门控，**不能将两参考差值叫作纯 GMC 效果**。若需要 B×G×TCO 因子归因，使用完整 BG 的独立固定配置追加对照。

### 1.2 运动不确定性改善，尚未转化为足够的闭环收益

③双向投影共检查 138,099 个候选代价，仅 16 个变化超过 0.01，没有改变实际赢家，30 段输出相同。⑤过程噪声使 95% 覆盖从约 67.28% 提高到 85.50%，同时区域面积由约 525.9 增至 2408.1，HOTA 约下降 0.045。④与 `evidence_both` 组合仍出现 FP 增长抵消收益。这些结果说明当前具体实现的作用量和选择性有限，不证明所有运动学习方法都无效。[③④⑤归档](https://github.com/fushiqinghua/BOT-SORT-Plus/blob/235fe4097066b5023cc84ca8cd6edd5757fd658c/results/vttiny/iterations/motion-345/README.md)

神经 Q/R 的全局 HOTA 也未稳定改善。R 网络使用分数、尺寸、重叠与 RGB/IR 对比度、标准差、梯度等 12 维代理量，并非学习空间图像编码器；图像代理加入后的 AUC 增量约 0.0091，区间包含零。仅低分 R 的 IDSW 改善高度集中在一个序列，且关联质量未同步改善。Q 优化还有达到最大步数但未收敛的限制，不能据此宣称学习式运动建模被彻底否定。[神经 Q/R 归档](https://github.com/fushiqinghua/BOT-SORT-Plus/blob/235fe4097066b5023cc84ca8cd6edd5757fd658c/results/vttiny/iterations/motion-345/neural-qr/README.md)

工作决策：本轮暂停继续叠加 Q/R 变体，先检验当前真实观测能否提供新增信息。此为研究资源分配，不是对整个方向的否定。

### 1.3 观测问题值得研究，但首版可达数量仍未知

D3 的特定恢复窗口中，210 次首次中断有 188 次当帧无可用检测：95 次定位不过 IoU 0.5，89 次分数不可用，4 次没有检测行。这既指向观测支持，也提醒本轮不能只看关联距离。它不是全部 117,264 个 FN 的分布。[D1–D4 诊断](https://github.com/fushiqinghua/BOT-SORT-Plus/blob/0a4b2273ebeac28134f937ff6c48749d4cde0901/results/vttiny/iterations/v0.4-motion-blur/README.md)

E3 中 6,563 个 FP 与同目标定位不足有关；持续缺 GT 支持的 3,158 个 FP 来自 41 条轨迹，但这些轨迹不能全部视为背景。22 例人工复核尚需收尾。[E1–E3 归档](https://github.com/fushiqinghua/BOT-SORT-Plus/blob/0a4b2273ebeac28134f937ff6c48749d4cde0901/results/vttiny/iterations/v0.4-motion-blur/round2-e1-e2-e3/README.md)

**研究假设**：当前 RGB 空间特征与真实短期历史可能比现有代理量更好地区分“值得细化的真实观测”。但 v0.1 仅处理原生低分阶段已有的合格提名，无法直接解决阈值外和无框事件。T1 需要先证明这一受限范围确有足够空间。

### 1.4 不能简单复用旧外观或平移策略

C2 的代价改变很少变成实际关联变化，旧模板与硬写入门控也会影响正确历史更新。原始红外难例覆盖少，模态局部错位对微小目标影响明显；旧手工平移只修正固定外推框，不等于真实低分测量的闭环更新。因此，TCO 必须在相同候选、相同位移预算上与当前像素、几何信息和手工方法比较，不能仅靠更改作用对象声称有独立收益。

来源：[C2 归因](https://github.com/fushiqinghua/BOT-SORT-Plus/blob/235fe4097066b5023cc84ca8cd6edd5757fd658c/results/vttiny/iterations/v0.3-c2-identity/02-c2-attribution/README.md)、[原始红外](https://github.com/fushiqinghua/BOT-SORT-Plus/blob/235fe4097066b5023cc84ca8cd6edd5757fd658c/results/vttiny/iterations/v0.3-c2-identity/04-raw-infrared/README.md)、[位置修正](https://github.com/fushiqinghua/BOT-SORT-Plus/blob/235fe4097066b5023cc84ca8cd6edd5757fd658c/results/vttiny/iterations/v0.3-c2-identity/05-position-refinement/README.md)。

## 2. 文献借鉴与本项目的差别

| 来源 | 借鉴点 | 本项目必须另行证明 |
|---|---|---|
| HGT-Track／HGTMT，IEEE TMM 2024 | 目标条件重检测、时序与跨模态观测；同一数据集背景 | 固定现有检测时的真实收益；不用完整重检测系统替代首版边界 |
| MCTrack，TCSVT 2025 在线发表／2026 卷期 | 时序空间与跨模态交互 | 小型局部 RGB 头的可用性、时延和因果性 |
| FDTA，CVPR 2026 | 身份特征中的时序与质量信息 | 当前数据的检测／历史特征是否确有新信息，不能套用其 DETR 系统收益 |
| ProgTrack，CVPR 2026 | 小目标关联中的上下文表征 | 目标上下文是否帮助本数据，不能假定局部背景始终稳定 |
| VLA-ReID，2026-07 arXiv 预印本 | 用历史查询当前候选，为竞争式身份学习提供参照 | 只在 T1 显示身份排序空间时另立设计，不能直接当作已落地方案 |

一手来源：

- HGTMT：[论文](https://arxiv.org/abs/2412.10861) · [作者仓库](https://github.com/xuqingyu26/HGTMT)。
- MCTrack：[论文 DOI](https://doi.org/10.1109/TCSVT.2025.3612499) · [作者仓库](https://github.com/ydhcg-BoBo/MCTrack)。
- FDTA：[会议论文](https://openaccess.thecvf.com/content/CVPR2026/html/Shao_From_Detection_to_Association_Learning_Discriminative_Object_Embeddings_for_Multi-Object_CVPR_2026_paper.html) · [预印本](https://arxiv.org/abs/2512.02392) · [作者仓库](https://github.com/Spongebobbbbbbbb/FDTA)。
- ProgTrack：[会议论文](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_ProgTrack_A_Multi-Object_Tracking_Algorithm_with_Progressive_Matching_Strategy_CVPR_2026_paper.html)。
- VLA-ReID：[预印本](https://arxiv.org/abs/2607.17157) · [作者仓库](https://github.com/holmescao/VLA-ReID)。

相关问题已有研究，并不排除硕士论文中的具体机制改进与场景适配。可主张的内容应来自本项目明确的结构、约束、学习目标及同口径实验。论文采用的标题与贡献数应在证据形成后确定，不预先承诺“第三个有效创新点”。

## 3. 暂定论文叙事与停止条件

若结果成立，可以形成三个相互补充的层面：B 在轨迹中断后恢复身份；自适应 GMC 改善相机运动影响；TCO 在现有身份与当前观测之间增强测量质量。模块位置不同只是潜在互补，实际协同仍需两个参考上的增量与联合消融证明。

若 T1 可达偏框机会很少，停止首版观测细化；若只有当前像素有效，改称局部观测细化；若历史仅改善离线而损伤闭环，不宣称跟踪提升；若正确候选存在但身份竞争主导，转向单独的候选竞争身份表征方案。具体阶段门槛见 [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)。
