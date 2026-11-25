# v0.4 下一轮实验执行计划：Lost 关联机会、坐标隔离、原生误报

版本依据：`b84d0b43267bb9f36f5ded485dde43b8f2f3ab71`。接续[首轮 D1–D4 与 D3 后续归档](../../../results/vttiny/iterations/v0.4-motion-blur/README.md)。当前任务是确定下一项可独立验证的算法贡献，不是继续搜索低分续接门限。

## 0. 冻结基线和任务范围

- 正式参考：`B_early_moderate`，HOTA 38.8450、IDF1 44.0596、MOTA 25.6942、FP 15943、FN 117264、IDSW 374。
- 30 段 / 12590 帧；冻结 YOLO11-M RGB-T 检测、RGB ReID、原生 KF、SOF、B 的两观测提前恢复及其他原生参数；关闭 A、旧 C、C2、motion2、IR/位置修正。
- 仅 E1 优先进入进一步研究；E2 为 D3 的一次收尾；E3 与 E1 并行审计。D4 仅补可比人工标签，不训练去模糊或 ReID 网络。不默认关闭 GMC，不增加低分续接门限组合。
- 以下所有 GT、身份标签、错误分组仅在独立离线侧使用。预测策略不读取 GT 或未来帧。已参与设计的 30 段仍为开发评估。
- 本次提交冻结任务安排并提供 E1-A 的独立启动脚本；E1-B/C、E2、E3 为本地实现和执行任务，不能将下面的拟议输出或验收条件误认为已经完成的实验。

## 1. 执行顺序和输入准备（P0）

1. 拉取同一分支：
   ```bash
   git switch diagnostics/v0.4-motion-blur
   git pull --ff-only origin diagnostics/v0.4-motion-blur
   ```
2. 保留原实验目录，不覆盖。准备首轮完整实验根目录（包含 `verification.json`、`analysis/summary.json`、`analysis/track_error.csv`、`analysis/detection_audit.csv`、`analysis/recovery_checks.csv` 和 `observed/`、`reference/`）。如果此前只下载了公开 ZIP，可能缺少大体积 `observed/diagnostics/B_early_moderate/*.jsonl.gz`。E1-A 可用表格，E1-B/C 和 E2 必须使用原始完整状态流，不允许重造状态。
3. 使用既有配置 `configs/diagnostics/motion_blur_v0.4.local.json`，将 `output_root` 指向原首轮完整实验目录。先运行既有的完整性检查：
   ```bash
   python -X utf8 tools/run_botsort_diagnostics.py verify --settings configs/diagnostics/motion_blur_v0.4.local.json
   ```
   若完整状态已丢失，在新的根目录按[首轮运行说明](README.md)重放 reference/observed；确认 30 份 MOT 与[历史 B 哈希](../../../results/vttiny/iterations/v0.4-motion-blur/historical_B_comparison.json)一致，并复用本分支已修复的 Removed=4 离线解析规则。不因缺文件跳过等价性验证。
4. 按[任务清单](../../../configs/diagnostics/v0.4_followup_tasks.json)记录每项 PASS/FAIL/BLOCKED，不把未执行项目标为通过。使用新的输出根，例如 `outputs/v0.4_followup_round2`。原始检测、特征、原图、GT、TrackEval 版本、实际源码及生成输出均留哈希。

## 2. E1：Lost 实际关联机会审计与预测对照（主任务）

### E1-A. 冻结队列（已有启动脚本）

```bash
python -X utf8 tools/prepare_v04_lost_cohort.py --run-root outputs/diagnostics_v0.4_full30_round1 --output outputs/v0.4_followup_round2/E1_cohort
```

替换 run-root 为实际首轮完整目录。脚本验证已归档的分析表指纹、参考/观察协议哈希、30 段/12590 帧与历史 B 输出摘要；输出逐轨迹帧、分组计数、独立 GT 帧数、GT ID 数、旧轨失联片段数及序列数。它只做离线分类，不改代码中的任何匹配逻辑。

定义：
- 主队列 `state_before=2`（BoxMOT Lost），`2 <= gap_frames <= 30`；Removed=4 与 gap>=31 单列；只使用过去实测锚定且当前有 GT 的样本做原因分析。
- 用原分析中的唯一检测标签联结 `gt_paired_det_index` 与 `(sequence,frame,det_index)`；记录其 `detection_tier` 和消费 ID/阶段。区分 `same_id_correct`、`different_id`、`wrong_identity`、`ambiguous_output`、缺失但可用检测、无可用检测；GT 有候选但配对不唯一另计，不能算作没有检测。
- 上轮复算的参照为：Lost、2–30、`missing_with_usable_detection` 共 1407 个轨迹帧；未消费低分 1098、未消费高分 129（103 个独立 GT 帧）、被消费高分 172。剩余类型保留，若不一致先检查原分析口径、唯一配对与边界，不能强行改数据对齐预期。
- GT 帧键为 (sequence,frame,anchor_gt_id)，旧 ID 失联片段键为 (sequence,track_id,last_seen)，GT 轨迹键为 (sequence,anchor_gt_id)。同一个 GT 帧可进入不同旧 ID 的竞争队列，所以各分类去重后不能简单相加。
- 主计数仅用于确定可作用范围，不代表这些 FN 都可以恢复；已有其他 ID 正确输出的帧属于身份竞争，不算额外覆盖。

### E1-B. 还原原生和 B 关联失败的实际阶段（本地新增）

复用 `botsort_plus/diagnostics/analysis.py`、`common.iter_trace`、`adapter_v2.py`、`candidates.py`、`recovery.py`、`geometry.py` 和原有事件。对 E1-A 中的唯一正确检测逐事件联结：

| 阶段 | 记录内容 |
|---|---|
| native high | 原生候选池、IoU、ReID 原代价、阈值、竞争者和分配结果；不能只把 proximity mask 当作硬几何拒绝 |
| native low | Lost 本身无原生低分匹配资格；不将其归类为普通低分几何拒绝 |
| candidate | 当前残余检测是否被占用、重复抑制、seed deferred、链接失败、候选是否过期；使用 candidate_id 和 det_index 对齐 |
| B history | 候选最后 5 帧中有多少实际观测；同帧预测是否存在；两观测提前条件、三观测路径和保留期 |
| B pair | 原几何门、当前/均值外观矛盾、总代价、双向竞争、带拒绝的一对一分配；记录原理由和最终分配 |
| result | 当前 GT 是否已被其他 ID 覆盖，正确候选是否真的获得输出；未知/多义原因单列 |

如果历史事件没有完整矩阵，增加**只读 shadow** 记录（保存原 native 矩阵、候选观测、历史预测及计算前的 MemoryBank、B raw_costs/eligible/margins）。不能从最终输出推断矩阵，也不能把后来的外观模板用于先前时刻。禁止额外执行原生 RANSAC 或改变随机状态。先单序列验证 MOT 和原事件字节完全相同，再全量，且复现当前 B 的 accepted pairs/拒绝原因。真实运行时 GT 不参与 shadow。

建议先用 DJI_0061_2（竞争）、DJI_0173_2（GMC 局部异常）、DJI_0327_3（困难密集）、DJI_0030_2（船舶）检查，再对所有 30 段统计。不得只报告选中的成功片段。

交付：`E1_failure_stages.csv`、`E1_episode_summary.csv`、`E1_shadow_parity.json`、`E1_failure_report.md`。候选与事件无唯一对应时用 unknown，不编造原因。

### E1-C. 固定状态预测反事实（本地新增）

先固定原生关联输入池、检测、历史外观、宽高、协方差和所有关联门限。每个版本只改 Lost 预测中心；不能额外给低分框原生关联资格。

- P0：原生 KF/GMC。
- P1：末次实测后的中心逐帧经实际 GMC 仿射运输，不加目标速度。坐标定义用最近一次已接受检测框中心；不把最终输出或 GT 当作测量。
- P2：在 P1 之上加入有界速度衰减：初速度取末次真实更新后的 KF 中心速度（来自完整 after_update）；每帧经仿射线性部分运输，按固定 `rho=0.8` 衰减；累计目标位移的范数上限为 48 原始像素。用向量限幅而不是逐坐标裁剪。该值仅是首轮开发假设，不搜索 rho/限幅。若历史仿射、速度或先验缺失，保留 P0 并记录不可用原因。

先验证 P0 固定状态重算等价。P1/P2 逐帧真实 GMC 运输，历史 B 观测须对每一观测帧使用相应历史预测，不允许从当前帧反推。B 所有竞争者（包括尚未成熟候选）都必须进入相同的歧义检查。若仅能得到几何边而不能重算完整分配，将结果明确命名为 geometry-only，不得声称“新增正确恢复”。高分第一阶段要保留原池和同一分配器、分数融合、外观门限及原阈值；B 仅处理原参考残余候选，属于条件固定状态对照，不能解释为完整闭环效果。

各版本都报告：
- 同 GT/同帧中心误差、几何通过率、代价间隔、正确/错误边新增与消失、原先正确分配损失。
- 固定状态分配变化与 GT 支持（correct / wrong / unknown），分开 native high 和 B。对已由另一 ID 覆盖的情况报告身份延续，不计作补 FN。
- 按独立 GT 帧、GT、旧 ID 失联片段、原视频分组及序列报告；剔除贡献最大序列的敏感性分析。
- GMC 前后同状态误差仅为归因项，不单独做“高残差关闭 GMC”变体。

交付：`E1_prediction_pairs.csv`、`E1_assignment_deltas.csv`、`E1_group_summary.csv`、`E1_decision.md`。如果只是误差变小而无实际分配改善、错误边抵消收益或集中在重复 Lost，不开发新 KF。如果多段真正可恢复事件改善且原有正确匹配保留，才另行实现均值改进、协方差改进及二者组合的闭环消融。当前不要把 P1/P2 直接加进正式跟踪器。

## 3. E2：固定 191 个提名，仅比较输出坐标（D3 收尾）

来源：[只补帧报告](../../../results/vttiny/iterations/v0.4-motion-blur/d3-follow-up/03-output-only/report.md)、[191 行提名](../../../results/vttiny/iterations/v0.4-motion-blur/d3-follow-up/03-output-only/parity/full/supplemental_rows.csv)、[误报溯源](../../../results/vttiny/iterations/v0.4-motion-blur/d3-follow-up/04-background-filter/report.md)。

冻结所有 191 个 `(sequence,frame,track_id,det_index)`；不重新运行低分规则，不重新决定接纳。原方案 72 TP / 119 FP，49 个固定 D3 首次中断原 ID 补回。44 个 FP 属于同一目标定位不过线；它们是重点，但必须对全部 191 行复评。

- E2-raw：原 `B_postlow_output_v1`，原始检测框。
- E2-kf：取原始 `after_gmc` 对应旧轨的 8D mean 和 8x8 covariance，分别复制；用固定 BoxMOT 16.0.11 `KalmanFilterXYWH.update(mean,cov,measurement)`（与原生更新接口一致，不新加置信度参数）只执行当前提名检测的一次测量更新；measurement 为中心 XYWH。取更新后前四维转换为输出 TLWH。不能使用已更新状态、不能调用真实 STrack.update/re_activate。宽高/非有限异常不静默修补，报告并停止。
- 原有 MOT 每一行字节保持一致；补框的 ID、帧、分数、数量、顺序不变，只替换坐标字段。核对原始两组输入及输出哈希（服务器历史脚本 `evaluate_output_only.py` / `rescore_filter.py` 有本地 `run_round.py` 依赖，仓库未收录该辅助文件；不要声称它们可以直接在任意 checkout 运行）。优先使用新的独立离线脚本；保留原实现不改。
- 用固定 TrackEval 提交 `12c8791b303e0a0b50f753af204249e622d0281a` 重新完整评估 B、raw、kf；先精确复现两组历史结果。生成逐行 CLEAR 匹配对照与 287 个原固定窗口对照，检查是否挤占原生 TP 或引入换号。原生输出不能进入修改集合，但其 CLEAR 分配仍可能变化，需要报告。
- Smoke：DJI_0303_2 第58帧轨迹2/检测5（同一检测 raw IoU 0.47351，历史状态版 0.52930；它不是本次复制状态的保证值），另检查 DJI_0327_3、DJI_0339_2、真实静止小目标的反例。完成后全量一次，不调混合系数。

交付：`E2_row_integrity.json`、`E2_box_pairs.csv`、`E2_clear_deltas.csv`、`E2_fixed_windows.csv`、`E2_metrics.csv`、`E2_report.md`。若仍有 191 行、原生 CLEAR 分配不变、额外 IDSW=2，则至少 97 新增 TP 才能使 MOTA 不低于 B（当前 72，需要净增25）。这是条件推导，不是通用阈值。无论结果是否改善都只做这一次坐标对照；未取得稳定净收益即结束 D3 补帧规则链。

## 4. E3：正式 B 的原生持续误报审计（并行）

对象为正式 B 原始 30 段轨迹，不是 191 行补帧。原 FP 15943。已有 DJI_0339_2 的 ID2/ID6（125+384 原生输出、全段最大 GT IoU=0）仅作为需复核线索。

E3-A：使用与官方指标一致的 CLEAR 匹配及连续性处理，逐行复算 FP/FN/IDSW，要求与历史 B 一致；同时计算对所有当帧有效 GT 的最大 IoU，以及原检测/输出与最近已确认 GT 的 IoU。独立标签的 D1 配对不能代替 CLEAR。按轨迹、原视频、目标帧分类：历史始终远离 GT（最大 IoU<0.10 且至少3次实测）、同一目标定位不过线、错误目标/重复/竞争、目标离开后的疑似残留、身份或标注未知。离线全生命周期分类只做归因，不能用于在线策略。空 GT、漏标或标注不完整不得自动宣称背景。报告 0.05/0.10/0.20 作为诊断敏感性，不用于调过滤器。

E3-B：对贡献较高和跨序列的持续无 GT 支持轨迹做人工复核；配对真实小且静止目标及近邻定位误差反例。保存原尺寸图块、上下文和相邻帧（无损 PNG）；标为 confirmed_background / possible_unlabelled_object / true_object / uncertain。记录人工判断依据；分层样本不能直接推算全量真假率。

E3-C：仅用截至当前帧可用的证据（检测分数与历史、尺度、GMC 补偿后运动、RGB 特征质量/变化、可用的 RGB/IR 局部目标性等）检查区分能力。原始 IR 未必有可靠配准，不默认加入；历史 ReID/分数与静止度可能同时在真目标和背景上成立。在线特征不能包含“全生命周期未匹配”标签。按原视频分组报告留出表现、正反例和有用特征；同源视频片段不得跨折。若可分样本不足，先补人工复核，不训练复杂模型。此轮不修改轨迹存亡与输出。

交付：`E3_native_fp_rows.csv`、`E3_trajectory_profiles.csv`、`E3_sequence_summary.csv`、`E3_manual_review.csv`、`E3_signal_analysis.csv`、`E3_report.md`。若影响只集中个别视频或现有信息不能区分真实静止小目标和背景，停止该方向。

## 5. 统一验收与回传

| 项目 | 完成条件 |
|---|---|
| P0 | 30 段输入、MOT/事件与历史 B 一致，GT 哈希及固定 TrackEval 核对；完整状态可用性明确 |
| E1-A | 队列可复算；独立 GT 帧、GT 轨迹及失联片段去重；未知不混入确定计数 |
| E1-B/C | P0 重算一致；报告实际可作用的失败阶段与正确/错误关联机会，固定状态和闭环结果明确区分 |
| E2 | 同191行、原生行未改、两组历史指标复现，三组全量指标和固定窗口完整 |
| E3 | FP/FN/IDSW 与历史一致；持续无 GT 支持与真实背景分开；人工复核和可分性分析 |
| 决策 | 每项给出 CONTINUE / STOP / INCONCLUSIVE 及证据；不因微小指标变化自动继续 |

回传目录应包含本计划版本、实际命令、运行环境、源文件及输入/输出指纹、配置、各 CSV/JSON、最终 Markdown 报告、必要样例和未完成原因。保留每个失败实验的记录。三组独立，禁止同时改变 B/KF/检测/ReID；全量指标使用 TrackEval COMBINED_SEQ，所有百分点注明口径。最终根据 E1 的真实分配作用与 E3 的可分性择一进入模块开发，而不是强行增加创新数量。
