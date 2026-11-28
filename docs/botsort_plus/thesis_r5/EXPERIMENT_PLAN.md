# Thesis R5：三模块完整消融、交互分析与论文图表准备

状态：**DESIGN_ONLY / 待实现并执行**。承接 `research/vttiny-weak-evidence` 的 `f90501853b07bcca5d10e99cd17f6dc9db42bfb4`。本文件是给本地 Codex 的新批次实验任务书，不代表新 runner 已实现，也不代表新实验已完成。从冻结的 R4.2 结果提交建立独立分支 `research/vttiny-thesis-finalization`；R5–R8 的新实现、结果及论文图表仅归档在本分支，不修改原 W 研究分支的历史归档和已锁定的 R4.1/R4.2 实现。

## 1. 研究目标与冻结边界

R5 不再设计新评分模块或搜索 W 参数。保留原 B_early_moderate、GMC 和 W0。主问题：
- B、G、W 是否有独立及条件贡献，是否存在交互；
- 长间隔重连保护到底增加了什么，避免将 evidence_both 当作纯相机补偿；
- W0 的弱观测使用相对于直接降低阈值有什么真实的召回/误报取舍；
- 为论文主表、消融表及真实案例准备可重复生成的证据。

符号约定：
- N：原始 BoT-SORT（SOF、固定 RGB ReID），B/G/W 全关闭；不是无 ReID 或无 GMC。
- B：B_early_moderate，仅开启冻结的时序失联恢复。
- G：G_core，完整自适应 GMC + dual_low（低分双几何确认）；不含 evidence_both 长间隔保护。
- W：R4.1/R4.2 原 W0，真实 S0 弱检测、原锚点与所有权/几何约束、一对一、真实状态反馈，不新建 ID、不写弱外观、不让弱观测刷新常规锚点。其 S0 判定和边界、top-5、阈值严格继承冻结源码。
- R：历史 evidence_both 中对长间隔重连的额外保护。仅在既有 B+G 上按原定义开关，不把 R 当作三因子中的 G 本身。历史 W 分支里的 BG = B+G+R。
- Final = B+G+R+W。W0 是冻结候选，不把它宣布为全指标最优或对降阈值全面占优。

保持同一 VT-Tiny-MOT 官方 30 段、12,590 输入帧、179,772 GT 实例；RGB 坐标及官方 GT 不变，全部类别 pooled；冻结 YOLO11-M RGB-T 检测缓存、RGB ReID 编码器及特征、SOF/自适应相机变换缓存、BoxMOT 16.0.11、TrackEval 版本、匹配与忽略处理。旧协议及其有效配置、输入/解码/特征/GT/评测源码哈希须核对。官方 test 已参与开发，全部 R5 结果仍为开发评测；不是独立盲测。预测进程禁止接收 GT 或评测参考路径。禁止重新训练、改检测器 NMS/置信度导出、调 ReID、启用 A/C/TCO/Q2/W1–W3 等已停止分支。若无法核验旧产物，重跑或标明不可直接对比，不手填指标代替预测文件。

## 2. R5-A：三因子 2^3 完整矩阵

| ID | B | G | W | 实际定义 | 已有 HOTA / IDF1 | 工作 |
|---|---:|---:|---:|---|---|---|
| N | 0 | 0 | 0 | native | 37.8608 / 42.4507 | 旧参考，新的 W 接入需 off 等价性 |
| B | 1 | 0 | 0 | B_early_moderate / B_off | 38.8450 / 44.0596 | 复用 |
| G | 0 | 1 | 0 | dual_low | 37.9603 / 42.6228 | 旧参考，新的 W 接入需 off 等价性 |
| W | 0 | 0 | 1 | N + W0 | 待测 | 新全量 |
| BG_core | 1 | 1 | 0 | B_early_moderate_dual_low | 38.8654 / 44.0621 | 旧参考，核验新 runner 的 off |
| BW | 1 | 0 | 1 | B_W0 | 39.0174 / 44.2224 | 复用 R4.1 |
| GW | 0 | 1 | 1 | G + W0 | 待测 | 新全量 |
| BG_core_W | 1 | 1 | 1 | B+G_core+W0, R off | 待测 | 新全量 |

另设保护与最终组合表，必须保留 BG_core、BG_core_W 两个无 R 参照：
| ID | 定义 | 工作 |
|---|---|---|
| BG_R | evidence_both = B+G+R（历史 BG_off） | 复用 R4.1/R4.2 |
| Final | evidence_both + W0（历史 BG_W0） | 复用 R4.1/R4.2 |
| BG_R_low05 | BG_R，W off，原生 track_low_thresh=0.05 | 复用 R4.2 |
| BG_R_low075 | BG_R，W off，原生 track_low_thresh=0.075 | 新全量，预先固定中间点 |
| BG_R_low10 | BG_R，W off，原生 track_low_thresh=0.10 | 与 BG_R 同一结果，非新增实验 |
| BG_R_W_relaxed | 历史 BG_W0_relaxed_anchor | 复用 R4.2 |
| BG_R_W_emit | 历史 BG_W0_emit_only | 复用 R4.2 |

本轮四个新增科学配置（3 个 W 组合 + 1 个固定阈值对照）。新的 N/G/BG_core W 接入必须以对应 off 从首帧闭环验证；原则上另安排 3 个完整 off 等价性回放，若已经有完全相同的接入代码、模式及完整等价性凭据可复用才省略。其他旧参考也只有完整资产及协议核验通过才复用。故不保证实际只运行四次；默认新增全量任务为 4 个实验 + 3 个技术回放，缺失旧资产时另列补跑。技术回放不计独立样本或科学重复。

## 3. 实现任务与技术验证（本地 Codex）

现有入口 tools/run_wem_r41.py、tools/run_wem_r42.py 的角色仅 B、BG，不能原样启动独立 N/G/W。特别是：
- experiments/vttiny_gmc/round3_joint/common.py 的 METHODS/frozen_config 基于冻结 B；worker.py 的 prepare 明确要求 recovery.enabled 且 early_evidence=moderate。
- tools/research/wem_r41/batch.py 通过固定源码 blob 和插入钩子连接 adapter_v2/worker；assets.py 只接受 B/BG；online.py 的 W0 依赖原生事件、候选所有权、正式预测和锚点。
- tools/research/wem_r42/batch.py 的 lower_low 注入固定 0.05 并记录实际生效值。不得伪造新的 CLI 或直接覆盖该历史协议。

建议另建 tools/research/thesis_r5/、tools/run_thesis_r5.py、configs/research/thesis_r5.example.json 与 thesis_r5_protocol.json。上述为待实现的建议路径。实现新 runner 时保留所有历史 pin/hash；复用既有 W0 逻辑，独立创建角色/配置，不修改旧 worker 的全局约束或伪造其协议。N/G 的 B off 模式在 v2 接入路径保持原生关联和生命周期，检查无 B 候选保留、无恢复事件、无 A/C；G 的 adaptive_gmc、dual_low 与单 G 历史定义相同，R 不激活。若 v2 disabled 配置不复现 native/dual_low，则先定位差异（阈值端点、检测行排序、原生二阶段/新生、KF/GMC、去重与输出、ReID），必要时针对真正原生路径增加等价钩子；禁止把不等价 v2-off 冒充历史 native。若确实无法达到等价，保留结果与明确的单独底座，暂不进入历史八组主消融表。

建议 runner 子命令 init/check/smoke/full/evaluate/summarize/figures（仅在实现且验证后写实际命令到 LOCAL_RUNBOOK.md）。继承 R4.1/R4.2 的本地设置和 round3 warp_sources；独立预测 bundle 与评测配置。全程记录实际配置、源码 hash、依赖版本、设备、输入/输出 SHA、完成状态、失败和重试；运行目录带日期，不覆盖旧结果。每段独立重置，从首帧运行，不接用其他策略的中间状态。

P0：检查所有 30 序列检测、特征、图像解码、native/adaptive warp、GT、TrackEval 一致性及原始运行目录完整性。验证阈值 .075 原生分带边界，确认相机缓存不因该参数变化被重估。确认参数变化确实到达 tracker 运行时。
P1：固定技术 smoke：DJI_0030_2、DJI_0169_3、DJI_0109_1、DJI_0173_2、DJI_0221_2、DJI_0028_3、DJI_0339_2（7 段 3180 输入帧）。检查 N/G/BG_core off 与同角色历史 MOT 逐字节或严格规范化行级相等；同接入 shadow/force-reject 与 off 的正式状态/原生事件相等。历史没有事件/状态的参考不得声称对其实现了字节级事件比对。检查 W0 的候选来源、历史、所有权、锚点不变量、状态/ID/外观写入保护及 R off；确保 B off 不存在 B 操作，G off 仍保留 native SOF。
P2：全部新增配置及必要 off 参考回放 30 段；对完整 N/G/BG_core off 进行同样的逐段 parity；复用凭据须清楚区分历史产物与新运行。
P3：同一 TrackEval 统一重新评测全部能导入的完整预测。历史仅有派生 CSV 的结果允许在明确标注的历史对照表中使用；不能宣称完成本轮统一原始轨迹评测。无 GT 的序列也必须运行；输入帧总数与 TrackEval CLEAR 有效帧计数单列。
P4：生成归因、图表、数据源清单和结果报告。仅技术完整性失败阻断；指标负向或 W 零提交照常完成并报告。不要因为 smoke 指标低而更换序列/参数。
测试最低要求：B disabled 路径、G core policy、R disabled、W off/shadow/forced rejection、W 真实接纳/所有权冲突/弱外观不写/原 ID 不变、阈值有效值、输入哈希变化失败、断点续跑、缺帧/重复行拒绝、TrackEval 聚合与 figure 数据对账。先合成和源码接入测试，再真实 smoke；不能用单元测试替代真实结果。

## 4. 评价与归因

主指标 HOTA（沿用此前预登记），同时报告 DetA、AssA、IDF1、MOTA、FP、FN、IDSW；建议附 LocA、DetRe/DetPr、AssRe/AssPr、Frag、MT/ML 及 CLEAR TP/precision/recall。百分制差值为百分点；FP/FN/IDSW 为数量。采用 TrackEval COMBINED_SEQ；对分组子集使用 HOTA 每个 alpha 的 TP/FN/FP 加和、AssA 以该 alpha TP 加权后重算 sqrt(DetA*AssA) 并平均；IDF1 按 IDTP/IDFP/IDFN 汇总，MOTA 按 CLEAR 计数和 GT 汇总。不能平均各序列 HOTA、IDF1 或 MOTA 百分数。

R5-A：每一模块输出四个条件增量（另外两模块固定），例如 W−N、BW−B、GW−G、BG_core_W−BG_core；G 与 B 同理。计算两因子交互（如 (BW−B)−(W−N)），条件明确；三因子交互如需要作为附表。不要把简单增量直接视作统计显著性或因果证明；以独立完整回放、固定输入支持方法级归因。R 额外保护报告 BG_R−BG_core、Final−BG_core_W 两项，避免把它算入纯 G。
R5-B：固定 BG_R 的 0.10、0.075、0.05 三个原生阈值以及 W0(原生阈值仍 0.10)。完整报告所有点；不依据官方开发结果再选阈值，只有确实达到相近 FP/FN 时才做近似同代价比较，不能假设单调或曲线插值是实测。画 FN–FP 散点（双轴越低越好），标注 HOTA/IDF1，比较实际 Pareto 取舍；MOTA 不替代主指标。
R5-C：用同一 CLEAR 匹配口径与有效 GT key=(sequence,frame,gt_id) 区分补回/丢失/保持；对任一比较，FN_ref−FN_arm=新增命中−丢失命中；以实际 GT 数与官方 ignore 为准，不将 GT ID 0 当缺失。计数 W emitted、state updates、当帧匹配/未匹配、后续 1/3/5/10 帧同 GT 同旧 ID 覆盖、B 恢复与错误延续，区分事件计数与 GT 帧计数，标记视频边界 censor 和无 GT/身份不确定。对同状态反事实与独立回放分别汇报，不把当帧未匹配一概称为海浪背景，也不把同一整条 FP 链都算作单一干预的因果贡献；必要时单事件从首帧重放反事实并另标来源。
R5-D：逐序列 ΔHOTA/IDF1/FP/FN/IDSW、正/负/零分布，按已验证原视频映射分组（不能仅从文件名猜测）。剔除最大 GT 量 DJI_0327_3、设计案例子集及逐原视频留一，按正确 COMBINED_SEQ 重算；这些是开发集敏感性分析，不是独立 holdout 或显著性。若组映射不能核验，仅做序列敏感性并说明限制。

## 5. 论文表格及图表交付

全部数值图由实际 CSV/JSON 生成，保存数据源、绘图脚本、参数、commit、文件哈希和 caption。采用统一论文配色和字体、矢量 PDF/SVG（字体可嵌入，非栅格化文字）、PNG 300–600 dpi；中英文标题可切换，图例/单位/箭头方向清楚，不对零结果或负例作视觉删减。矢量流程图用可编辑源（如 SVG/Graphviz），真实案例直接使用本地图像和真实框，不用生成式图片补造证据。

| 编号 | 图表 | 内容及数据来源 | 批次 |
|---|---|---|---|
| T1 | 三因子完整消融表 | 8 个核心组合，另列 R 的条件增量与最终组合；HOTA/DetA/AssA/IDF1/MOTA/FP/FN/IDSW | R5 |
| T2 | W 内部消融与阈值对照表 | R4.2 五组、固定 0.075、R4.1 W1/W2/W3 如需附表，实际配置、差值 | R5 |
| T3 | GMC 及长间隔保护消融表 | 历史 round3 7 组（含 skip_only/no dual_low），R5 新条件结果；不混淆复合模块 | R5 |
| T4 | 主流跟踪器横向对比 | 同检测及评测下 ByteTrack/OC-SORT 等可复现方法、native、Final、降阈值强对照 | R6 |
| T5 | 运行效率表 | 固定平台、p50/p95/p99、平均/FPS、显存、后端与端到端分别报告 | R8 |
| F1 | 总体框架图 | RGB-T 检测/ReID、GMC/KF、原生高低分关联、B、W、轨迹输出；按实际执行顺序，体现 B 优先、GMC 保护范围 | R5 方法图草案 |
| F2 | G 模块结构与实际相机补偿案例 | 原生/identity/LightGlue 验证选择、dual_low、R 独立注记；同一真实帧对和残差统计，明确 LG 并非每帧采用 | R5 结构，R7 真图 |
| F3 | B 身份恢复机制与案例 | Lost、残余检测、候选历史、门控、一对一接回原 ID；实际 gap/cost/ID 对照 | R5 结构，R7 真图 |
| F4 | W 机制及状态反馈图 | 已确认轨迹、常规锚点、真实 S0、所有权/IoU、当前输出和未来状态；emit_only 对照，不把 W 画成插值或生成检测 | R5 结构，R7 真图 |
| F5 | 完整消融的条件增量图 | 每模块四个条件 ΔHOTA/ΔIDF1，附 FP/FN/IDSW，零线；R 单列 | R5 |
| F6 | FN–FP 实测取舍图 | BG_R 0.10/.075/.05 与 Final，标注 HOTA/IDF1；不虚构连续曲线 | R5 |
| F7 | 逐序列配对增益热图/点图 | 30 段，ΔHOTA、ΔIDF1、ΔFP/ΔFN/ΔIDSW，原视频分组与零变化；同一颜色尺度、注明单位 | R5 |
| F8 | 困难目标分层分析 | 固定目标像素等效边长、失联长度、运动可靠性分层；显示样本数，说明匹配/覆盖口径 | R6 |
| F9 | RGB–IR 真图跟踪效果及失败例 | 相同帧、视野、缩放、原始 ID 和置信度；GT 虚线、预测实线、FP/FN 有明确标记；ID 在每方法内独立 | R7 |
| F10 | 运行延迟图 | 后端耗时 p50/p95、端到端延迟与模块增量；不将缓存诊断运行速度当部署 FPS | R8 |

正文优先 T1–T5、F1–F6、F8/F9；F7 或更多案例可放附录。不要为了篇幅重复绘制相同信息的柱图。R5 先交付 T1–T3/F5–F7 的数值源和脚本；F1–F4 结构图按冻结代码审核后绘制。R7 在本机生成真实图像图卡，若素材不能发布仅回传获准的裁切/派生数据。

## 6. 本轮需要保留的案例数据（R7 图卡素材）

先在已有原始运行与新 R5 产物中导出候选帧号/行号/GT ID/track ID、实际 xyxy、raw score、状态、阶段、有效 GT 与忽略区、GMC 选择/备用矩阵、匹配门控理由。保存原尺寸 RGB/IR 对应关系与配准方向；不同模态未严格对齐时标明参考坐标，不能直接把 RGB 框叠到未配准 IR。
固定 W 正例：DJI_0327_3，GT16 / ID55，228–237 帧（228 当帧未匹配，230–236 完整 W0 同 ID 覆盖，237 高分接回）；固定 W 锚点负例：DJI_0339_2，ID2，95–98 帧及之后真实错误延续；降阈值负例：DJI_0339_2，从 R4.2 已有逐帧案例恢复具体帧；W 反馈负例：DJI_0181_3，先找 W0/emit_only 的首个造成新增未匹配输出链的实际分叉，报告匹配/定位/背景不确定性，不预填帧号。
G 先用 DJI_0109_1 / DJI_0173_2 作为正向候选，DJI_0030_2 作为保护/错误案例；B 从已存 changed_B_events 和完整 Lost 恢复事件中按预登记规则选取成功/失败（按 gap、竞争、原 ID、后续覆盖筛选，优先事件足够清楚且跨母视频）。同样保留普通/无收益和失败例；展示用例不等于发生频率。图卡素材不作为在线 GT 门控。人工复核标签未完成的例子只陈述实际可核验的 CLEAR 与几何事实。

## 7. 回传目录与验收

建议新目录 results/vttiny/thesis_r5/<实际日期>/，含 README.md、run_provenance.json、manifest.json、protocol.json、smoke/verification.json、trackeval/summary.json 与原始明细、report/{REPORT.md,comparison.csv,factorial.csv,conditional_effects.csv,protection_deltas.csv,threshold_tradeoff.csv,paired_per_sequence.csv,grouped_sensitivity.csv}、attribution/{coverage.csv,events.csv,case_manifest.csv}、figures/{data,svg,pdf,png,figure_manifest.json}、repro/。原始本机图像/GT/缓存/权重/MOT/完整坐标事件留本地；公开版可去坐标/私有路径，保留原件和发布 SHA256、删减字段说明，图表仅发布授权素材。保存实际使用的运行命令及新的 LOCAL_RUNBOOK，不以未实现的命令冒充运行。

验收：所有计划组完成同协议全量或明确列出技术阻断；off 等价验证通过才允许与旧 native/G/BG_core 合表；所有主表指标可从源数据复算；四个 W 条件增量、R 两个条件比较、三阈值/最终 W0 取舍均有据；图表数据和汇总逐项对账；报告标注负结果、无变化、重用及未验证项。HOTA 增加不是软件 PASS 条件，下降也不删除。不要把当前开发集的结果包装为独立盲测。

## 8. R6–R8 后续范围（不在 R5 新增算法）

R6：选择少量公开、可复现的同检测输入主流 MOT 方法，冻结版本及参数来源；若需调参仅训练/验证侧统一预算，测试侧固定。所有方法统一类别、坐标、GT、输出格式，明确是否使用 ReID/GMC，不能将不同检测器的论文数值直接混为公平对比。分层分析按 GT 像素等效边长 <8 / 8–16 / 16–32 / >=32、已有 gap 桶、相机可靠性/运动强度；报告 GT 支持数、CLEAR 覆盖与 ID 事件。目标级分层若不能以正确 ignore/匹配定义重跑 HOTA，则仅报告事件级覆盖/身份诊断，不能把筛选后百分比称为标准 HOTA。
R7：生成真实图卡、方法正反案例，优先独立母视频/飞行架次或自有真实落水人员序列；如果有独立标注，先锁定方法/参数再运行并单列结果。没有则如实报告，不冒充盲测。
R8：关闭诊断、状态预演及重复日志，验证输出等价；同设备、线程、输入、分辨率、warm-up、同步条件测量 native/B/BG_R/Final/BG_R_low05；后端缓存 replay 与 live GMC、检测/ReID/端到端分开，至少 3 次同条件重复，报告 p50/p95/p99、均值、总帧/实测总时间、显存/内存；NX 另列实际功耗模式/精度/引擎/输入，不假设已部署成功。

## 9. 已有证据入口

- [R4.1 全量十组](../../../results/vttiny/wem_r4_1/full_20261128/README.md)
- [R4.2 全量五组](../../../results/vttiny/wem_r4_2/full_20261128/README.md)
- [R4.2 设计与定义](../wem_r4_2/EXPERIMENT_PLAN.md)
- [GMC 第三轮七组](../../../results/vttiny/iterations/v0.4-motion-blur/round3-b-gmc/README.md)
- [R4 早期八组](../../../results/vttiny/wem_v0.1/r4/README.md)

当前仅提交实验与图表任务书；待本地实现、测试及数据执行，不改生产默认值。
