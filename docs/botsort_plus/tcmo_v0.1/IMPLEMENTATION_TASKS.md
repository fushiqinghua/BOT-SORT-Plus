# 本地 Codex 实施与验证清单（当前未实现）

建议新目录 tools/research/tcmo_v01/，新入口 tools/run_tcmo_v01.py，保留旧 r41/r42 原件及所有 pin。以下文件与 CLI 为实施合同，不是现有可运行入口。

## P0 输入与来源
- 读取 frozen R4.1 settings/prediction/evaluation manifests 和完整 R4.2/R4.1 BG_W0 本地目录，复用 tools/research/wem_r41/assets.py:prepare 与既有 evaluator 检查。冻结 protocol、detector/reid、features、decoded images、native/adaptive warps、GT、TrackEval、BoxMOT。记录有效配置、commit、源码 blob、文件 sha256；修复机器路径通过配置，禁止修改历史凭据。新 branch 继承 f905018 的代码，不改 R5–R8 论文分支。独立 V/T 权重/图像/训练数据不在公开归档验证范围，先输出 inventory.json，注明已存在/缺失/来源待核。核对官方配对序列、帧号从1、尺寸及 mapping direction。权重缺失先训练并冻结新协议，未完成就保留 BLOCKED_ASSET 状态。
- 实现 assets.py、export_aux.py、calibrate.py，输出辅助 manifest 与逐帧缓存、独立预测/eval bundle；GT 仅在训练校准和离线评测进程出现，运行 child 只收预测资产。禁止从 GT 文件名反推对象或将 GT label 写入可在线访问缓存。保持所有原行和新来源唯一 key。

## P1 只读 sampler、R0
- 阅读 tools/research/wem_r41/online.py:ClosedLoop.finish_before_output（W0 真实候选和 anchor 更新）、tools/research/vttiny_wem_r4/online.py:transact/native_preview（私有预演）、tools/research/vttiny_wem_r3/observer/capture.py、botsort_plus/adapter_v2.py: update_v2 / _record_output、tools/research/wem_r42/online.py（output-only 对照）。
- 在独立的新钩子实现 observe-only，采样 W 后实际剩余风险状态，保留普通 accepted、reserved/consumed/W committed、selected/backup prior mean/P、anchor ready/spent/observed/mean、轨迹 raw history、最终 W 输出与所有权。不得先修改 W0 策略来获取更多候选。现有 observer 的 current_assignments 只接受原生 match/recovered/native_birth；保留该含义。
- 独立 offline labels/opportunity.py 读取 GT_00/GT_01 和状态快照，参考 label_rules.py 的严格历史 raw+CLEAR 双支持身份。实现 all_FN/eligible_FN、same/other/unknown/ignored、native IR/映射/预测尺寸 proxy/posterior 的不同 IoU 口径、跨模态交并集与来源分层。CLEAR 不匹配不是人工确认背景。不要把低 IoU 当跨身份，不用 GT 决定线上配对。保存原始 GT id=0（不是缺失）。R0 独立运行不影响真实跟踪。

## P2 新闭环接入
- 实现 geometry.py（声明映射、中心/角点变换、裁边/可见性、真实像素域、稳定分组键、冲突/所有权、GMC 备用门）；association.py（校准 q、proxy IoU、top5 group、dummy assignment、稳定 ties、拒绝原因）；measurement.py（二维观测、原 KF 真实 R block、Joseph 更新、有限/PSD 检查、私有更新）；transaction.py、online.py（W 优先、同一事务、按真实 W commit 而不是 proposal 排除、共享 budget、对其他输出与状态保持）。
- 具体接入：R4.1 batch.py:_adapter_source 在 update_v2 的 _prepare_output 处替换为 tracker.wem.finish_before_output，发生在 _update_track_states(removed) 之后、_record_output 前；重用该单一边界。现有 transact 正式赋值 tracker.active_tracks/lost_stracks/removed_stracks，禁止将 TCMO 追加在该函数完整执行之后。新组合必须复现 W 原本每项预演与列表行为，再扩展 aux 的私有操作；先针对 W only 做等价，不擅自调整 _update_track_states 或 native cleanup。frame_count、last_frame、state、is_activated、tracklet_len、conf、class、det_ind、history/feature buffers、ID counter 等按 pinned STrack.re_activate 精确核查；中心更新不能调用二次完整 re_activate 量测。额外日志 source_key=(sequence,frame,V|T,row) 及 group id 独立，原生 det index 原义保持。
- C1 用单独 runner 或 adapter，原主行/分数/特征顺序不可变；新辅助行有自己的映射 provenance 和真实 RGB embedding（尤其 high score），GMC 必须沿用相同 frozen cached transforms，即使输入检测用于背景遮挡的数量改变；记录因控制构造产生的差异。不运行修改后的 GMC 重新估计，不假称额外框未进原生起轨。
- M0/off 和 shadow、force_reject 只记录不干预；独立 full 先对照原 BG_W0 30 段 MOT、W 事件、原生 events、正式 state hashes（可比字段），不读取 GT。新诊断日志单列。任何差异先修复，不允许通过删除不利序列通过 parity。
- 定义 new runner 的 init/check/export/calibrate/smoke/r0/full/evaluate/summarize 入口（实现后再写实际命令），独立子进程、防重复输出目录、断点恢复需验证完整文件和输入/代码 fingerprint。训练侧校准锁完成且固定 smoke 技术通过才执行完整 30 段，科学结果正负不影响执行。

## P3 最低测试集
1. 真实 paired source 行数/顺序/空帧、1-based frame、宽高/letterbox、方向/逆映射、映射无效/越界、GT 泄漏拒绝；C2 shift 保持来源 key/score/q，记录越界。
2. 多候选/两轨迹/跨模态同目标/与已占用目标近重复/歧义/无有效候选/绝对几何门/预算/平局/不匹配选项；一对一分配不得重复 group/raw key。
3. 中心 KF 数值对照、R block 正确、Joseph PSD、交叉协方差、零残差、非有限与负宽高；只一次 predict/measurement、原 ID/外观不变、W 成功优先、W 拒绝才可进入 TCMO、跨帧 spent/rearm、finalizer 拒绝回滚、无假 det_index、正常事件和 B 记忆保持。若辅助 frame 进入常规 history 或负索引读取主检测必须测试失败。
4. 源码级 pinned hook 唯一定位，真实 adapter 的 off/shadow/force-reject、W-only 与原 30 段等价性；首个 TCMO 真实提交前 state/event 与 M0 一致，其他已匹配 ID/输出不变；不同模态拒绝/无观测正常退出。逐段独立 reset，不改变原 G/B/W。
5. 评测与归因：同一 TrackEval、完整范围、GT key、有效/ignore、跨源重复/竞争、unknown 不提升标签、FP/FN/IDSW 对账、标准聚合、C1 新增 row 的真实 feature、断点和 hash 改变失败、全部负向结果报告。单测和 synthetic smoke 不代表真实科学验证。

## P4 全量与产物
每组 30 段、12590 输入帧；TrackEval CLEAR 有效帧另列（不能用有效帧代替输入范围）；冻结源不匹配就单列阻断。按任务书先生成 R0 统计、随后执行 M1/M2/M3/C1/C2 + 已核验 M0。输出 1/3/5/10 帧后状态传播、B 恢复变化、误报/漏检/换号端点及图卡。所有新增指标由完整轨迹计算，冻结候选诊断不是新 HOTA。R2 只有下一轮决定后实现，不擅自扩大搜索。
