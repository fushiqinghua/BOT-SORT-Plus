# W v0.1 数据合同

## 1. 已实现：R1a 输入

输入目录下逐序列 .json 文件，序列名为文件 stem。每个文件为对象，frames 是列表；每帧含正整数 frame、detections 列表。每行严格为：

~~~text
[x1, y1, x2, y2, score, detector_class]
~~~

空帧 detections=[] 合法。必须保留原行顺序。重复帧、非法尺寸、非有限值、错列数等应报错，不能默默丢行后重编号。审计记录实际帧覆盖和缺口，不虚构缺失帧。

解析语义：每行先转 float32，再按原生拼接整数行号后升为 float64 的行为划分 high/low。原生阈值使用命令行显式传入的数值。

S0 = 非 high 且非 low，且属于低于/等于 low 门槛的真实行。等于 high 门槛的遗漏另列，不混入 S0。

## 2. 已实现：R1a 输出语义

schema 的具体字段见 tools/research/wem_v01/cache_audit.py；根 summary 明确状态 INVENTORY_ONLY_NOT_W_OPPORTUNITY。

原始行索引从 0 开始，frame 保持输入帧号。每一候选的身份键必须能够定位 sequence / frame / row。input SHA256 对原文件字节计算；不要用 JSON 重排后的摘要伪装原文件摘要。

ReID 策略标记仅模拟 feature_export.py 的 float32 score > 0.1 规则。它不证明现存 npz 中确有有效特征，不证明裁剪可辨别，不是身份标签。

本工具不读取 GT、轨迹文件、图像或 ReID 特征，因此不能填充当前 owner、轨迹资格、真值身份、W 准入或恢复成功。

## 3. 拟实现：统一 ObservationKey

~~~text
sequence
frame_id
source: native_cache | extra_post_nms | pre_nms
original_row_or_null
raw_proposal_id_or_null
~~~

S0 保留原行号。S1/S2 用 Registry 显式解析，不得用 det_ind=-1 使原数组读取误落到最后一行。

候选来源保存多标签：低于 native low、低于 export floor、被 NMS 抑制、容量截断等。suppressed_by 是抑制关系，不是同一身份标签。互斥的 first_exclusion_stage 必须给出明确处理顺序，并保留多标签交集。

## 4. 拟实现：R1b 状态快照

按 reference / sequence / frame 存储：

- 帧首 confirmed Active ID、所有状态的类型与上次真实观测时间；
- 本帧正式 KF 的 selected/backup 预测、均值与协方差；
- 使用的 GMC 及推进次数；
- 原生 high/low/unconfirmed、B、birth 的观测更新与来源键；
- B 已消费、预留、仍存活候选的当前真实观测；
- 没有当前观测的历史占用/预测区域，另列而非冒充真实检测；
- 最终无 W 整理后的输出与保留轨迹状态；
- 当前 S0 候选与每条相关轨迹的原生 low / GMC guard 可行性；
- 常规 raw 检测历史，而非 STrack.history_observations；
- 采样前后既有事件、KF、候选池和记忆摘要。

旧 TCO 采样不能默认覆盖这些字段。对无法从旧日志还原的字段，重新只读回放，不猜测。

逻辑产物为 snapshots.jsonl、candidate_ownership.jsonl；实现可用压缩形式，但 schema 和解码后摘要须固定。

## 5. 拟实现：独立可信锚

trusted_raw_observations、trusted_appearance、anchor_mean/covariance、anchor_track_state、anchor_clock、last_trusted_frame、pending_regular_confirmation、weak_episode_used。

先用旧锚和已有历史验证常规观测，再决定写入。弱观测不写入。重置需要两个不同帧的待确认常规观测，两者不能通过先写入第一个观测来相互自证。

锚点无效期间禁用 W；重建与正常预算重置区分。锚点状态不得复制 W 改变后的正式 TrackState，避免间接修改预测分支。

## 6. 拟实现：离线标签

线上预测进程不接收 GT 路径。独立离线进程使用官方 GT 和冻结参考输出，标记：

- same_identity；
- other_identity；
- confirmed_background；
- unknown / ignored。

至少两次过去真实常规观测支持同一 GT，才形成严格历史身份锚；当前框低 IoU 不自动视为背景。模型辅助标签和人工确认分别记录。

offline_labels.jsonl 保存 source IDs、锚定依据、当前标签、可见性、判定规则、版本和人工状态。机会汇总 opportunity_summary.json 同时报轨迹帧、去重 GT 帧、episode、目标轨迹和来源组计数。

GT 仅用于分析，不能生成原候选中不存在的框、改变在线选择或回写状态。

## 7. 拟实现：R2 量测对照

frozen_pair_effects.csv 每行绑定同一正式 KF 先验和候选键，保存 raw candidate、prior、一次原 KF update 的 posterior、同 ID GT 的 IoU/中心误差、原生可行边、GMC guard、所有权和候选组。

constrained_opportunities.json 的 GT 辅助选择仍受轨迹—候选组一对一约束。报告“当前候选集合可达机会”，不得命名 HOTA_upper_bound。

## 8. 拟实现：R3 校准与 R4 提交

calibration_lock.json 必须含数据划分、来源/母视频验证状态、输入与特征哈希、候选池版本、错误接纳预算、工作点、选择规则、未知处理和留出用途。无有效校准锁时 W 关闭。

R4 每次提交记录 weak_proposed / verified / updated / committed / rejected_finalization，保存 raw score、raw box、prior/posterior、所比较竞争身份、拒绝原因和弱预算。

Coast-only / W-coast 的输出标为 prediction，不伪装成真实量测，不调用 update(predicted_box)，也不刷新可信 raw 历史。记录预测输出时间与最后真实观测时间的差异。

## 9. 发布与可复核性

每个批次保留：代码 commit、配置快照、输入原文件 SHA256、环境版本、执行范围、每阶段退出状态、产物哈希和阶段结论。

失败、中断、零采用、无机会和未知标签都应保留。真实结果为空时写 NOT_RUN 或 NOT_AVAILABLE，不填 0。公开归档只做路径别名化，不改数值，保留路径转换说明。

