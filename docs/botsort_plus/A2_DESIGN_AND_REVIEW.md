# A v0.2：机制、实现与理论一致性复核

## 1. 问题与边界

本版依据 2026-11-25 诊断包进行重设计。原 A 相对 baseline：HOTA 37.8608→36.8092，FP +12,247、FN −7,064、IDSW +1,127；配对新增覆盖为 7,147 个 GT 目标帧，丢失原有覆盖为 83 帧。840 次换号发生在弱出生当帧，其中 671 次前一 ID 仍在 lost 池。936 个始终远离标注的弱 ID 产生 8,306 个 FP 目标帧。

这些结果支持检查“候选存在确认”和“公开新身份”的交接，不证明所有 lost ID 均可安全恢复，也不证明所有 never-near 输出在画面语义上都是不存在的物体。原 B 在真实实验中有效，故主要验证目标为 A2B 相对 B 的增量，而非仅超过退化的 A。

来源：用户提供 `VT-Tiny-MOT_模块A诊断_模型分析包_20261125.zip` 中的 `模块A_实现与错因诊断.md`、`analysis/attribution.json`、`analysis/metrics_with_controls.csv`、`背景_四组消融/metrics/comparison.csv`。本补丁不携带原图、GT、模型或完整诊断包。

## 2. E：分阶段确认

实现：`botsort_plus/birth_policy.py`。该模块只维护 A 自己的私有状态，不修改 CandidatePool、B、KF、正式 ID 或身份记忆。

令 H 为最近 W 帧通过候选关联的真实观测数，S 为这些观测分数的算术平均。新确认不读取候选的旧几何累计 evidence，也不调用旧 ready()。

第一阶段：当前帧有检测，最近 5 帧至少 3 次观测，S≥0.18，记录 armed_frame=t0，进入 VERIFYING。只积累私有证据，不公开。

第二阶段：t0 之后 3 个实际视频帧内，至少 2 次有效观测，其自己的均分≥同一门槛，且当前帧有检测，才 READY。后续观测严格满足 frame>t0，不能重用第一阶段观测。0.20 配置是仅改变门槛的敏感性对照。

连续合格观测最早在第 5 帧公开，而非第 3 帧。在 15 FPS 下最早额外延迟约 0.13 秒；实际缺帧、冲突会增加延迟，应记录分布。

### 时钟与过期

- 对每个仍在共享池的候选逐帧 evaluate，空帧也计时；跳过已知候选的视频帧会明确报错。
- 初次复核在 deadline 当帧仍可成功；若失败，epoch_start 推进到 deadline，之后重新确认不能复用该边界以前的观测。
- 失败只重置 A 私有确认状态，共享候选与原始历史保留，B 仍可恢复。
- READY 若被身份门暂缓，之后每次提交仍要求最近 3 帧且晚于 t0 的至少 2 次合格观测及当前命中；旧 READY 不是永久许可证。
- 同帧重复调用幂等；同帧改变历史、未来帧、重复或乱序观测均失败。

### 对照模式

| mode | 规则 |
|---|---|
| legacy | 调用原 Candidate.ready()，保留原几何证据含义 |
| score_window | 最近 5 帧 3 次命中及均分门，直接就绪 |
| fixed_window | 最近 8 帧 5 次命中及均分门，近似延迟对照 |
| two_stage | 第一阶段 3/5，再以后续 2/3 次新观测复核 |

均分不是校准概率；相邻帧的新观测也不是统计独立证据。中等分数且持续的误响应仍可能通过两阶段。几何仍用于共享候选的连接和容量排序；旧 evidence 在共享池中继续作为固定管理优先级，不能解释为 A2 的存在证据。

## 3. I：新身份准入检查

实现：`botsort_plus/birth_identity.py`。A 只返回 CLEAR_NEW / DEFER / REJECT，不选择或恢复旧 ID。RECOVERED 只能由现有 B 的实际接受结果产生；适配器在 B 接受后消费候选，A 不再处理它。

针对仍物理保留且 state=Lost 的轨迹，先检查 1≤当前帧−最后观测帧≤max_age；max_age 默认 30，并校验不得超出实际 FPS 缩放后的 native 保留期。

当前预测在当帧 KF predict 与 GMC 之后、接受检测更新之前保存。只用各历史帧当时的预测，不倒推历史。当前几何采用原有限尺度/中心门：明确空间或尺度不可能的旧 ID 可以排除。当前预测缺失则是未知，不能当作已分离。

当前附近仍相容时，历史缺失、历史不一致、观测与 last_seen 重叠、外观缺失都不能自动证明新身份。历史对齐信息用于解释风险原因；首版不以某次历史失配强行消除当前附近的旧身份风险。质量合格的当前外观若明显矛盾，可排除此旧身份；历史外观均值仅用于诊断。

这是一项保守的出生准入检查，不是新的恢复算法，也不是证明两个目标相同的概率模型。只靠有限几何会阻止一些近邻新目标，需要实际覆盖/身份收益权衡。

### 不可续期的等待

首次发现合理冲突即开始 5 帧预算；AB 模式继承 B 更早的 reservation_start。缺帧或竞争 ID 改变不重置。elapsed≥5 且仍冲突返回 REJECT，不自动允许新 ID。之后若出现真正空间分离或可信外观反证，可以重新 CLEAR_NEW，但旧时钟保留，再出现冲突仍按原预算拒绝，且必须通过 E 的近期证据复核。

候选整体寿命和容量保持原规则。拒绝的是本次 A 出生提议，不删除 B 可用的历史，不额外延长 track_buffer。

## 4. v2 基座与出生事务

实现：`botsort_plus/adapter_v2.py`。步骤为原生高/低/待确认关联→更新候选→B→原生高分出生→A 的 E/I→去重及提交。

### 单列基座变化

1. 原生第一轮使用完整 native lost 列表，与固定 BoxMOT 的枚举一致。临时 Removed 对象的原生处理仍由原生流程决定；B 和 I 的资格独立过滤真实 Lost 状态和年龄。
2. A 的私有候选不额外抑制原生高分起轨。B 已有的候选歧义抑制和 reservation 保留。
3. 弱出生采用提议与提交；原生 duplicate 检查属于全部 v2 A 组共用基座，关闭 I 时仍执行，不能计为 I 专属收益。

原配置继续走 legacy adapter，便于重放旧 A/B/AB。B_ref 使用 v2 基座，不宣称一定逐字节等同历史 B；应保存两者并单独审查差异。A1_ref/A1B_ref 使用 v2 基座+旧确认规则，因此也不是历史 A/AB 的别名。

### 事务顺序

- 私有 STrack 先设置 frame_id=start_frame=当前帧，使用真实 native remove_duplicate_stracks 对预计 lost 列表做预检；此时未 allocate 公共 ID。
- 若预检拒绝，保留候选历史，不写正式 memory，不增加 ID 计数。
- 通过后才 activate，提交时 is_activated=True 代表已完成自有确认。
- native 最终 _prepare_output 之后再次核实该轨确实保留并输出，才消费候选、写正式记忆和 birth_committed。
- 最终拒绝的防御分支同样保留证据；可能有未使用的 ID 空号，其本身不是跟踪错误。
- candidate_first_frame 与正式 birth_frame 分开；不伪造轨龄、不回填、不修改过去 ID。

## 5. 不改变或不纳入首版的因素

`candidates.py`、`geometry.py`、`assignment.py`、`recovery.py`、`appearance.py` 的算法内容保持 v0.1 字节。新 A 状态与旧池 evidence 分离。B 不受 A 的均分与五帧公开延迟限制，仍可在其自身三次弱观测条件满足时恢复旧 ID。

不加入速度继承、公开弱轨强制短寿命、弱来源恢复禁令或额外检测器。track_origins 只作诊断，不进入关联或恢复资格。C 未在新配置启用；也没有凭 RGB 编码器生成假的 thermal 特征。

## 6. 日志口径

- `birth_evidence`：每个候选的阶段、均分、真实支持帧、armed/deadline/epoch、是否近期就绪。
- `birth_identity`：旧身份列表、当前/历史几何及可用外观、风险原因、首次冲突与 elapsed。
- `birth_proposed`：通过 E/I 后的提议，含 shadow 标识。
- `birth_rejected` / `birth_deferred`：阶段及原因；可能每帧记录同一个候选，应区别事件次数和唯一候选数。
- `birth_committed`：最终保留且开始输出的新弱身份。
- `weak_birth`：为旧事件读取器保留的同一事务别名，带 alias_of=birth_committed；两者不能相加。
- `first_emitted`：实际首次输出。
- `recovered.track_origin`：可检查 B 恢复是否来自 weak_birth。
- 序列末 `track_lifetime`：实际输出帧数、首末输出、恢复次数、原生高分接受次数。不是持续区间内每一帧都输出的假定。

仅在独立 GT 分析中计算新增覆盖、错误恢复和 FP 寿命；在线日志不读取 GT。`audit_botsort_plus_a2.py` 只做结构与输出一致性检查，不计算或推断 HOTA。

## 7. 复核映射

| 设计要求 | 直接实现 | 验证 |
|---|---|---|
| 几何不能替代存在支持 | BirthPolicy 新模式不读 evidence | test_birth_policy、test_a2_native |
| 后续观测不复用 | frame>armed_frame | test_birth_policy、test_a2_runner |
| 空帧计时、失败轮隔离 | consecutive evaluate、epoch_start | test_birth_policy |
| 等待后重新检查近期支持 | READY_STALE 分支 | test_birth_policy |
| A-only 不恢复 ID | BirthIdentityGuard 只读、B 独占 re_activate | test_birth_identity、test_a2_native |
| 到期不变成新身份 | elapsed>=wait_frames 时 REJECT | test_birth_identity、test_a2_native |
| B 继续使用被 A 拒绝的候选 | 共享池不因 E 失败删除 | test_birth_policy、test_a2_native |
| 原生高分优先 | native birth 在 A 提议之前 | test_a2_native |
| 观察模式无输出副作用 | observe_only 在 activate 前返回 | test_a2_native、test_a2_runner |
| 去重拒绝不消费/污染 | 私有预检+最终提交核验 | test_a2_native |
| 不回填/不伪造轨龄 | 当前帧 activate 与输出 | test_a2_native、test_a2_runner |
| 原输入、GT 隔离、独立评价 | 继承原运行器/评估器协议 | 原有 IO、runner、evaluation 测试 |
| 安装不覆盖未知改动 | SHA 前置检查、备份/回退 | tests_packaging |

完整执行结果和代码指纹见 `validation_v0.2/`。软件夹具上的通过仅验证机制与工程合同；真实数据收益仍需按 A2_CONFIGS 的二因素矩阵回放。

## 8. 学位论文中的表述

低分起轨、轨迹感知初始化等思想已有研究基础，可参考 SFTrack（IROS 2024，https://songinpyo.github.io/SFTrack-Project/）与 TrackTrack（CVPR 2025，https://openaccess.thecvf.com/content/CVPR2025/html/Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.html）。本版不声称首次提出这些一般概念。

若消融支持，可将改进表述为：面向 RGB–T 微小目标的持续弱观测和短缺口身份竞争，将候选连接、分阶段存在确认与新身份准入分离，并与现有短轨迹恢复协同。应以 A2B 相对 B_ref 的真实新增覆盖、HOTA 和身份质量支持结论，同时披露延迟、误报与近邻新目标被拒绝的代价。
