# E1-B 冻结 Lost 关联失败阶段

冻结 `B_early_moderate`，30 段、12,590 帧。单序列和全量只读观测的 MOT、原生事件和 B 相关事件逐字节一致。
使用预关联矩阵、原始候选及预测历史、关联前外观原型和冻结检测特征，独立重算 native high 与 B 的原始代价、可分配矩阵、分配与 B 拒绝理由。
确认标注匹配、失联时间及原轨迹跟踪状态仅用于离线分析。

主队列共有 7792 个轨迹帧。原 ID 无正确输出且有可用检测的行数为 1416，其中 9 行无法唯一配对检测。唯一配对为 1,407 行。
下表各分组的 GT 帧可重复出现，不能直接对去重列求和。规范化原因保留 `unknown`。

### 未消费低分框的 B 阶段

| 原因 | 轨迹帧 | 独立 GT 帧 | GT 轨迹 | 旧 ID 失联片段 | 序列 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `insufficient_B_history` | 596 | 594 | 153 | 419 | 24 |
| `candidate_seed_deferred:competing_links` | 178 | 178 | 54 | 69 | 15 |
| `appearance_contradiction` | 155 | 155 | 57 | 72 | 11 |
| `geometry_gate` | 87 | 82 | 16 | 28 | 14 |
| `cost_reject` | 35 | 33 | 16 | 19 | 12 |
| `overlaps_last_observation` | 34 | 34 | 23 | 24 | 11 |
| `ambiguous` | 10 | 10 | 4 | 4 | 2 |
| `outside_original_B_eligible_pool` | 2 | 2 | 2 | 2 | 1 |
| `candidate_observation_suppressed:occupied_track` | 1 | 1 | 1 | 1 | 1 |

### 未消费高分框的原生高分阶段

| 原因 | 轨迹帧 | 独立 GT 帧 | GT 轨迹 | 旧 ID 失联片段 | 序列 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `native_high_cost_above_gate` | 129 | 103 | 32 | 72 | 17 |

### 未消费高分框的 B 阶段

| 原因 | 轨迹帧 | 独立 GT 帧 | GT 轨迹 | 旧 ID 失联片段 | 序列 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `insufficient_B_history` | 51 | 48 | 20 | 42 | 14 |
| `geometry_gate` | 30 | 30 | 11 | 18 | 11 |
| `cost_reject` | 19 | 19 | 10 | 12 | 9 |
| `ambiguous` | 12 | 6 | 1 | 4 | 1 |
| `appearance_contradiction` | 11 | 11 | 8 | 8 | 5 |
| `candidate_seed_deferred:competing_links` | 6 | 5 | 5 | 6 | 3 |

### 已被其他轨迹消费的候选

| 原因 | 轨迹帧 | 独立 GT 帧 | GT 轨迹 | 旧 ID 失联片段 | 序列 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `geometry_gate` | 90 | 82 | 10 | 29 | 7 |
| `cost_reject` | 38 | 33 | 14 | 24 | 8 |
| `not_residual_consumed_high` | 25 | 20 | 7 | 18 | 3 |
| `appearance_contradiction` | 17 | 17 | 12 | 12 | 8 |
| `ambiguous` | 2 | 1 | 1 | 2 | 1 |
| `not_residual_consumed_low` | 2 | 2 | 2 | 2 | 1 |

处于 Lost 的旧轨均沿用冻结的原生低分关联资格。
未消费低分的 596 个历史不足案例，398 例只有 1 次实际观测，198 例只有 2 次且未达提前恢复证据。高分的对应数为 18 和 33。
B 原始候选未进入分配时，继续区分原生消费、占用导致的抑制、候选竞争与生命周期。
已接受一条关联也可能没有得到足够重叠的 MOT 输出。

详见 [逐帧失败阶段](E1_failure_stages.csv.gz)、[旧 ID 失联片段汇总](E1_episode_summary.csv.gz) 和 [只读记录校验](../E1_shadow_parity.json)。
