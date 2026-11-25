# C2 v0.3.0 软件验证与独立复核

日期：2026-11-25。代码基于 `6d9c258556a143036cebde891c93079a772b9945`（`iteration/v0.2-a2`），交付分支为 `iteration/v0.3-c2-identity`。

**结论：软件机制验收通过，未发现阻断性代码问题。C2 的真实数据准确率和端到端速度尚未测量。**

## 1. 实际执行结果

完整回归命令：

```bash
.venv/bin/python -m unittest discover -s tests/botsort_plus -p 'test_*.py' -v
```

| 项目 | 结果 |
|---|---|
| 全部测试条目 | 289 |
| 通过 | 288 |
| 失败 / 错误 | 0 / 0 |
| 跳过 | 1：项目原有可选的固定 TrackEval 集成测试，未设置 `VTTINY_TEST_TRACKEVAL_ROOT` |
| 本轮新增测试 | 60，全部通过，无跳过 |
| 完整回归耗时 | 77.329 秒，仅测试运行时间 |
| 编译检查、Git 空白检查 | 通过 |

评估器源码未改动。跳过的可选测试不能当作已验证；本报告也没有把软件合成夹具当成真实模型结果。完整输出见 [test_output.txt](test_output.txt)，机器可读摘要和源码 SHA-256 见 [validation_report.json](validation_report.json)。运行环境为 Python 3.12.14、BoxMOT distribution 16.0.11、PyTorch 2.8.0+cpu，详细版本和六份 BoxMOT 实际源码指纹记录在 JSON 中。用户继续沿用自己已验证的环境，无需按本次临时 CPU 环境重装。

## 2. 新增机制测试覆盖

| 测试文件 | 条目数 | 实际验证 |
|---|---:|---|
| `test_identity_reliability.py` | 32 | 代价上下界、因果快照、冷启动、缺失/零向量、双向竞争、未知竞争者、几何/外观门控、恢复隔离、持续渐进变化、每帧写入约束、80 组独立枚举对照及 1280 维尺寸测试 |
| `test_identity_native.py` | 13 | 真实 STrack/SOF/KF 的状态断言、配置桥接与非法组合 |
| `test_identity_audit.py` | 14 | 完整/缺失/损坏日志、协议开关、实际 writer 日志、B/low 约束、shadow、分数融合代价与浮点误差 |
| `test_identity_runner.py` | 1 | 七配置实际 CLI、预检、输入/输出指纹、实现哈希、事件和两个实际审计 CLI |

原生测试使用真正的固定版本 BoxMOT、Kalman filter 和 SOF GMC。图像为明确生成的纹理夹具，embedding 为明确构造的单位向量，未称为真实 ReID 网络输出。

关键原生断言如下：

- 几何续轨遇到与历史冲突的 embedding 时，ID、帧号和位置仍正常更新；原生 `smooth_feat`、`curr_feat`、feature deque 及 custom bank 的原型、质量和计数保持不变。
- low 阶段的可信观测可更新 custom bank，仍不新增 native appearance 写入。
- B 经过三帧低分观测恢复原 ID；恢复当帧两套记忆不写，后续可靠 high 可以恢复写入。
- 低质量原生出生会清除其 native 外观种子；之后的合格高分帧能够重新建立历史，避免因缺失种子永久无法启动。
- 仅关联消融不会误应用对 native/custom bank 的写入门控。
- shadow 与 B_ref 每帧预测的形状、dtype、字节完全一致；原生和自定义记忆、轨迹列表、几何状态、候选池、B 时钟、预测历史和全局 ID 计数也一致。
- 同帧 high 提交后，unconfirmed 读取的仍是帧首快照：真实运行中 live 更新数已为 4，snapshot 更新数仍为 3，snapshot 向量也保持前帧值。
- 实际成熟外观并列竞争中，关联加成被削减，匹配按几何区分；输入检测与特征数组不被原地修改。

## 3. 旧版本差分回放

```bash
.venv/bin/python tools/check_botsort_plus_legacy.py --output /tmp/c2_legacy_compatibility.json
```

工具从固定基准 Git 对象提取旧代码，对旧版与新版分别启动隔离进程。原有 `00`—`26` 共 27 个配置，每组 60 帧，每个版本 1,620 次逐帧更新，覆盖原生/A/B/C、A2、shadow、低分候选、恢复、空帧、阈值边界等软件场景。

**27/27 的预测数组哈希及完整事件序列哈希均一致。** 每组原始哈希保存在 [legacy_compatibility.json](legacy_compatibility.json)。这是选定软件夹具的兼容性证据，不扩大解释为已经逐帧重跑了真实 30 段数据。

与基准提交直接比较确认以下八份算法文件字节不变：`recovery.py`、`candidates.py`、`appearance.py`、`geometry.py`、`assignment.py`、`birth_policy.py`、`birth_identity.py`、`types.py`。新模块通过适配层控制既有写入入口，未改写 B 的匹配器或 A 的算法文件。

## 4. 真正 CLI 接入

测试通过实际 `tools/run_botsort_plus.py` 分别运行：`12_baseline_ref`、`13_B_ref`、`27_B_C2_shadow`、`28_B_C2_memory`、`29_B_C2_association`、`30_B_C2`、`31_C2`。每组 12 帧，包含 high、low、unconfirmed、缺帧与 B 恢复，夹具没有 GT。

验证了 `implementation_version=0.3.0`、完整配置、每份实现文件哈希、特征指纹、输出哈希、预测唯一性及顺序。第 9 帧确实经 B 恢复原 ID；B_ref 与 C2 shadow 的 MOT 文件逐字节一致。既有 A2 完整性审计和新增 C2 机制日志审计均实际调用并返回 `PASS`。

## 5. 理论—实现独立复核

独立复核者重新执行 core 32 项、native 13 项及 audit 14 项，共 59 项全部通过，并逐项检查了真实对象状态断言。主回归另外验证了七配置 CLI 接线。

| 理论要求 | 落实及证据 |
|---|---|
| 只控制既有 ReID 加成 | 以真实原生基础代价 `base_cost` 为上界，参考代价为下界；保留 unconfirmed 的分数融合尺度 |
| 当前观测不能先写后证明自己 | 帧首冻结、接受后提交、同帧最多写一次，跨阶段实际状态验证 |
| 匹配接受不等于允许改写身份 | `apply_match` 在 native 更新之前移除被拒特征，同时控制 custom writer |
| 冷启动/缺失不虚构可信证据 | 临时种子、成熟计数、缺失为未知、无竞争者不使用无限 margin |
| B 作为已验证基座继续工作 | B 算法源文件不变，恢复同 ID；新记忆确会影响未来 B 输入，未承诺启用后 B 结果恒定 |
| 可解释消融和安全观察 | 两个干预开关独立，shadow 输出及共享状态精确一致 |

已明确写入设计文档的限制：竞争集合只覆盖当前阶段；质量乘积可能过度削弱 ReID；错误初始模板及持续外观突变没有自动重置；几何噪声和恢复隔离可能造成真实观测写入不足。当前未发现与这些已声明边界相矛盾的代码实现。

## 6. 仍需真实实验完成的部分

本工作目录没有完整原始 RGB 视频帧与冻结的真实 ReID 特征缓存，未用占位数据替代。HOTA、AssA、IDF1、IDSW、FP/FN 及端到端 FPS 均未产生新的实测数值。

下一步按接入文档先做 B_ref/shadow 一致性检查，再固定输入运行记忆、关联及完整方案消融。审计通过仅说明协议和日志自洽；是否改善身份连续性、是否降低误写、是否存在覆盖率损失，要由真实预测与离线 GT 对齐回答。
