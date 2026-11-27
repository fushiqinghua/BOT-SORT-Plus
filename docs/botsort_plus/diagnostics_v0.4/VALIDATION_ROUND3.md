# Round 3 实现复核与软件验证

2026-11-27。基于 `7d549bc8bbddfefe99fbb8119ea13d40a4b5bd6d`。本记录核验代码机制与执行流程；没有运行新的 VT-Tiny-MOT 全量实验，没有新的真实数据集性能结论。

## 1. 最终结果

**112 项测试通过，0 失败、0 错误、0 跳过；另有 88 个参数子检查通过。** 其中新增 round 3 测试 66 项，相关既有回归 46 项。子检查不作为额外独立研究样本或算法实验计数。一次与业务无关的 pytest 插件弃用提示未影响结果。

[机器可读记录与源码摘要](validation_round3.json) · [运行说明](NEXT_JOINT_ROUND.md) · [任务状态定义](../../../configs/diagnostics/v0.4_joint_round3_tasks.json)

测试环境：Python 3.12.14、BoxMOT 发行包 16.0.11、PyTorch 2.5.1 CPU、torchvision 0.20.1 CPU、NumPy 2.3.5、SciPy 1.17.0、OpenCV 4.11.0、lapx 0.13.1。测试核对实际上游源码摘要，而非只信包版本。正式本地实验继续沿用已跑通 B/GMC 的环境；软件测试环境不等于历史 GPU 性能环境。

TrackEval 固定提交为 `12c8791b303e0a0b50f753af204249e622d0281a`，核验 45 个源码文件；仅允许原基线约定的 CRLF/LF 表示差异。

## 2. 设计与实现对应

| 设计要求 | 实际实现与验证 |
|---|---|
| J2-S 仅增加 skip 候选 | `B_skip_only` 使用原生／恒等候选，`dual_low=false`；B 配置除名称外保持相同 |
| J2-F 拆开变换与保护 | `B_adaptive_no_dual_low` 使用原生／恒等／LightGlue，明确关闭保护；与已有 BG 比较保护作用 |
| 预测不读取 GT | 独立 worker 仅接收预测设置白名单，保留原 runner 的 no-GT audit hook；带 `gt_root` 的设置被拒绝 |
| 正式流程不受观察器影响 | 实际 KF/STrack、轨迹输出、原事件、公共 ID、记忆及额外门控计数均做一致性检查 |
| P 按同帧历史比较 | 在共同真实后验锚点复制轨迹，按原版预测／GMC 向前传播；每次实际观测后重新共锚；不逆变换历史框 |
| N 保留完整竞争 | 私有高分、低分、未确认阶段完整矩阵与分配，再进入私有候选池和 B；不新建公共 ID |
| C 比较真实观测成员 | 私有候选池使用相同残余与占用信息；跨分支按 `(frame,det_index)` 对齐成员和实际消费 |
| 活跃身份仅作为诊断竞争者 | 使用帧前记忆与相同几何／外观口径；不套用 Lost 年龄资格，不比较不同尺度的原生与 B 代价 |
| 判断保护是否转入 B | 独立记录真实低分拒绝边、候选观测成员、B 提名及接受；报告后来新增观测，避免以代价降低替代新证据 |
| 人审标签真正读取 | 新 helper 独立校验和读取 22 例人审，旧 E3 脚本及代理分析不改写 |
| 缺失证据保留未知 | 缺失旧身份锚点、特征、日志、GT 支持、窗口尾部、不同外观覆盖等分别报告 |

## 3. 已执行的关键核验

### 归档 B 与当前入口兼容

在独立进程中加载归档 `joint/adapter_v2.py` 的真实更新函数，与新 worker 的 B 路径比较；合成输入上的 MOT 和原事件逐字节相同。核心 adapter 仅加入默认关闭的观察器边界，以及与已归档联合实现一致的可选低分保护入口。

### 真实运行时的只读一致性

合成纹理图像、检测与明确标注的单位向量软件夹具覆盖出生、Lost、B 三观测恢复、低分续接、未确认轨迹和相机变换变化。使用真实固定 BoxMOT/KF/STrack，比较诊断关闭和开启结果。

独立 CLI 又验证 live skip，以及缓存模式的 J2-F、BG、`evidence_both`。合成相机分歧确实触发了 `dual_low` 拒绝；记录开关前后 MOT、事件和决策记录一致。带状态的 `LostCameraMargin` 拒绝计数没有被影子重算污染。

### 完整执行链

一个六帧合成子集通过实际 `run` 命令执行了：J0、J1、记录关／开试跑、两组 J2、官方 HOTA/CLEAR/Identity 评测、BG 只读回放、两份 J3 汇总、E3 清单准备、证据 ZIP。GT ID 0、忽略标注、空帧、重复检测分数、身份切换、未知旧身份和右侧截断另有覆盖。

该测试仅验证软件预期。合成缓存有显式标记，不冒充 LightGlue 在真实视频上估计出的矩阵；本机没有执行 live LightGlue GPU 推理。

### 来源、范围与输出完整性

缓存候选集、policy、源码／模型来源、变换选择、备用矩阵、帧顺序与像素摘要均核验。试跑范围不足、配置／源码／参考改变、数据截断、重复复核样例、错误身份键、缺少复核者或输出覆盖均会失败。评测子集只复制原 MOT 并显式投影协议，不修改轨迹内容。

## 4. 必须保留的解释边界

- P 是共同锚点之后的条件影响；锚点本身可能已包含更早的 B/GMC 影响。
- N 在 J2-F 中完整运行；BG 开启 `dual_low`、或 evidence 开启附加门控时，纯 N 重算明确不可用。这样避免把不同关联规则混成纯 warp 反事实。
- evidence 的额外恢复门控不参与 P/C 再调用；此时保留被动信号，不能把未重算结果当作通过。
- C 使用实际残余与消费日程；影子分支产生的变化不会回写正式状态，也不能报告为完整算法净收益。
- 身份共同支持分数是诊断分数；历史或有效外观覆盖不同的比较保留未知，不宣称已经校准为概率。
- J1 的 GT 覆盖集合、P/N/C 变化量与连续 FP 行数都不是可相加的独立 HOTA 收益。
- E3 的 22 例仍需独立人工复核；`uncertain` 允许保留，标签仅用于实际审阅范围。
- J4 等待真实诊断结果后选择一个实现；本次没有提前加入新门控或新网络。

## 5. 本地还需要验证什么

本地实际 30 段回放仍需确认历史 B/BG 轨迹一致性、新对照正式指标、跨原视频效果和真实资源代价。若 full 数据上的影子一致性失败，应保留日志、停止解释该阶段，不修改检查结果来继续。

本批结果仍属于开发性评估；原来的官方测试序列已参与研究设计。缓存和观察器计时不代表检测＋ReID＋相机估计＋跟踪的端到端速度。

## 6. 复现软件测试

在具备已固定的 BoxMOT、NumPy、SciPy、OpenCV、PyTorch 和 pytest 的环境中，将环境变量 `J1_TRACKEVAL_ROOT` 指向固定 TrackEval 目录，从仓库根目录运行：

```bash
python -m pytest -q tests/test_v04_joint_analysis.py tests/test_v04_joint_camera.py tests/test_v04_joint_e3_review.py tests/test_v04_joint_probe.py tests/test_v04_joint_probe_analysis.py tests/test_v04_joint_worker.py tests/test_v04_joint_suite.py tests/test_v04_joint_suite_integration.py tests/botsort_plus/test_native_adapter.py tests/botsort_plus/test_a2_native.py tests/botsort_plus/test_identity_native.py tests/botsort_plus/test_diagnostic_capture.py
```

没有运行时或固定评测源码时，部分集成测试会明确跳过；不能把带跳过的执行结果等同于本记录的完整验证。
