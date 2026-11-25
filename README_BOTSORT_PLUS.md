# BoT-SORT-Plus：当前诊断与模块文档入口

## 当前 v0.4：运动与模糊诊断

新增四组可运行实验：预测/观测误差、GMC 可靠性、B 恢复后稳定性、图像质量与人工模糊复核。固定 `B_early_moderate`，先检查诊断开关的输入、MOT 输出与原有事件完全一致，再执行离线分析。

**本地运行步骤、配置和证据打包：[诊断实验说明](docs/botsort_plus/diagnostics_v0.4/README.md)。** [实现机制与统计口径](docs/botsort_plus/diagnostics_v0.4/MECHANISM.md) · [软件验证](docs/botsort_plus/diagnostics_v0.4/VALIDATION.md)

[仓库首页与当前进展](README.md) · [本地路径配置示例](configs/diagnostics/motion_blur_v0.4.example.json)

## 历史 v0.3：C2 身份可信度

以下保留原 C2 的设计与复现实验入口；当前四组诊断使用上方 v0.4 的固定 B 配置。

本版从 `iteration/v0.2-a2` 的 `6d9c258556a143036cebde891c93079a772b9945` 接续实现。在 **YOLO11-M RGB–T + BoT-SORT + 微调 RGB ReID + B** 主基座上，将“接受一次几何关联”与“允许这次观测改写身份外观”分开。

[真实数据消融与后续迭代](results/vttiny/iterations/v0.3-c2-identity/README.md) · [B 恢复、红外及 C2 预热的模块设计与复算](docs/botsort_plus/FOLLOWUP_DESIGN.md)

新增 C2 包括：

1. **关联约束**：读取上一帧的可靠身份锚点，以历史一致性、当前阶段双向竞争间隔和现有 crop quality 控制 ReID 加成。可信度不足时回到该原生阶段的几何代价，保持空间门控和匹配阈值。
2. **记忆写入门控**：几何续轨继续执行，外观写入另行判断；同时覆盖原生 `smooth_feat/curr_feat/features` 与 B 的 `MemoryBank`。B 恢复当帧隔离写入，后续可靠原生匹配可以继续更新。

原版 C2 使用已缓存的 RGB 特征，配置关闭 A 和旧 C，并限定 `adapter_mode="v2"`。C2 shadow 的冻结输入回放与 B 参考在全部 30 段逐字节一致。

## C2 实现与复核入口

| 内容 | 文件 |
|---|---|
| 纯 NumPy 身份证据、因果快照、代价约束 | `botsort_plus/identity_reliability.py` |
| 原生关联与双记忆写入接入 | `botsort_plus/identity_adapter.py` |
| high / low / unconfirmed / B / birth 调度 | `botsort_plus/native_adapter.py`、`adapter_v2.py` |
| 设计公式、代码映射与局限 | [C2_DESIGN_AND_REVIEW.md](docs/botsort_plus/C2_DESIGN_AND_REVIEW.md) |
| 接入和实际数据实验步骤 | [C2_INTEGRATION.md](docs/botsort_plus/C2_INTEGRATION.md) |
| 参数和消融矩阵 | [C2_CONFIGS.md](docs/botsort_plus/C2_CONFIGS.md) |
| 新日志审计和统计 | `tools/audit_botsort_plus_identity.py` |
| 旧版本差分回放 | `tools/check_botsort_plus_legacy.py` |
| 实际软件验证记录 | [VALIDATION.md](docs/botsort_plus/validation_v0.3/VALIDATION.md) |

## 历史 C2 运行顺序

| 配置 | 作用 |
|---|---|
| `13_B_ref.json` | 同基座参考 |
| `27_B_C2_shadow.json` | 观察模式，与 B_ref 逐文件一致 |
| `28_B_C2_memory.json` | 仅应用记忆写入门控 |
| `29_B_C2_association.json` | 仅应用关联约束 |
| `30_B_C2.json` | 完整 C2 + B |
| `12_baseline_ref.json`、`31_C2.json` | 补充 B/C2 互补性对照 |

沿用上轮固定环境、原始 RGB 图像、完整真实 ReID 缓存和检测缓存。先预检：

```bash
python tools/run_botsort_plus.py --baseline-root /path/to/frozen_rgbt --config configs/botsort_plus/30_B_C2.json --images-root /path/to/VT-Tiny-MOT --features-dir /path/to/features_rgb_plus --output-dir outputs/botsort_plus_c2_full30 --preflight
```

预检 `READY` 后用 `--trace` 替换 `--preflight`，按表顺序回放。完整 shadow、审计、评估命令见接入文档。运行器拒绝覆盖既有结果，参数修改后应使用新的输出根目录。

原版完整 C2 加到 B 后，实测 HOTA 从 38.727 降到 38.504，IDF1 从 43.862 降到 43.494。拆分消融定位到 B 恢复库的写入门控。确认记忆和连续低分预热的后续对照见 [完整实验索引](results/vttiny/iterations/v0.3-c2-identity/README.md)。

前版入口：[v0.2 说明](docs/botsort_plus/history/README_v0.2.md) · [A2 真实实验与错因](results/vttiny/iterations/v0.2-a2/README.md)。
