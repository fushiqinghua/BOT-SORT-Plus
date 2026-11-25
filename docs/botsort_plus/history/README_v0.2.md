> 历史归档：下文为 v0.2 交付时说明。当前 C2 接入以仓库根目录 README_BOTSORT_PLUS.md 为准。

# BoT-SORT-Plus v0.2：A 的分阶段确认与身份约束

本版本接续已完成真实实验的 v0.1：主基线为 YOLO11-M RGB–T + BoT-SORT + 微调 RGB ReID。针对 A 的持续误起轨和旧身份竞争，实现可独立开关的 E（存在确认）与 I（新身份准入），保留原 B/C 入口。

补丁接入后已完成三段 shadow 验收和 30 段、15 组全量回放。默认 A2B 采用 0.18，另有门槛 0.20 的对照。`docs/botsort_plus/validation/` 与 `validation_v0.2/` 记录补丁交付时的软件测试。实跑结果、逐帧证据和复算记录见[第二轮实验索引](../../../results/vttiny/iterations/v0.2-a2/README.md)。

## 主要入口

| 功能 | 文件 |
|---|---|
| 新存在确认与时间状态 | `botsort_plus/birth_policy.py` |
| 旧身份竞争的只读检查 | `botsort_plus/birth_identity.py` |
| 原生交接与出生提交 | `botsort_plus/adapter_v2.py` |
| 原 B 恢复 | `botsort_plus/recovery.py`（算法文件保持 v0.1 字节） |
| 运行、评估 | `tools/run_botsort_plus.py`、`tools/evaluate_botsort_plus.py` |
| 输出和事件一致性审计 | `tools/audit_botsort_plus_a2.py` |
| 机制与理论复核 | `docs/botsort_plus/A2_DESIGN_AND_REVIEW.md` |
| 接入和回放步骤 | `docs/botsort_plus/A2_INTEGRATION.md` |
| 完整消融配置矩阵 | `docs/botsort_plus/A2_CONFIGS.md` |

## 版本选择

- 原配置 `00`—`11` 保留，默认 `adapter_mode="legacy"`，继续走 v0.1 跟踪路径。
- 新配置 `12`—`26` 显式选择 `adapter_mode="v2"`。原生 lost 列表枚举、A-only 观察旁路与弱出生提交流程构成单独的 v2 基座。
- `birth.enabled` 是 A 开关；`birth_policy.mode` 选择 `legacy`、`score_window`、`fixed_window` 或 `two_stage`；`birth_identity.enabled` 是 I 开关。
- `birth_policy.observe_only=true` 记录判断、不提交 A 轨迹。A shadow 与 baseline_ref 比，AB shadow 与 B_ref 比。
- 新配置全部关闭 C；C 的接口保留，不需要为了运行 A2 另训 thermal ReID。

## 建议首次运行

先看接入说明并复用上轮已经验证的原图、完整特征缓存和固定 BoxMOT 环境。依次运行 `12_baseline_ref`、`13_B_ref`、`25_A2_shadow`、`26_A2B_shadow`，通过观察模式的逐文件一致性检查后，再运行 `16_A2` 和 `19_A2B`。

```bash
python tools/run_botsort_plus.py --config configs/botsort_plus/19_A2B.json --images-root /path/to/VT-Tiny-MOT --features-dir /path/to/features_rgb_plus --output-dir outputs/botsort_plus_v2 --preflight
```

预检成功后去掉 `--preflight`，添加 `--trace` 正式回放。始终显式指定新的 `--output-dir`；运行器的旧默认目录为兼容性保留，不会自动覆盖旧结果。

实际预测不读取 GT、不会回填确认前输出、不会改变过去 ID，也不会用测试图像或伪造特征补齐缺失输入。HOTA 等指标继续由独立的固定 TrackEval 入口计算。
