# 本分支的软件验证记录

验证日期：2026-11-25。代码从冻结 `baseline` 提交 `70f28b112ba34014b851152fbeedf007dd7cbb52` 建立。

## 初次提交的软件验证

### 1. 冻结检测缓存检查

使用新 CLI 的 `doctor` 验证仓库 RGB–IR 缓存：

| 项目 | 结果 |
|---|---:|
| 序列 | 30 |
| 帧 | 12,590 |
| 检测框 | 133,386 |
| 空检测帧 | 1,100 |
| 检测器权重 SHA256 | `59ecddb827adb8b6ed75d78b644cc09e8fcdc9c83870cd1f8a9446ad679dc3ec` |

检查了连续帧编号、有限值、框宽高、置信度范围以及检测器指纹一致。此步骤没有读取真实图像或 GT。

### 2. 55 项自动测试全部通过

使用独立 CPU 验证环境运行：

```bash
M2_TEST_BOXMOT=1 TRACKEVAL_ROOT=/path/to/pinned/TrackEval \
  python -m unittest discover -s tests -v
```

| 测试文件 | 数量 | 覆盖重点 |
|---|---:|---|
| `test_calibration.py` | 11 | 录像隔离、非重叠小框采样、歧义排除、非负拟合、真实 I/O 契约 |
| `test_io.py` | 6 | 压缩缓存、空检测帧、GT 合并口径、非法输入、路径解析与分区边界 |
| `test_m2_core.py` | 19 | R 公式、Joseph 更新、NLL、全局拒配、物理门控、真实 BoxMOT 生命周期与同一 R 更新 |
| `test_diagnostics.py` | 10 | 已知平移、无纹理失败、特征 SHA / 索引 / 零向量、连续低分与输出中断 |
| `test_reporting.py` | 8 | 同输入比较、SOF 基准选择、GT / 图像 / 序列不一致拒绝、指标来源与输出保护 |
| `test_cli_integration.py` | 1 | 完整命令链、四种变体、空 GT 序列、错误校准来源拒绝、TrackEval 与汇总 |

关键数值验证包括：虚拟节点 Hungarian 与穷举最优解一致；无重叠微小框可被合法关联；logdet 惩罚扩散轨迹；原生工厂与直接 BoxMOT 实例的逐帧输出完全一致；低分第二阶段实际后验与独立计算的同一 R / Joseph 更新吻合。

CLI 测试实际生成了三段各 24 帧的合成图像、检测、GT 与特征缓存，以独立子进程执行预检查、划分、校准、四组回放、M1、M3、固定 TrackEval 和结果汇总。包含一个空 GT 序列，并检查其 FP 被计入。合成输出只用于软件检查，未作为算法实验成绩归档。

### 3. 实际加载的验证依赖

- Python 3.12.14；NumPy 2.3.5；SciPy 1.17.0。
- PyTorch 2.8.0+cpu；torchvision 0.23.0+cpu；OpenCV 5.0.0。
- BoxMOT 分发版本 16.0.11；包内版本字符串 16.0.10。四个已审阅核心文件源码全部匹配 v16.0.11。
- TrackEval `12c8791b303e0a0b50f753af204249e622d0281a`；全部 44 个归档文件经纯 LF / CRLF 差异核对后通过。

上面的记录来自独立 CPU 软件验证环境。下节的真实数据回放使用原服务器的 Python 3.11 / CUDA PyTorch 环境，在 CPU 上运行跟踪器。

## 首轮真实数据验证

在原服务器环境复跑了 55 项软件测试，并用原始图像和冻结检测缓存逐帧复现了 30 段无 ReID 基线。随后在预先固定的 9 段上拟合噪声，用同一批 21 段评估 M2 四组和 M1 的 SOF 关闭对照，并完成 M3 的候选覆盖与 RGB 特征诊断。 [首轮实验报告](results/round1_2026-11-25/README.md)

以上真实数据结果采用 VT-Tiny-MOT 七类合并的回顾性开发评估口径。划分、源码与输入指纹见 [首轮复现清单](results/round1_2026-11-25/reproducibility.json)。
