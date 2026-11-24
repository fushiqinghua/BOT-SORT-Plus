# 实现来源、许可与论文表述

## 原生跟踪代码

接入流程改编自 [BoxMOT](https://github.com/mikel-brostrom/boxmot) 的 PyPI 16.0.11 wheel，沿用其 AGPL-3.0 许可。涉及 `BotSort.update` 阶段组织、STrack、KF、GMC、匹配与状态合并，均保留来源说明。补丁下 `botsort_plus/LICENSE` 提供许可证全文；新增扩展随此补丁按 AGPL-3.0 提供。安装和运行不会下载或替换用户模型权重。

`native_adapter.py::UPSTREAM_HASHES` 给出实际使用的关键文件校验值。该 wheel 的包内版本字符串和发行版本不一致，故使用源码哈希确认。主基线 NumPy 兼容修改涉及 OC-SORT 的 xysr KF，未把该修改错误计入本次 BoT-SORT 创新。

评估复用用户仓库的 `tools/evaluate_vttiny.py`、原 TrackEval commit 和 `provenance/trackeval.json`，没有另实现 HOTA/IDF1 公式。

## 研究结构参考

这些引用解释设计来源，不表示本补丁完整复现了对应算法；本文研究贡献可定位为在既有思想上针对微小目标弱观测、观测中断和多模态质量差异的具体机制改进。

- [SFTrack](https://arxiv.org/abs/2410.20079)：小而快目标、低分检测利用，参考其问题定位；本补丁采用时序候选确认与尺度几何。
- [TrackTrack](https://openaccess.thecvf.com/content/CVPR2025/html/Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.html)：轨迹视角关联与初始化；本补丁只使用冻结的 NMS 后检测，不声称使用被 NMS 丢弃的候选。
- [DualTTrack](https://doi.org/10.1016/j.ipm.2026.104843)：潜在轨迹和恢复结构，参考候选池组织。
- [BoostTrack++](https://arxiv.org/abs/2408.13003)：检测可靠性、几何线索和未更新时间处理，作为对照参考。
- [UTrack](https://arxiv.org/abs/2408.17098)：不确定性建模；本补丁仅使用原生 KF 投影协方差，不从 NMS 后单框虚构检测分布方差。
- [ARC-SORT](https://arxiv.org/abs/2610.22306)：自适应关联与身份处理，作为近期结构参考；本补丁没有其全部组件，也不包含离线轨迹合并。
- [HGT-Track / VT-Tiny-MOT](https://arxiv.org/abs/2412.10861)：双模态微小目标跟踪问题；本补丁保持当前 RGB 坐标、冻结检测口径。
- [MCTrack](https://github.com/ydhcg-BoBo/MCTrack)：跨模态与时间特征交互，参考其互补性动机；本补丁实现的是目标级外观距离和可信记忆。

## 版本状态

补丁生成日期：2026-11-24。初始参数属于待验证假设；软件测试通过不能支持“领先”“提升若干点”或“完成 thermal 适配训练”的表述。论文中应分别报告：借鉴的机制、我们改变的决策规则、固定输入的消融、真实指标与失败例。
