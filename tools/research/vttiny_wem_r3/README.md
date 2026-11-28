# VT-Tiny-MOT W R3 复现

此目录含 R3 训练侧回放、离线真值标签、冻结 RGB 特征提取、准入比较与逐帧图卡脚本。运行依赖已归档的 VT-Tiny-MOT 官方训练数据及原始 RGB-T 检测缓存、RGB ReID 缓存、固定追踪器和评估运行时。对应的原始 SHA-256 记录在 R3 代码和已发布来源凭据中。用于程序校验的 [预登记原文](R3_PROTOCOL.md) 保留原字节，它引用旧工作区中的冻结材料路径。

## 配置

在本目录运行。终端需有 Python 3.11 与 OpenSSH。图卡使用 Pillow。SSH 使用自己的主机别名或账户及现有密钥配置。环境变量 `VT_WEM_SSH_TARGET` 为 SSH 目标，`VT_WEM_REVIEW_PROJECT_ROOT` 为服务器 VT-Tiny-MOT 项目绝对目录，`VT_WEM_REMOTE_PYTHON` 为归档运行时的 Python 路径。服务器使用已有 NumPy、SciPy、Torch、OpenCV、BoxMOT、TrackEval 和固定的 `BOT-SORT-Plus-followup-53faeccb` 工作树。

现有服务器归档以项目目录为根，`integrations/vttiny_wem_r3_v1` 存放 R3 产物，`integrations/botsort_plus_v0_4_motion_blur/tco_53faeccb/t23_v1` 存放冻结训练划分、训练输入、官方训练 GT 转换和历史 B 回放。`data/raw/VT-Tiny-MOT` 和 `reports/image_hashes.jsonl` 用于重建图卡。检测、权重或历史协议摘要出现漂移时，脚本会停止。完整回放和特征提取需要已授权的服务器与对应数据，发布的摘要可离线查阅。[R3 结果归档](../../../results/vttiny/wem_v0.1/r3/README.md)

## 从现有服务器归档复现图卡与索引

```bash
python observer/transfer.py fetch full
python labels/transfer.py fetch full --detail
python transfer.py fetch full
python review/select_candidates.py
python review/collect_frames.py
python review/build_cards.py
python review/summarize_review.py
```

前三条下载原始运行凭据与标签到本地生成目录。后续依次筛选 113 例、按原始哈希提取成对图像、渲染和汇总。最后一步读取仓库中公开的逐例模型辅助判断。人工表独立生成并保留已有编辑。`python inspect_gates.py` 可读取服务器归档并在本地生成逐门拆解。[图卡说明](review/README.md)

## 在具备冻结资产的环境中重跑 R3

启动命令提交后台任务。例行先运行 `smoke`，通过 `status` 核对结束后再运行 `full`，然后依次进入后续阶段。完整重跑使用已备齐相同冻结资产的独立项目根目录。现有项目上的归档可直接 `fetch`。

| 阶段 | 启动命令 | 核对命令 |
| --- | --- | --- |
| 只读训练侧回放 | `python observer/transfer.py deploy` 后执行 `python observer/transfer.py run full` | `python observer/transfer.py status full` |
| 离线训练 GT 候选标记 | `python labels/transfer.py launch full` | `python labels/transfer.py status full` |
| 冻结 RGB 编码器提取 S0 特征 | `python transfer.py features full --device 4` | `python transfer.py status full` |
| 校准网格和留出诊断 | `python transfer.py score full` | `python transfer.py status full` |

冻结样本、门槛和拒绝全部策略按原预登记执行。代码不包含在线 W 或 R4 入口。新跑的压缩和运行回执可能因环境及时间戳变化而产生新摘要。归档复现以服务器原凭据逐字节核对。
