# W R4 复现

在该目录运行以下命令。脚本复用相邻 R3 目录中的 SSH 桥接、状态采样、评分和几何函数。归档的 R0、R3 与原版 BoT-SORT 代码或输入发生漂移时，R4 核验会失败。[R4 结果](../../../results/vttiny/wem_v0.1/r4/README.md)

## 配置和服务器归档

本地需 Python 3.11 和 OpenSSH。SSH 使用已有主机别名、账户和密钥配置。设置 `VT_WEM_SSH_TARGET`、`VT_WEM_REVIEW_PROJECT_ROOT`、`VT_WEM_REMOTE_PYTHON`。后两项分别指服务器项目绝对路径和运行时 Python。R4 复用带 Torch、CUDA、BoxMOT、OpenCV、SciPy 和固定版 TrackEval 的 R3 实验环境。

预测读取 `integrations/vttiny_wem_r01_v1/observer` 的冻结设置和 R0 核验，`integrations/vttiny_wem_r3_v1` 的原始 R3 锁，以及 `integrations/botsort_plus_v0_4_motion_blur` 的追踪器、相机补偿缓存与 TCO 参考。真值只在预测完成后由评测进程按冻结清单读取。

```bash
python -B transfer.py fetch smoke
python -B transfer.py fetch pilot
python -B transfer.py fetch full
python -B summarize.py pilot
python -B summarize.py full
```

命令从既有 `integrations/vttiny_wem_r4_v1` 取回实际核验、评测和逐帧事件。取回的事件含实验环境原始身份键，请在本地使用。仓库结果目录提供衍生公开表和完整指标。`summarize.py` 核对 TrackEval 官方聚合指标、逐段评测及逐帧归因的 FN、FP 和换号计数，再输出配对结果。

## 重跑顺序

在同布局的独立项目目录备齐冻结资产。运行脚本默认使用物理 GPU 4，可通过 `--gpu` 指定空闲卡。后台任务每阶段结束后会写出 `complete.json`，使用 `status` 验收后再启动下一阶段。

```bash
python -B transfer.py run smoke --gpu 4
python -B transfer.py status smoke
python -B transfer.py run pilot --gpu 4
python -B transfer.py status pilot
python -B transfer.py run full --gpu 4
python -B transfer.py status full
```

`smoke` 比较 B 与 `evidence_both` 的关闭、影子和强制拒绝。`pilot` 使用同样三段，`full` 使用冻结的 30 段。两阶段对照包含两套参考、B 加原生低阈值、旁路几何、Coast-only、W-coast、完整 W，以及 `evidence_both` 加完整 W。研究用运行器将线上启用限制在独立回放目录，R3 冻结默认策略继续拒绝全部弱框。
