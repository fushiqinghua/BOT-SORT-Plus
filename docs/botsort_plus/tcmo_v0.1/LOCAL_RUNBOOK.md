# TCMO v0.1 本地执行交接

本次提交只有设计、配置模板与任务书；**尚无 tools/run_tcmo_v01.py，也没有 GPU/真实数据运行**。不要把下述拟议命令当作已实现。按 [IMPLEMENTATION_TASKS.md](IMPLEMENTATION_TASKS.md) 先实现并验证，之后将实际命令与测试结果写入本文件。新工作只写本分支。

从已有 repo 根目录：
```bash
git fetch origin
git switch --track origin/research/vttiny-multimodal-observation
```
若本地分支已存在，使用 git switch research/vttiny-multimodal-observation，再 git pull --ff-only origin research/vttiny-multimodal-observation。不要在 thesis-finalization 或 weak-evidence checkout 中实现。继续使用旧 R4.1 成功的环境（BoxMOT 16.0.11、固定 TrackEval）。复制 `configs/research/tcmo_v0.1.example.json` 到本地私有配置路径，填写真实 R4.1 settings 和完整 M0 运行目录、RGB/IR 原图/GT、训练划分、辅助权重/训练日志或候选缓存及声明映射。GT 仅离线，绝不传入预测 child。相对路径解析必须明确，输出放新目录，原资产只读。权重、图像、GT、配置、缓存不提交 GitHub。用配置模板只是登记资产，不代表可以执行新 tracker。
第一阶段按 P0/P1 生成 inventory、校准锁、辅助 cache；若缺资产，先回传清单，说明是否需训练。然后实现 P2 和测试，固定 smoke 后 R0 全量只读和 R1 六组完整评测。建议实现后提供 init/check/export/calibrate/smoke/r0/full/evaluate/summarize CLI；不要提前承诺命令已存在。提交本地实现版本、合成与真实 smoke 日志、固定协议锁，再启动全量。不因数据指标不好中止。
完成后按 [结果回传规范](RESULTS_TEMPLATE.md) 上传派生数据；目前不必实现 R2 或额外 IR ReID。
