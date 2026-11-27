# TCO 本地研究入口

本目录是下一批实验的索引。**采样器、网络、训练器及闭环适配器尚未实现。** 本次提交只提供完整设计、机器可读协议与准备工具。

1. 阅读[模块设计](../../docs/botsort_plus/tco_v0.1/MODULE_DESIGN.md)。
2. 按[实验协议](../../docs/botsort_plus/tco_v0.1/EXPERIMENT_PLAN.md)实现并执行 T0/T1。
3. 使用[数据接口](../../docs/botsort_plus/tco_v0.1/DATA_CONTRACT.md)分离预测输入和离线标签。
4. 按[本地交接](../../docs/botsort_plus/tco_v0.1/LOCAL_RUNBOOK.md)完成最小可运行工作包，再根据证据推进 T2/T3。

现有命令为 `python -X utf8 tools/manage_tco_batch.py --help`。不要将设计文档中的拟产物路径当作已存在结果，也不要用空模型或未训练权重替代研究实现。
