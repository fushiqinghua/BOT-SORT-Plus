# W R4.1 完整闭环评测

**已完成。** B 和 BG 各运行 off、W0、W1、W2、W3，十组均覆盖同一官方开发评测范围，共 30 段、12,590 帧。检测、常规 RGB ReID 与对应的 GMC 输入均沿用冻结协议。

[2026-11-28 完整结果与复现](full_20261128/README.md) · [十组汇总](full_20261128/report/REPORT.md) · [DJI_0073_3 与 DJI_0327_3 逐帧错因分析](full_20261128/frame_audit/ANALYSIS.md)

W0 相对各自关闭组提升约 0.17 个百分点 HOTA，同时增加 554／578 个误报。W1、W2 与 W0 的轨迹逐字节一致。W3 在两套底座上相对 W2 均增加 57 个误报、27 个漏检，换号持平。

[实验设计](../../../docs/botsort_plus/wem_r4_1/EXPERIMENT_PLAN.md) · [本地运行手册](../../../docs/botsort_plus/wem_r4_1/LOCAL_RUNBOOK.md) · [R3.1 冻结诊断](../wem_r3_1/full_20261128/README.md) · [R4 探索性闭环](../wem_v0.1/r4/README.md)
