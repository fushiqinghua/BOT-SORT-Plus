# W v0.1 分支准备与软件核验

## 本次完成

- 从 176f2a81613c34d16a860765ea7be0845116b534 建立 W 研究分支。
- 归档设计、R0/R1a/R1b/R2/R3/R4 计划、I0–I8 本地实现任务和结果模板。
- 更新根 README，明确 TCO/Q2 最新结论与本分支实际实现范围。
- 提供标准库准备工具及 S0 缓存库存审计。
- 主跟踪代码、B/GMC 参数和历史实验产物未修改。

## 软件测试

在本次准备环境执行：

~~~bash
python3 -m unittest discover -s tests -p 'test_wem*.py' -v
~~~

**26 项通过：准备工具 13 项，库存审计 13 项。**

覆盖有实际解释风险的行为：

- 原生 float32 → float64 分数边界；high 门槛遗漏与 S0 分离；
- 原始行索引、空帧、帧缺口、重复帧及非法框；
- 输入发生变化、不完整批次和非空结果目录的拒绝；
- 模板缺少资产、配置/协议变化、快照损坏，不被标为实验通过；
- 仓库相对路径、源目录与输入资产保护；
- 中文、空格和撇号路径；Windows PowerShell 命令中的美元符和反引号按字面值生成。

Windows 检查是路径/命令生成合同测试，没有在本环境实际启动 PowerShell。

## README 命令串联

另以明确标识为 toy_sequence 的构造缓存执行真实 CLI：

1. 在 configs/research 中创建新的本地配置；
2. 缺少检测路径时 check --strict 正确返回非零；
3. 填入含中文/空格的临时路径后检查通过；
4. prepare 写入 PREPARED_NOT_RUN；
5. 运行其打印参数对应的 audit_wem_s0.py；
6. 审计输出保持 INVENTORY_ONLY_NOT_W_OPPORTUNITY；
7. status 只报告结果文件存在，科研验收仍为 NOT_EVALUATED。

该串联检验使用构造数据，目的为确认命令可用、边界与状态语义正确。没有使用构造数据冒充真实实验。

## 尚未验证

- 用户本地缓存、权重、图像及参考运行来源；
- 新 W 采样器关闭/开启的真实轨迹、事件、KF 一致性；
- R1b 身份与所有权机会；
- R2 当前量测增量；
- R3 准入工作点与独立留出；
- 在线 W、Coast-only、W-coast；
- 完整数据集 HOTA、IDF1、FP、FN、IDSW；
- 新增收益或第三个有效创新点。

软件测试通过仅证明本次准备/库存工具的已测试行为。真实数据实验由本地按 LOCAL_RUNBOOK.md 和 EXPERIMENT_PLAN.md 执行。
