# TCO T2

状态 `STOP_T2_INSUFFICIENT_HOLDOUT_INCREMENT`。

| Seed | Arm | Holdout upward | Holdout downward | Net | Adopted | Damage | Negative adoption |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 41 | P1 | 0 | 0 | 0 | 0 | 0.0 | 0.0 |
| 41 | P2 | 0 | 3 | -3 | 97 | 0.0008973975471133712 | 0.03225806451612903 |
| 41 | P3 | 0 | 2 | -2 | 66 | 0.0005982650314089142 | 0.0 |

检查点与采用门槛只由 calibration 确定。holdout 使用固定配置。
未知标签不参与正确率及预算计算。按来源组的结果及停止原因见 offline_metrics.json。
