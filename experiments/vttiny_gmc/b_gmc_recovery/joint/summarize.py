import csv
import json
from pathlib import Path

root = Path(__file__).resolve().parent / "results"
read = lambda path: json.loads(path.read_bytes())
items = read(root/"full/metrics/summary.json")["results"]
records = {(row["tracker"],row["sequence"]):row for row in items}
base, b, g, both = "native","B_early_moderate","dual_low","B_early_moderate_dual_low"
fields = ["HOTA","DetA","AssA","IDF1","MOTA","FP","FN","IDSW"]
sequences = sorted({r["sequence"] for r in items} - {"COMBINED_SEQ"})
assert len(records)==124 and len(sequences)==30
def values(row):
    return " | ".join(f"{row[k]:.4f}" if k not in fields[-3:] else str(row[k]) for k in fields)
def delta(x,y,seq):
    return {k:records[x,seq][k]-records[y,seq][k] for k in fields}
def signed(row):
    return " | ".join(f"{row[k]:+.4f}" if k not in fields[-3:] else f"{row[k]:+d}" for k in fields)
def tbl():
    return ["| 方法或对照 | "+" | ".join(fields)+" |","|---|"+"---:|"*len(fields)]
seq="COMBINED_SEQ"
lines=["# 原始 BoT-SORT、B 与自适应 GMC 的四组消融","",
       "VT-Tiny-MOT 官方 30 段，12,590 帧。使用原版卡尔曼滤波、相同的 YOLO11-M RGB–T 检测、RGB ReID 缓存和 RGB 图像。官方 RGB GT 合并类别，使用 TrackEval COMBINED_SEQ。官方测试集曾用于选模和规则设计。","",*tbl()]
for m in (base,b,g,both): lines.append(f"| {m} | {values(records[m,seq])} |")
lines.extend(["","## 改进量","","百分比指标的差值以百分点计，最后三列为事件数的增减。","",*tbl()])
for x,y in [(b,base),(g,base),(both,base),(both,b),(both,g)]:
    lines.append(f"| {x} − {y} | {signed(delta(x,y,seq))} |")
effect={k:delta(both,b,seq)[k]-delta(g,base,seq)[k] for k in fields}
lines.append(f"| 交互差值（组合 − B − GMC ＋ 原始） | {signed(effect)} |")
stats=read(root/"joint_result.json")["actions"]
lines+=["","## 接入与校验","",
       "组合与单 B 在服务器重新回放。原始和单 GMC 复用已核验的全量轨迹，四组一起重新评分。协议逐序列比对检测指纹、特征数组、元数据和解码像素。单 B 的 30 份轨迹逐字节复现历史参考。",
       f"B 使用 v2 适配器的低分关联流程。选用 LightGlue 且其矩阵不同于备用变换的 {stats['lightglue_frames']} 帧内，在 v2 执行低分分配前应用备用变换的 IoU 0.5 门限。累计屏蔽 {stats['blocked_native_edges']} 条原本达到原生门限的候选边。候选边计数发生在分配前。高分中心修正输出数为 {stats['center_outputs']}。"]
for m in (b,both):
    files=list((root/"full/timing"/m).glob("*.json"))
    assert len(files)==30
    events=sum(read(p)["event_counts"].get("recovered",0) for p in files)
    lines.append(f"- {m} 实际触发 {events} 次 B 恢复事件。")
lines+=["","## 逐序列结果","","[四组逐序列绝对指标及全量汇总](full/metrics/comparison.csv) · [组合相对单 B 及单 GMC 的差值](sequence_deltas.csv)"]
for ref in (b,g):
    differences = sorted((delta(both,ref,s)["HOTA"],s) for s in sequences)
    wins = sum(v>1e-9 for v,_ in differences)
    losses = sum(v< -1e-9 for v,_ in differences)
    low, high = differences[0], differences[-1]
    lines.extend(["",f"相对 {ref}，组合在 {wins} 段 HOTA 提升、{losses} 段下降、{30-wins-losses} 段持平。"
                  f"最大提升为 {high[1]}（{high[0]:+.4f}），最低差值为 {low[1]}（{low[0]:+.4f}）。"])
(root/"实验报告.md").write_text("\n".join(lines)+"\n",encoding="utf8")
with (root/"sequence_deltas.csv").open("w",encoding="utf-8-sig",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["sequence","reference",*fields]); w.writeheader()
    for seq in sequences:
        for ref in (b,g): w.writerow({"sequence":seq,"reference":ref,**delta(both,ref,seq)})
print("\n".join(lines))
