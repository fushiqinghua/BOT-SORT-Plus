"""Assemble the episode audit and counterfactual results into the local experiment report."""
import csv
from workflow import ROOT


LABELS={'fixed':'长间隔重连修正版','combo':'修正版＋④',
        'probe_weight':'撤回指定轨迹的降权','probe_B':'阻断一次指定 B 接回','output_only':'仅替换当帧输出框',
        'velocity':'位置与尺寸降权，速度用原生更新','weight_floor':'降权系数至少 0.5','b_guard':'收紧受降权轨迹的低分 B 接回',
        'native':'原始 BoT-SORT','evidence_both':'长间隔重连修正版',
        'evidence_both_bam_overlap':'修正版＋④','full_velocity':'修正版＋④，速度用原生更新'}


def load(path):
    with path.open(encoding='utf-8-sig',newline='') as handle:
        return list(csv.DictReader(handle))


def table(head,body):
    def line(fields):return '| '+' | '.join(map(str,fields))+' |'
    return '\n'.join([line(head),line(['---']+['---:']*(len(head)-1)),*(line(r) for r in body)])


def main():
    out=ROOT/'results'
    cases=load(out/'case_metrics.csv')
    counts=('FP','FN','IDSW')
    def total(method):
        rows=[r for r in cases if r['method']==method]
        assert len(rows)==3,method
        return [sum(int(r[k]) for r in rows) for k in counts]
    base=total('combo')
    focus_rows=[]
    for m in ('fixed','combo','probe_weight','probe_B','output_only','velocity','weight_floor','b_guard'):
        tt=total(m)
        focus_rows.append((LABELS[m],*tt,*(f'{a-b:+d}' for a,b in zip(tt,base))))
    hist=load(out/'full_velocity/tracking.csv')
    pooled={r['tracker']:r for r in hist if r['sequence']=='COMBINED_SEQ'}
    fields=('HOTA','IDF1','MOTA',*counts)
    ms=('native','evidence_both','evidence_both_bam_overlap','full_velocity')
    def metric(k,val,sign=False):
        return format(float(val),('+' if sign else '')+('.0f' if k in counts else '.4f'))
    display=[(LABELS[m],*(metric(k,pooled[m][k]) for k in fields)) for m in ms]
    display.append(('速度方案 − 原组合',*(metric(k,float(pooled['full_velocity'][k])-float(pooled['evidence_both_bam_overlap'][k]),True) for k in fields)))
    s=f"""# 修正版＋④的三段新增误报诊断

在完整 30 段实验中，加入④后净增 71 个误报。其中 DJI_0028_3、DJI_0061_2 和 DJI_0169_3 合计净增 60 个。对照冻结检测行和官方 RGB 真值，这三段一共新增 230 条、消失 170 条误报。回放所得的两组对照轨迹文件与此前全量实验的对应序列逐字节一致。

## 误报来自哪里

| 序列 | 修正版 FP → 修正版＋④ FP | 新增 → 消失 | B 接回连续片段新增 / 消失 | 原生关联连续片段新增 / 消失 |
| --- | ---: | ---: | ---: | ---: |
| DJI_0028_3 | 531 → 557 | 32 / 6 | 32 / 3 | 0 / 3 |
| DJI_0061_2 | 2,301 → 2,318 | 169 / 152 | 114 / 79 | 55 / 73 |
| DJI_0169_3 | 1,504 → 1,521 | 29 / 12 | 3 / 6 | 26 / 6 |
| 合计 | 4,336 → 4,396 | 230 / 170 | 149 / 88 | 81 / 82 |

“B 接回连续片段”从 B 接回当帧计到该 ID 首次没有输出或进入下一次 B 恢复。在此期间，高低分关联接住的后续帧也计入片段。这个表按直接输出路径计数，定位因果还需要下面的回放干预。在 230 条新增误报中，180 条所用原始检测与任何真值的 IoU 均低于 0.5。另有 47 条的原始检测达到 0.5，但跟踪输出框低于 0.5，还有 3 条在多候选的官方匹配中未获得真值。仅更换当前降权帧的输出框、保持后续关联相同，反而净增 9 个误报，说明这轮误报无法靠替换降权帧的输出框解决。逐检测行和连续片段可由冻结输入重放生成。

## 三段逐帧回放

### DJI_0028_3

ID 58 在第 273、274 帧接受相同低分检测。④的权重分别约为 0.546、0.368。修正版继续在第 275–277 帧续接该轨迹，组合在这三帧没有输出 ID 58。第 282 帧，两条候选观测通过 B 的提前恢复，检测及输出框与最近真值的 IoU 约为 0.420 和 0.440，官方评分计为误报。此后 ID 58 反复中断及被接回，累计贡献 32 条新增误报中的 29 条。

只撤回 ID 58 第 273、274 帧的降权，整段 FP 557→531，FN 979→982，换号仍为 25。保留④且只阻断第 282 帧的指定 B 接回，FP 557→529，FN 979→982，换号仍为 25。这两次干预分别切断了后续误报的上游状态变化及首次 B 恢复。两种干预都使一个原先消失的换号位置重新出现，换号总数仍持平。[时间线](typical_track_timelines.csv) · [干预明细](intervention_effects.csv)

### DJI_0061_2

该段没有选用 LightGlue，两个版本的相机矩阵也一致。组合中的 ID 149 在第 125 帧接受低分检测，权重约为 0.481，第 136 帧由 B 接回。该帧的检测与最近真值的 IoU 仅 0.132。这次接回至第 176 帧的连续输出产生 25 条新增误报。修正版同帧的 ID 149 已走另一条高分关联路径。

只阻断第 136 帧的指定接回，ID 149 第 137 帧改由 B 接回，使用与原组合第 137 帧续接相同的检测，整段只少 1 个误报。撤回 ID 149 在第 125 帧及以前的所有降权，整段 FP 2,318→2,280，FN 17,233→17,237，换号仍为 175。第 137 帧的重新接回解释了单次阻断的有限效果。+17 个净误报对应 169 条新增和 152 条消失，B 和原生路径上的净误报方向相反。[时间线](typical_track_timelines.csv) · [干预明细](intervention_effects.csv)

### DJI_0169_3

ID 15 在第 136、137 帧继续匹配低分检测，两次原始检测与真值最大 IoU 都约为 0.108。第 147 帧曾由 B 接回。第 174、191 帧各由原生高分关联启动一个输出片段，两段分别产生 13、9 条新增误报，合计占该段 29 条新增误报中的 22 条。

只阻断第 147 帧指定 B 接回，FP 1,521→1,520，FN 和换号持平。撤回 ID 15 在第 137 帧及以前的降权，FP 1,521→1,510，FN 2,903→2,905，换号仍为 10。后者移除了 14 条组合误报，但只有 2 条属于最初相对修正版新增的 29 条，同时又产生 3 条误报。整段误报下降来自多处轨迹变化，单点 B 阻断并未解决主要的两段原生关联误报。[时间线](typical_track_timelines.csv) · [干预明细](intervention_effects.csv)

## 能否在保住换号收益时减少误报

{table(['三段合计','FP','FN','IDSW','ΔFP','ΔFN','ΔIDSW'],focus_rows)}

差值以“修正版＋④”为参照。“撤回指定轨迹的降权”和“阻断一次指定 B 接回”使用诊断选定的事件。保留原生速度的方案不读取序列、帧或轨迹 ID 规则。它在三段上净减 20 个误报、4 个漏检和 2 次换号。DJI_0061_2 本身的误报仍增加 9 个，逐帧核对有 12 个曾被④消除的换号位置重新出现，说明全局净次数改善伴随个别换号回归。更严格的 B 接回压低了较多误报，也增加了 84 个漏检。[逐段结果](case_metrics.csv) · [换号位置回归](intervention_effects.csv)

## 冻结参数的完整 30 段复核

{table(['方法','HOTA','IDF1','MOTA','FP','FN','IDSW'],display)}

速度方案在焦点三段少 20 个误报，在其余 27 段多 24 个，最终全量多 4 个误报。其中 DJI_0283_1 多 12 个、DJI_0327_3 多 8 个、DJI_0073_3 多 5 个、DJI_0257_2 少 1 个。全量漏检多 2 个，换号多 1 次，IDF1 下降 0.0832 个百分点。原组合仍是本轮对照中完整 30 段的取舍。

复核使用固定检测、RGB ReID 与原始 RGB 帧。三段速度方案输出与完整运行的相应输出逐字节一致，30 段和原组合的 GMC 矩阵一致。测试集此前参与选模和调参，本轮按开发性消融记录。[全部逐序列结果](full_velocity/tracking.csv) · [全量差值与评估记录](full_velocity/metrics/comparison.csv) · [复现命令](../README.md)
"""
    (out/'实验报告.md').write_bytes(s.encode('utf-8'))


if __name__=='__main__':main()
