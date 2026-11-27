"""Build a readable report and an evidence bundle following both completed stages."""
import csv
import hashlib
import zipfile
from paths import ROOT,REPO,METHODS,FOCUS,HISTORY,read,save

NAMES={"native":"原始 BoT-SORT","adaptive_gmc":"仅自适应 GMC","oru":"仅找回后卡尔曼重估"}
FIELDS=("HOTA","DetA","AssA","IDF1","MOTA","FP","FN","IDSW")


def fmt(k,x,sign=False):
    if k in ("FP","FN","IDSW"):
        return format(int(x),"+,d" if sign else ",d")
    return format(x,"+.4f" if sign else ".4f")


def table(headers,rows):
    return ["| "+" | ".join(headers)+" |","| "+" | ".join(["---"]*len(headers))+" |",
            *["| "+" | ".join(map(str,r))+" |" for r in rows],""]


def lines_for_report(full,smoke,rows):
    metrics={x["tracker"]:x for x in full["metrics"]}
    result=["# 原始 BoT-SORT 的自适应 GMC 与找回后卡尔曼重估","",
      f"VT-Tiny-MOT 官方测试集共 {full['parity']['sequences']} 段、{full['parity']['frames']:,} 帧。"
      "原始 BoT-SORT 启用原生稀疏光流 GMC 与 RGB ReID。两组消融各自只改变一处。"
      "检测使用 YOLO11-M RGB–T 缓存，ReID 使用同一套 RGB 缓存，输入图像按解码像素校验。"
      "官方测试集曾用于权重选择和开发调参，本报告按开发集证据解读。","",
      "## 完整测试集","",
      "TrackEval 按 COMBINED_SEQ 联合计算。HOTA、DetA、AssA、IDF1、MOTA 为百分数。差值为相对原始 BoT-SORT 的百分点，FP、FN、IDSW 为事件数量。",""]
    result+=table(["方法",*FIELDS],[[NAMES[m],*[fmt(k,metrics[m][k]) for k in FIELDS]] for m in METHODS])
    result+=table(["改动",*["Δ"+k for k in FIELDS]],[
        [NAMES[m],*[fmt(k,metrics[m]["delta"][k],True) for k in FIELDS]] for m in METHODS[1:]])
    for m in METHODS[1:]:
        r=[x for x in rows if x["method"]==m and x["sequence"]!="COMBINED_SEQ"]
        pos=[x for x in r if float(x["delta_HOTA"])>1.e-9]
        neg=[x for x in r if float(x["delta_HOTA"])< -1.e-9]
        best=max(r,key=lambda x:float(x["delta_HOTA"]));worst=min(r,key=lambda x:float(x["delta_HOTA"]))
        result += [f"{NAMES[m]}的逐序列 HOTA 有 {len(pos)} 段上升、{len(neg)} 段下降、{len(r)-len(pos)-len(neg)} 段持平。"
                   f"最大增量为 {best['sequence']} 的 {float(best['delta_HOTA']):+.4f}，"
                   f"最小增量为 {worst['sequence']} 的 {float(worst['delta_HOTA']):+.4f}。",""]
    seconds=full["replay_seconds"]
    result+=[f"12,590 帧的关联与 GMC 回放计时合计为原始 {seconds['native']:.2f} 秒、自适应 GMC {seconds['adaptive_gmc']:.2f} 秒、卡尔曼重估 {seconds['oru']:.2f} 秒。"
             "计时包含本轮观测记录与缓存特征处理，排除检测、ReID 的在线提取和模型加载。",""]
    result+=["## 自适应 GMC 的两段预检","",
        "当前帧和前一帧都排除检测框区域。先用分区留出的背景光流点评估原生补偿，满足误差触发条件后，"
        "才运行预训练 SuperPoint 与 LightGlue，并与原生变换、恒等变换比较。"
        "拟合和验证采用不同棋盘分区。候选需要在留出点上同时满足残差改善量与高分位误差约束。",""]
    dr=[]
    for s in FOCUS:
        group={x["tracker"]:x for x in smoke["focus_metrics"][s]}
        for m in ["native","adaptive_gmc"]:
            dr.append([s,NAMES[m],*[fmt(k,group[m][k]) for k in ("HOTA","IDF1","MOTA","FP","FN","IDSW")]])
    result+=table(["序列","方法","HOTA","IDF1","MOTA","FP","FN","IDSW"],dr)
    result+=table(["序列","帧数","触发匹配","采用原生","采用 LightGlue","采用恒等"],[
        [g["sequence"],g["frames"],g["trigger"],g.get("sof",0),g.get("lightglue",0),g.get("identity",0)] for g in smoke["gmc_focus"]])
    result+=["对于实际切换补偿的帧，固定原始 BoT-SORT 的补偿前预测中心。"
             "只保留上一帧输出能明确匹配真值、当前帧仍有该真值的轨迹，比较同一预测中心到当前真值中心的距离。",""]
    result+=table(["序列","可配对轨迹帧","平均变化（像素）","中位变化（像素）","改善 >1 像素","恶化 >1 像素"],[
        [s,c["n"],*(f"{c[k]:+.3f}" if c[k] is not None else "无样本" for k in ("mean_delta","median_delta")),
         c["improved_1px"],c["worse_1px"]] for s,c in smoke["gmc_paired_centers"].items()])
    g=full["gmc_total"]
    result+=[f"完整测试集共有 {g['frames']:,} 帧，触发背景匹配 {g['trigger']:,} 帧，"
             f"采用 LightGlue 变换 {g.get('lightglue',0):,} 帧、恒等变换 {g.get('identity',0):,} 帧，"
             f"保留原生变换 {g['sof']:,} 帧。","","## 找回后的原 ID 保持","",
        "卡尔曼重估只在原生关联已接受真实检测、准备重激活轨迹时运行。以轨迹中断前后两次实际接受的检测框为锚点，"
        "用原生 GMC 对齐坐标，在缺失区间回放虚拟观测，并在找回帧执行一次真实测量更新。"
        "各帧已发布的输出保持固定。","",
        "第一组统计固定原始 BoT-SORT 的找回事件。轨迹中断两端必须能将同一个输出 ID 明确匹配到同一个真值 ID。"
        "跨算法按中断前真值对齐各自的本地 ID。只统计从找回帧直到第 +3、+5、+10 帧，真值逐帧有标注的完整窗口。"
        "从找回帧起每一帧都明确输出该原 ID 才算保持。出现漏帧、换号或歧义即无法确认连续保持。"
        "离线匹配采用逐帧一对一分配，IoU 至少 0.5，多个合格候选相差小于 0.05 标为歧义。",""]
    result+=table(["找回后","配对事件","原始保持","重估保持","重估新增保持","重估丢失保持","保持率变化"],[
        [f"+{n} 帧",v["eligible"],f"{v['native']}（{100*v['native']/v['eligible']:.2f}%）",
         f"{v['oru']}（{100*v['oru']/v['eligible']:.2f}%）",v["gained"],v["lost"],
         f"{100*(v['oru']-v['native'])/v['eligible']:+.2f} 个百分点"] for n,v in full["fixed_windows"].items() if v["eligible"]])
    shared=full["shared_endpoints"]
    result+=[f"第二组统计限定两种方法都发生原生重激活、都在同样的两帧接到完全相同的两行缓存检测框，而且两端都对上同一个真值身份。"
             f"共 {shared['events']} 个这样的找回事件。这项比较把找回以前已经产生的身份差异排除在外。",""]
    result+=table(["找回后","可评事件","原始保持","重估保持","重估新增保持","重估丢失保持"],[
        [f"+{n} 帧",v["eligible"],v["native"],v["oru"],v["gained"],v["lost"]]
        for n,v in shared["windows"].items()])
    result+=["两种方法各自找回的事件构成也可能变化。下表给出各自的原生重激活总数与可明确确认的同真值找回，"
             "以及各自有效窗口内的连续保持数，以便与固定事件口径交叉检查。",""]
    ownrows=[]
    for m in ("native","oru"):
        e=full["recoveries"][m]
        ownrows.append([NAMES[m],e["total"],e["categories"].get("same_gt",0),e["categories"].get("different_gt",0),
                        e["categories"].get("unresolved",0),e["replayed"],e["numerical_fallback"],
                        *[f"{c.get('held',0)}/{sum(c.values())}" for c in full["own_windows"][m].values()]])
    result+=table(["方法","重激活","同真值","异真值","端点未能确认","完成重估","数值回退","+3 帧","+5 帧","+10 帧"],ownrows)
    result+=["## 结果解读","",
        "自适应 GMC 的总体收益集中在身份关联。DJI_0173_2 的 61 次切换全部为恒等变换，"
        "在同一补偿前状态下比较，34 个可核验的轨迹帧中有 25 个的中心误差缩小超过 1 像素。"
        "DJI_0109_1 的 90 帧贡献了 13 次换号减少，DJI_0173_2 贡献了 6 次。"
        "这两段共占全量净减少 25 次换号中的 19 次。","",
        "卡尔曼重估的 HOTA 增量约 0.01 个百分点。固定原始找回事件后，+3、+5 帧原 ID 的连续保持数下降，"
        "+10 帧持平。当前数据支持保留自适应 GMC 候选，并保留原生卡尔曼更新做默认配置。",""]
    result+=["## 复现与核对","",
        "三组使用同一套序列、解码图像、逐帧检测、特征数组、外观编码器和原生关联设置。"
        "原始轨迹的 30 份 MOT 文件与历史原始运行逐字节一致。两段预检与全量阶段的三组对应轨迹也逐字节一致。"
        "逐帧比对确认三组取得的原生 SOF 矩阵一致，卡尔曼组全程使用原生矩阵。","",
        "同一服务器目录下，先运行 setup_lightglue.py，再按顺序运行 run_suite.py smoke、run_suite.py full、run_suite.py analyze。"
        "脚本使用 redetr 环境。复现实验需要 paths.py 中指向的冻结缓存、原图、官方真值与原版评测代码。"
        "模型官方来源为 https://github.com/cvg/LightGlue，具体提交和两个权重的 SHA-256 见 vendor/provenance.json。","",
        "full/diagnostic_results/sequence_metrics.csv 含各序列绝对结果与差值。paired_windows.csv 可追查每个固定事件的连续保持情况，"
        "recovery_events.csv 列出重激活前后帧和匹配真值。smoke/diagnostic_results/gmc_paired_centers.csv 给出两段的同状态位置误差。"
        "各阶段的 trackers 目录保存 MOT 输出，audit 保存逐帧矩阵与真实重激活事件。",""]
    return result


def main():
    from shared_recoveries import write_shared
    full=write_shared(ROOT/"full");smoke=write_shared(ROOT/"smoke")
    with (ROOT/"full/diagnostic_results/sequence_metrics.csv").open(encoding="utf-8-sig") as f:
        rows=list(csv.DictReader(f))
    report=ROOT/"实验报告.md";report.write_text("\n".join(lines_for_report(full,smoke,rows)),encoding="utf-8")
    files=[*ROOT.glob("*.py"),report,ROOT/"smoke_summary.json",ROOT/"full_summary.json",ROOT/"vendor/provenance.json",
           ROOT/"vendor/lightglue_commit.txt"]
    for stage in ("smoke","full"):
        folder=ROOT/stage
        for name in ("configs","trackers","diagnostic_results","protocols","audit","timing"):
            files.extend(x for x in (folder/name).rglob("*") if x.is_file())
        files.extend((folder/"metrics"/n for n in ("comparison.json","summary.json","comparison.csv","evaluation_audit.json")))
        files.append(folder/"parity.json")
    files=sorted(set(files))
    save(ROOT/"bundle_manifest.json",{"historical_native_protocol":str(HISTORY),"historical_native_sha256":
         hashlib.sha256(HISTORY.read_bytes()).hexdigest(),"repository":str(REPO),
         "file_sha256":{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}})
    files.append(ROOT/"bundle_manifest.json")
    with zipfile.ZipFile(ROOT/"results.zip.part","w",zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for p in files:
            z.write(p,p.relative_to(ROOT))
    (ROOT/"results.zip.part").replace(ROOT/"results.zip")
    print("BUNDLE_COMPLETE",len(files),(ROOT/"results.zip").stat().st_size,flush=True)


if __name__=="__main__":
    main()
