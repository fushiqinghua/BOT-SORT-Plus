"""Cross-check the online comparisons and package the numerical/visual evidence."""
import csv
import gzip
import json
from pathlib import Path
import zipfile
import numpy as np
from common_audit import (ROOT,REPO,INTEGRATIONS,SETTINGS,FACTOR,TRACK_EVAL,SEQ,SKIP,FULL,FIELDS,
                          read,write,sha,csvin,csvout,motions)

LOW="dual_low";CENTER="dual_low_center";PIXEL="pixel_margin"
NAMES={SKIP:"仅跳过补偿",FULL:"完整自适应 GMC",LOW:"低分匹配双重确认",CENTER:"双重确认＋高分检测中心",PIXEL:"仅提高重估采用门限"}
COUNTS={"FP","FN","IDSW"}


def table(headers,rows):
    return ["| "+" | ".join(headers)+" |","| "+" | ".join("---" for _ in headers)+" |",
            *["| "+" | ".join(map(str,r))+" |" for r in rows],""]


def value(k,x,delta=False):
    if k in COUNTS:
        return format(int(x),"+,d" if delta else ",d")
    return format(float(x),"+.4f" if delta else ".4f")


def check_centers():
    root=ROOT/"full"
    seqs=read(root/"protocols"/(LOW+".json"))["sequences"]
    path=Path(read(SETTINGS)["baseline_root"])/"detections"
    checks=[]
    for seq in seqs:
        with gzip.open(path/(seq+".json.gz"),"rt",encoding="utf-8") as f:
            frames=json.load(f)["frames"]
        lg={r["frame"] for r in motions(FULL,seq) if r["choice"]=="lightglue"}
        allowed=lg|{f+1 for f in lg}
        files=[(root/"trackers"/m/"data"/(seq+".txt")).open() for m in (LOW,CENTER)]
        total=changed=eligible=0
        try:
            for a,b in zip(csv.reader(files[0]),csv.reader(files[1]),strict=True):
                total+=1
                assert (a[:2],a[6:])==(b[:2],b[6:]),(seq,a[:2],"output identity")
                f=int(a[0]);score=float(a[6])
                active=f in allowed and score>.5
                eligible+=int(active)
                x=np.array(a[2:6],float);y=np.array(b[2:6],float)
                assert np.allclose(x[2:],y[2:],rtol=0,atol=1e-9)
                if not active:
                    assert a==b,(seq,f,"outside center window")
                else:
                    ds=np.asarray(frames[f-1]["detections"],np.float32).reshape(-1,6)
                    matches=ds[ds[:,4].astype(float)==score,:4].astype(float)
                    centers=(matches[:,:2]+matches[:,2:])/2
                    assert len(centers)>0 and np.any(np.max(np.abs(centers-y[:2]-y[2:]/2),axis=1)<1e-8)
                changed+=int(not np.allclose(x,y,rtol=0,atol=1e-9))
        finally:
            for s in files:s.close()
        assert eligible==read(root/"actions"/CENTER/(seq+".json"))["center_outputs"]
        checks.append(dict(sequence=seq,outputs=total,eligible=eligible,geometry_changes=changed,identities_and_scores_equal=True))
    csvout(ROOT/"evidence/center_integrity.csv",checks)
    return dict(rows=sum(x["outputs"] for x in checks),eligible=sum(x["eligible"] for x in checks),
                changed=sum(x["geometry_changes"] for x in checks),sequences=len(checks))


def text_report(focus,full,count):
    cfs={r["tracker"]:r for r in read(ROOT/"counterfactual/summary.json")}
    keys=("HOTA","IDF1","MOTA","FP","FN","IDSW")
    order=[SKIP,FULL,LOW,CENTER]
    lines=["# DJI_0030_2 的 11 个净增误报与身份保持","",
      "590 帧、4,596 个官方 RGB 标注框。原始卡尔曼、RGB–T 检测、RGB ReID 和原生关联参数保持一致。"
      "逐帧调用本项目固定版本 TrackEval 的 CLEAR 分配，按同帧同一行冻结检测配对两个方法的输出，"
      "每一行对应关系都由唯一的检测分数验证。此处的换号遵循 CLEAR 的跨帧身份定义。","",
      "完整 GMC 相对仅跳过补偿有 13 个新增误报、2 个原有误报消失，净增 11 个。"
      "其中九个来自同一条轨迹的低分续接。双重确认消除了这九个误报，输出中心修正再取得一个净误报改善。"
      "两个版本均保住真值 2 的原 ID。","",
      "## 哪些帧发生了什么",""]
    lines+=table(["帧","完整 GMC 相对仅跳过的变化","净增 FP"],[
        ["398–406","轨迹 4 多输出九个低分框，九帧均未通过真值匹配。","+9"],
        ["408","轨迹 4 的冻结检测行由 6 换为 4，两次输出均为 FP。","0"],
        ["410","轨迹 1 输出了原先缺失的检测。原始检测与真值 2 的 IoU 为 0.642，滤波输出为 0.493。","+1"],
        ["474","真值 1 的同一检测从 TP 变为 FP。真值 2 附近另有一次 FP 检测行替换（1 换为 3）。","+1"]])
    lines+=["按检测行统计，12 项由无输出变为 FP，1 项由 TP 变为 FP，2 项由 FP 变为无输出。"
       "这些净变化分布在 11 帧。第 411 帧的 FN 减少 1 个，第 413 和 474 帧各增加 1 个，合计 FN 增加 1。","",
      "第 398 帧的检测置信度为 0.469，进入原生低分第二阶段。同一状态下备用变换与该检测的预测 IoU 为 0.378，"
      "LightGlue 变换使其升到 0.878，跨过 0.5 门限。已有轨迹从而持续接上第 399–406 帧的低分检测。"
      "第 398 帧独立背景点的中位残差从 1.624 降到 1.096 像素，背景对齐改善与这个局部跟踪框能否命中真值是两个不同的测量。","",
      "这九帧的真值 6 框均为 4×4 像素，冻结检测宽 8.0–9.0、高 8.0–9.1 像素。"
      "按面积计算，即使两框完全对齐，原始检测的 IoU 上限也只有 0.211–0.227。"
      "按官方真值，轨迹 4 在仅跳过组的 363 次输出、完整组的 372 次输出全部为 FP。"
      "它长期位于已标注目标附近，图像和 IoU 支持检测框尺度不符的解释。","",
      "![低分续接逐帧放大](figures/01_low_score_chain.png)","",
      "## 少掉的换号与哪一帧有关","",
      "仅跳过组在第 412 帧有两次换号。真值 1 从 ID 2 换到 12，真值 2 从 ID 1 换到 11。"
      "完整组保住真值 2 的 ID 1，真值 1 仍换号一次。两组通过真值身份和冻结检测行对齐。","",
      "对两组不同的十个变换逐一做两种反事实，共 20 次。每次只替换一帧的变换，然后从头实际重跑完整跟踪序列。"
      "下表列出能区分误报与换号来源的对照，完整明细在 counterfactual/metrics_rows.csv。",""]
    cfkeys=["drop_one_398","only_one_398","drop_one_408","drop_one_409","only_one_409","drop_one_474"]
    cfnames=["完整组仅撤回 398","仅跳过组只在 398 启用","完整组仅撤回 408","完整组仅撤回 409",
             "仅跳过组只在 409 启用","完整组仅撤回 474"]
    lines+=table(["单帧干预","FP","FN","IDSW","真值 2 换号帧","真值 1 换号帧"],[
        [name,r["FP"],r["FN"],r["IDSW"],r["gt2_switch_frames"] or "无",r["gt1_switch_frames"] or "无"]
        for name,r in zip(cfnames,[cfs[k] for k in cfkeys])])
    lines+=["撤回第 398 帧就少了 9 个 FP，仍然只换号一次。撤回第 409 帧则使真值 2 在 412 帧重新换号。"
            "仅在 409 帧启用 LightGlue 虽然保住真值 2，真值 1 后来在 435 帧又换了一次。","",
      "## 可在线使用的修正","",
      "低分匹配双重确认仅用于完整 GMC 选择 LightGlue 的帧。原生第二阶段的低分检测继续一条活动轨迹时，"
      "从同一补偿前状态计算 LightGlue 与备用变换的两个预测框，二者都要达到原生 IoU 门限 0.5。"
      "备用变换沿用仅跳过组的规则，从原生光流与恒等变换中选取。高分关联和卡尔曼状态更新沿用原实现。","",
      "第二版在此基础上处理边界定位。选用 LightGlue 的当前帧及紧接的一帧，对已接受且置信度超过原生高分门限的观测，"
      "发布检测框中心和原滤波框宽高。关联、输出 ID、轨迹状态和后续滤波继续使用第一版的结果。"
      "两条规则只读取截至当前帧的图像和检测。",""]
    lines+=table(["序列上的方案",*keys],[[NAMES[m],*[value(k,focus[m][k]) for k in keys]] for m in [*order,PIXEL]])
    lines+=["单用双重确认，相对完整 GMC 只移除了第 398–406 帧的九个 FP。加入输出中心修正后，"
        "第 410 帧的真值 2 从 0.493 提高到 0.635，第 474 帧的真值 1 从 0.462 提高到 0.551，均从 FP 转为 TP。"
        "代价是第 473 帧真值 1 的输出从 0.506 降到 0.479，由 TP 变为 FP，因而净减少 10 个 FP。"
        "相对仅跳过方案剩余的 1 个净增误报来自这一帧。","",
        "另试过提高总体重估采用门限，要求相对备用方案的背景中位残差至少额外改善 1 像素。"
        "该组在 412 帧保住真值 2，但真值 1 在 435 帧再次换号，整段 IDSW 回到 2。","",
        "![真值 2 的原 ID 与真值 1 的定位边界](figures/02_identity_and_localization.png)","",
        "## 固定规则迁移到官方 30 段","",
        "下列百分比指标由 TrackEval 在 12,590 帧上联合汇总，FP、FN、IDSW 为事件计数。两种修正都在真实在线图像链路重跑。",""]
    lines+=table(["方案",*FIELDS],[[NAMES[m],*[value(k,full[m][k]) for k in FIELDS]] for m in order])
    lines+=table(["相对完整 GMC",*["Δ"+k for k in FIELDS]],[[NAMES[m],*[value(k,full[m][k]-full[FULL][k],True) for k in FIELDS]] for m in (LOW,CENTER)])
    other=[]
    for m in (LOW,CENTER):
        row=dict(method=m)
        for k in ("FP","FN","IDSW"):
            row[k]=full[m][k]-full[FULL][k]-focus[m][k]+focus[FULL][k]
        other.append(row)
    csvout(ROOT/"evidence/other_29_delta_counts.csv",other)
    lines+=["剔除 DJI_0030_2 后，其余 29 段相对完整 GMC 的误报、漏检和换号净变化如下。事件计数按序列相加。",""]
    lines+=table(["其余 29 段","ΔFP","ΔFN","ΔIDSW"],[[NAMES[v["method"]],*[value(k,v[k],True) for k in ("FP","FN","IDSW")]] for v in other])
    lines+=["全量上，双重确认加中心修正相对完整 GMC 少 11 个 FP、1 个 FN，IDSW 保持 397。"
            "HOTA 增加约 0.0026 个百分点。其余 29 段合计少 1 个 FP，FN 及 IDSW 的净变化为零。"
            "其中 DJI_0028_3 少 1 个 FP 和 1 个 FN，DJI_0181_3 各多 1 个，DJI_0303_2 少 1 个 FP。"
            "单用双重确认时，目标序列以外只有 DJI_0303_2 的 FP 减少 1 个，其他序列三项事件计数持平。",""]
    lines+=[f"逐行核验双重确认与叠加输出中心修正的 {count['rows']:,} 条输出，二者的发布帧、ID、检测分数与框尺寸一致。"
            f"共有 {count['eligible']} 条高分输出位于中心修正窗口，实际坐标有变化的为 {count['changed']} 条。","",
            "## 复现与数据口径","",
            "规则是在检查 DJI_0030_2 的官方真值后设计，该序列是针对性诊断样本。完整 30 段包含该序列，"
            "而且历史上用于选模和开发调参。30 段结果用于检查同一测试集上的取舍。","",
            "source/official_gt_DJI_0030_2.txt 附该序列官方真值。diagnosis/frames.csv 含 590 帧各自的正确匹配、误报、"
            "漏检与换号数，detection_pairs.csv 含检测行、轨迹、真值、框与 IoU，changed_fp.csv 含 15 个发生变化的检测行。"
            "focus/audit 含修正前后相同口径的事件核对。trace/trace 含低分关联代价与实际检测索引，"
            "counterfactual/metrics_rows.csv 含 20 组单帧干预，full/compared.csv 含全量逐序列结果。","",
            "在同一服务器目录与 redetr 环境下，依次运行 python run_stage.py inspect、trace、counterfactual、focus、full、package"
            "（每次传一个阶段名）。上一轮冻结检测、ReID、变换和上游代码位置见 common_audit.py。"
            "两组历史轨迹在目标序列的冻结变换重放、实时重放和历史文件中逐字节一致。",""]
    return "\n".join(lines)


def main():
    from figures_case import render
    render()
    count=check_centers()
    focus=read(ROOT/"focus/live_summary.json");full=read(ROOT/"full/live_summary.json")
    assert focus["status"]==full["status"]=="COMPLETE" and full["frames"]==12590
    report=ROOT/"实验报告.md"
    report.write_text(text_report(focus["target"],full["joint"],count),encoding="utf8")
    files={p.name:p for p in ROOT.glob("*.py")}
    files[report.name]=report
    for part in ("diagnosis","trace","counterfactual","focus","full","figures","evidence"):
        for f in (ROOT/part).rglob("*"):
            if (not f.is_file() or f.suffix==".log" or "previous_attempts" in f.parts or
                ("metrics" in f.parts and f.name not in ("summary.json","comparison.json","comparison.csv","evaluation_audit.json"))):
                continue
            files[f.relative_to(ROOT).as_posix()]=f
    cfg=read(SETTINGS)
    files["source/official_gt_DJI_0030_2.txt"]=Path(cfg["gt_root"])/SEQ/"gt/gt.txt"
    files["source/frozen_detections_DJI_0030_2.json.gz"]=Path(cfg["baseline_root"])/"detections"/(SEQ+".json.gz")
    for m in (SKIP,FULL):
        files["source/"+m+"_camera.jsonl.gz"]=FACTOR/"full/warps"/m/(SEQ+".jsonl.gz")
    for name in ("frozen_gmc.py","gmc_variants.py","replay_factor.py","settings_factor.py"):
        files["source/"+name]=FACTOR/name
    manifest=ROOT/"bundle_manifest.json"
    write(manifest,dict(scope=SEQ,gt_sha256=sha(files["source/official_gt_DJI_0030_2.txt"]),
        trackeval_clear_sha256=sha(TRACK_EVAL/"trackeval/metrics/clear.py"),center_audit=count,
        files={name:sha(path) for name,path in files.items()}))
    files[manifest.name]=manifest
    temp=ROOT/"results.zip.partial"
    with zipfile.ZipFile(temp,"w",zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for n,p in sorted(files.items()):z.write(p,n)
    temp.replace(ROOT/"results.zip")
    print("COMPLETE",len(files),sha(ROOT/"results.zip"),count,flush=True)


if __name__=="__main__":
    main()
