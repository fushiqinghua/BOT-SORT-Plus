"""Produce a self-contained result bundle with the four independent replays."""
import csv
import zipfile
from settings_factor import ROOT,PREVIOUS,REPO,VENDOR,FOCUS,ARMS,FIELDS,COUNT,load,save,sha

NAMES={"native":"原始 BoT-SORT","skip_only":"仅跳过不可靠补偿",
       "lightglue_only":"仅 LightGlue 重估","adaptive_gmc":"完整自适应 GMC"}
COMPARE={"skip_alone":"仅跳过 − 原始","lightglue_alone":"仅 LightGlue − 原始","both":"完整 − 原始",
         "add_lightglue_to_skip":"完整 − 仅跳过","add_skip_to_lightglue":"完整 − 仅 LightGlue",
         "interaction":"交互项"}


def format_value(k,v,signed=False):
    if k in COUNT:
        if v==int(v):
            return format(int(v),"+,d" if signed else ",d")
        return format(v,"+,.1f" if signed else ",.1f")
    return format(v,"+.4f" if signed else ".4f")


def table(headers,rows):
    return ["| "+" | ".join(headers)+" |","| "+" | ".join("---" for _ in headers)+" |",
            *["| "+" | ".join(str(x) for x in row)+" |" for row in rows],""]


def report(full,smoke):
    frames=full["inputs"]["frames"]
    joint=full["joint"];effects=full["effects"];counts=full["choices"]
    lines=["# VT-Tiny-MOT 自适应 GMC 的四组消融","",
        f"服务器在官方 30 个测试序列、{frames:,} 帧上独立重放四组原始 BoT-SORT。检测使用固定 YOLO11-M RGB–T 缓存，"
        "外观关联使用固定 RGB ReID 缓存。四组均使用原版卡尔曼预测、更新及相同的关联设置。","",
        "跳过不可靠补偿单独使用就减少了 48 个漏检和 20 次换号，误报总数持平。"
        "在这个方案上增加 LightGlue 后，HOTA 再提升 0.0778 个百分点、换号再减少 5 次，"
        "同时误报多了 14 个，漏检总数持平。完整方案取得四组最高的 HOTA、IDF1 和最低的换号数，仅跳过方案取得最高的 MOTA。","",
        "## 完整 30 段","",
        "百分比指标按 TrackEval 的 COMBINED_SEQ 联合汇总。HOTA、DetA、AssA、IDF1、MOTA 的差值单位为百分点，"
        "FP、FN 和 IDSW 分别为误报、漏检和换号次数。",""]
    lines+=table(["方法",*FIELDS],[[NAMES[m],*[format_value(k,joint[m][k]) for k in FIELDS]] for m in ARMS])
    for title,key in [("仅跳过补偿，相对原始", "skip_alone"),("仅使用 LightGlue，相对原始","lightglue_alone"),
                      ("已启用跳过后，再加入 LightGlue","add_lightglue_to_skip")]:
        d=effects[key]
        lines+=[f"{title}的 HOTA 变化 {d['HOTA']:+.4f} 个百分点，IDF1 变化 {d['IDF1']:+.4f} 个百分点，"
                f"换号变化 {d['IDSW']:+d} 次，误报变化 {d['FP']:+d} 个，漏检变化 {d['FN']:+d} 个。",""]
    lines+=["## 两个机制各自带来的变化","",
        "单独收益以原始 BoT-SORT 为参照。条件收益比较完整方案与只启用另一项机制的结果。"
        "两个候选可能在同一帧都合格，且相机变换会影响后续轨迹，两个单独收益可以重叠。",""]
    lines+=table(["差值",*FIELDS],[[label,*[format_value(k,effects[key][k],True) for k in FIELDS]]
                                 for key,label in COMPARE.items()])
    lines+=["交互项采用“完整 − 仅跳过 − 仅 LightGlue + 原始”计算。若要让两个数字恰好加总到完整方案的变化，"
        "可将各机制先加入、后加入时的两个增量取平均。下表给出这种对称分摊，分配出的半次是归因份额。",""]
    lines+=table(["两种加入顺序的平均",*FIELDS],[
        [label,*[format_value(k,effects[key][k],True) for k in FIELDS]]
        for key,label in [("skip_order_average","跳过不可靠补偿"),("lightglue_order_average","LightGlue 重估")]])
    lines+=["## 两个指定序列",""]
    brief=("HOTA","IDF1","MOTA","FP","FN","IDSW")
    lines+=table(["序列","方法",*brief],[[s,NAMES[m],*[format_value(k,smoke["focus"][s]["metrics"][m][k]) for k in brief]]
                                            for s in FOCUS for m in ARMS])
    lines+=["在 DJI_0109_1 上，两种机制单独使用都消除了原始方案的 13 次换号。"
            "有 25 帧两种候选都合格。完整方案在其中 15 帧选恒等变换、10 帧选 LightGlue。"
            "仅 LightGlue 与完整方案的八项评测指标相同，逐帧轨迹文件仍可区分。","",
            "在 DJI_0173_2 上，独立运行匹配器 446 次，LightGlue 变换通过全部采用条件的次数为零。"
            "仅 LightGlue 与原始轨迹逐字节一致，跳过 61 帧原生补偿后的轨迹与完整方案逐字节一致。",""]
    lines+=["## 逐帧决策","",
        "遇到触发条件时，两种单项组各自检查允许的候选；若没有候选通过则使用原生稀疏光流。"
        "完整方案从通过检查的候选中选背景中位残差更小者，残差相同时继续比较第 80 百分位。",""]
    key_order=("frames","triggers","skip_selected","lightglue_selected","both_eligible","full_skip","full_lightglue","full_sof")
    lines+=table(["范围","帧数","触发","仅跳过采用","仅 LightGlue 采用","两者都合格","完整选跳过","完整选 LightGlue","完整选原生"],
                 [[s,*[c[k] for k in key_order]] for s,c in
                  [(s,smoke["focus"][s]["choices"]) for s in FOCUS]+[("全量",counts)]])
    lines+=[f"全量中，{counts['both_eligible']} 帧同时存在两个合格候选，完整方案在其中 "
            f"{counts['both_choose_skip']} 帧选择跳过，{counts['both_choose_lightglue']} 帧选择 LightGlue。"
            f"仅跳过组的学习匹配器调用次数为 {counts['skip_calls']}，仅 LightGlue 和完整组分别调用 "
            f"{counts['lightglue_calls']:,}、{counts['full_calls']:,} 次。",""]
    with (ROOT/"full/analysis/metrics_by_sequence.csv").open(encoding="utf-8-sig") as f:
        byseq=list(csv.DictReader(f))
    changes={}
    for s in {x["sequence"] for x in byseq if x["sequence"]!="COMBINED_SEQ"}:
        group={x["method"]:x for x in byseq if x["sequence"]==s}
        changes[s]={k:float(group["adaptive_gmc"][k])-float(group["skip_only"][k]) for k in FIELDS}
    ranked=sorted(changes,key=lambda s:(-abs(changes[s]["HOTA"]),s))
    delta_seqs=[s for s in ranked if any(abs(changes[s][k])>1e-9 for k in FIELDS)]
    lines+=[f"在仅跳过方案上再允许 LightGlue 后，共 {len(delta_seqs)} 段出现至少一项正式指标变化。"
            "下表列出这些序列，全量逐序列结果保留在 CSV 中。",""]
    if delta_seqs:
        lines+=table(["序列","ΔHOTA","ΔIDF1","ΔMOTA","ΔFP","ΔFN","ΔIDSW"],[
            [s,*[format_value(k,changes[s][k],True) for k in brief]] for s in delta_seqs])
    lines+=["完整方案相对仅跳过方案，进一步减少的 5 次换号来自 DJI_0028_3（4 次）和 DJI_0030_2（1 次）。"
            "其中 DJI_0030_2 的 HOTA 提高 2.6547 个百分点，同时贡献新增 14 个误报中的 11 个。"
            "这也解释了为什么完整方案的身份关联指标更好，而 MOTA 略低于仅跳过方案。",""]
    lines+=["## 运行成本与复现","",
        "四种方法依次运行，LightGlue 使用同一张 GPU。以下为跟踪更新的累计耗时，包含缓存特征处理、关联、GMC 和实验记录。"
        "检测器推理、ReID 特征提取、图像读取、模型加载和 TrackEval 评测在这个计时范围以外。",""]
    lines+=table(["方法","全量跟踪更新（秒）","平均每帧（毫秒）"],[
        [NAMES[m],f"{full['replay_seconds'][m]:.2f}",f"{1000*full['replay_seconds'][m]/frames:.2f}"] for m in ARMS])
    lines+=["原生和完整方案在 30 段上的 MOT 轨迹均与上一轮逐字节相同，原生 SOF 矩阵、原图解码像素、"
        "检测指纹、特征数组与元数据均已校验。两个指定序列在预检和全量阶段的四组轨迹也完全一致。","",
        "设置沿用上一轮。原生变换在独立背景验证点上的中位残差超过 1 像素，或第 80 百分位超过 2 像素时触发。"
        "候选需使中位残差至少改善 0.35 像素且至少改善 20%，第 80 百分位最多恶化 0.25 像素。"
        "完整方案和两个单项组在同一帧使用相同的检测掩膜、验证点和候选几何约束。","",
        "在服务器相同目录结构和 redetr 环境中，依次运行 python run_suite.py smoke、python run_suite.py full、"
        "python run_suite.py analyze。冻结输入的位置在 settings_factor.py 中。原版实现与模型的校验信息见 provenance。"
        "预测过程仅使用截至当前帧的信息，官方真值由独立评测进程读取。","",
        "官方 30 段曾用于选模与开发调参。本轮保持设置固定，结果按这批序列上的开发性对照解读。",""]
    return "\n".join(lines)


def package():
    full=load(ROOT/"full_summary.json")
    smoke=load(ROOT/"smoke_summary.json")
    assert full["inputs"]["frames"]==12590 and full["inputs"]["sequences"]==30
    assert full["prior_metrics_and_frame_choices_match"] and smoke["prior_metrics_and_frame_choices_match"]
    report_file=ROOT/"实验报告.md"
    report_file.write_text(report(full,smoke),encoding="utf-8")
    files={p.name:p for p in ROOT.glob("*.py")}
    for p in (report_file,ROOT/"smoke_summary.json",ROOT/"full_summary.json"):
        files[p.name]=p
    files["provenance/lightglue.json"]=VENDOR/"provenance.json"
    files["provenance/previous_model_setup.py"]=PREVIOUS/"setup_lightglue.py"
    for stage in ("smoke","full"):
        root=ROOT/stage
        for sub in ("analysis","configs","protocols","timing","warps","trackers"):
            for p in (root/sub).rglob("*"):
                if p.is_file():
                    files[p.relative_to(ROOT).as_posix()]=p
        for name in ("input_parity.json","metrics/summary.json","metrics/comparison.json","metrics/evaluation_audit.json"):
            p=root/name;files[p.relative_to(ROOT).as_posix()]=p
    manifest=ROOT/"bundle_manifest.json"
    save(manifest,dict(repository=str(REPO),previous_experiment=str(PREVIOUS),
        sha256={name:sha(p) for name,p in files.items()}))
    files[manifest.name]=manifest
    tmp=ROOT/"results.zip.part"
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for name,path in sorted(files.items()):
            z.write(path,name)
    tmp.replace(ROOT/"results.zip")
    print("BUNDLE_COMPLETE",len(files),sha(ROOT/"results.zip"),flush=True)
