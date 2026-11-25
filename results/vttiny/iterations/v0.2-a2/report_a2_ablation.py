"""Build the portable results from the validated full run and offline attribution."""
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main(stage):
    root = stage / "experiments/full30"
    smoke = stage / "experiments/three_sequences"
    previous = stage.parent / "botsort_plus_v0_1_0/experiments/ablation_ab_full_v1"
    scores = {v["tracker"]:v for v in read(root/"metrics/comparison.json")["results"]}
    old = {v["tracker"]:v for v in read(previous/"metrics/comparison.json")["results"]}
    old_repo = stage.parent / "botsort_plus_v0_1_0/BOT-SORT-Plus"
    revisions = {k:subprocess.check_output(["git","rev-parse","HEAD"],cwd=p,text=True).strip()
                 for k,p in [("A2",stage/"BOT-SORT-Plus"),("v0.1",old_repo)]}
    with (root/"gt_analysis/summary.csv").open(encoding="utf-8-sig") as f:
        outcomes = {v["method"]:v for v in csv.DictReader(f)}
    refs = {}
    for now, before in [("baseline_ref","baseline"),("B_ref","B")]:
        x,y = [read(p)["output_prediction_hashes"] for p in
               [root/"protocols"/(now+".json"),previous/"protocols"/(before+".json")]]
        assert set(x)==set(y) and len(x)==30
        refs[now] = dict(historical=before,changed_sequences=[k for k in x if x[k]!=y[k]],
                         historical_metrics=old[before], current_metrics=scores[now])
    (root/"historical_refs.json").write_text(json.dumps(refs,indent=2),encoding="utf-8")
    met = read(root/"metrics/evaluation_audit.json")
    proto = read(root/"protocols/A2B.json")
    assert len(proto["sequences"])==30 and len(scores)==len(outcomes)==15
    assert met["status"]=="COMPLETE"
    for d, total in [(smoke,3),(root,30)]:
        assert read(d/"structure_audit.json")["status"]=="PASS"
        pairs=read(d/"paired_bytes.json")
        assert len(pairs)==2 and all(len(v)==total for v in pairs.values())
    keys=["HOTA","DetA","AssA","IDF1","MOTA","FP","FN","IDSW"]
    lines=["# VT-Tiny-MOT 模块 A2 全量消融","",
           f"新版本提交 `{revisions['A2']}`。上一版保存在独立目录，提交为 `{revisions['v0.1']}`。",
           "", "在 DJI_0030_2、DJI_0129_2、DJI_0327_3 上先完成四组验收，每一对 shadow 与参照的轨迹均逐文件对比。通过后使用相同检测与 RGB ReID 缓存、SOF 和官方 RGB 真值，对 30 段共 12,590 帧实施 15 组联合评估。两组 shadow 在全量 30 段上同样逐字节一致。","",
           "## 跟踪指标","", "| 配置 | HOTA ↑ | DetA ↑ | AssA ↑ | IDF1 ↑ | MOTA ↑ | FP ↓ | FN ↓ | IDSW ↓ |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name,v in scores.items():
        lines.append("| "+" | ".join([name]+[str(v[k]) if k in keys[-3:] else f"{v[k]:.3f}" for k in keys])+" |")
    lines.extend(["","## 主比较",""])
    ref, primary = scores["B_ref"],scores["A2B"]
    delta={k:primary[k]-ref[k] for k in keys}
    lines.append("A2B 相对 B_ref 的变化为 "+ "，".join(f"{k} {delta[k]:+d}" if k in keys[-3:] else f"{k} {delta[k]:+.3f} 点" for k in keys)+"。")
    lines.extend(["","## 逐目标帧与轨迹分析","",
                  "| 配置 | 相对 B 新增目标帧 | 丢失目标帧 | 正式弱起轨 | 起轨当帧换号 | 始终远离 GT 的弱 ID | 这些 ID 的误报帧 | 起轨延迟中位数 / 帧 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for name in scores:
        v=outcomes[name]
        lines.append("| "+" | ".join([name]+[v[k] or "—" for k in
                     ("gained_vs_B","lost_vs_B","weak_births","switches_at_weak_birth","never_near_ids","never_near_fp_frames","median_delay")])+" |")
    lines.extend(["","逐目标帧收益按同序列、同 GT、同帧的一对一 CLEAR 匹配计算。始终远离 GT 表示该弱 ID 在所有输出帧与当帧所有 GT 的最高 IoU 都小于 0.10。延迟按实际提交帧减去候选首帧，只统计成功提交的轨迹。逐序列 FP、FN、IDSW 已同 TrackEval 核对。","",
                  "E 改变弱观测确认，I 检查旧身份竞争，二者的同基座参照是 A1_ref 或 A1B_ref。score_window 为 3/5 均分，fixed_window 为 5/8 均分，score020 为两阶段均分提高至 0.20。官方测试集曾用于选模与门限校准，本实验按开发集结果解释。","",
                  "## 文件","", "全量指标与协议见 full30/metrics 和 full30/protocols。离线新增覆盖、换号和弱身份寿命见 full30/gt_analysis。两阶段的逐字节核验见 three_sequences/paired_bytes.json 与 full30/paired_bytes.json，原始轨迹和事件位于服务器相应实验目录。",""])
    report=stage/"A2全量消融结果.md"
    report.write_text("\n".join(lines),encoding="utf-8")
    zpath=stage/"A2_results_v020.zip"
    with ZipFile(zpath,"w",ZIP_DEFLATED,compresslevel=4) as z:
        z.write(report,report.name)
        for d in [smoke,root]:
            for sub in ["protocols","metrics","gt_analysis","trackers"]:
                for f in sorted((d/sub).rglob("*")):
                    if f.is_file():z.write(f,d.name+"/"+f.relative_to(d).as_posix())
            for n in ["structure_audit.json","paired_bytes.json","historical_refs.json"]:
                if (d/n).is_file():z.write(d/n,d.name+"/"+n)
        for f in sorted((stage/"logs").glob("*")):
            if f.name in ["install_apply.json","install_check.json","tests_core.log","tests_installer.log","orchestrator.log"]:
                z.write(f,"logs/"+f.name)
        for f in sorted((stage/"BOT-SORT-Plus/configs/botsort_plus").glob("*.json")):
            if 12<=int(f.name[:2])<=26:z.write(f,"configs/"+f.name)
        for n in ["run_ablation.py","a2_gt_ablation.py","report_a2_ablation.py"]:
            z.write(stage/n,n)
    with ZipFile(zpath) as z: assert z.testzip() is None
    sha=hashlib.sha256(zpath.read_bytes()).hexdigest()
    print(json.dumps(dict(sha256=sha,bytes=zpath.stat().st_size,branch=revisions,
                          scores=list(scores.values()),primary_outcomes=outcomes["A2B"]),ensure_ascii=False))


if __name__=="__main__":
    main(Path(sys.argv[1]))
