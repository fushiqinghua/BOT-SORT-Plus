"""Build synchronized RGB crops and machine-readable full-sequence timelines."""
import argparse
import csv
import gzip
import json
import os
import sqlite3
from bisect import bisect_left
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from build_evidence import METHODS, csv_file, read


CASES = [
    ("01_novel_car","DJI_0327_3","gt",65,[(6,18)],[8,13,15,16],"新增车辆与小框定位","GT #65 全程 390 帧。两组 A2 均命中 137 帧，B_ref 全程漏跟。第 8 至 12 帧 A2 已有近框，第 13 帧首次达到 CLEAR 匹配条件。"),
    ("02_earlier_ship","DJI_0030_2","gt",3,[(1,14)],[4,6,10,11],"已知船舶提前起跟","GT #3 首次匹配分别在第 489、6、11 帧。全段匹配帧数分别为 100、500、495，三组在各自命中帧中均维持一个预测 ID。"),
    ("03_threshold_true_loss","DJI_0141_2","gt",33,[(121,145)],[128,132,138,142],"提高门限后丢失的真目标","GT #33 是 0.18 独自补回的五个目标之一。A2B #50 命中 25 帧；B_ref 和 0.20 全程漏跟。两阶段实际均分为 0.191 和 0.188。"),
    ("04_persistent_false","DJI_0181_3","fp",12,[(1,12),(29,40)],[5,32,34,36],"高于两个门限的长时误报","A2B #12 在第 5 至 200 帧之间共输出 170 帧，其框与任何 GT、任何 B_ref 框的整个时段最大 IoU 均为 0。0.20 在这 170 帧均输出同一检测行。"),
    ("05_threshold_fp_removed","DJI_0283_1","fp",33,[(283,317)],[298,302,309,314],"提高门限后消失的误报","A2B #33 的第一阶段均分为 0.18019，第二阶段为 0.26657。45 个输出帧均与 GT、B_ref 和 0.20 的框零重叠。"),
    ("06_guard_switch","DJI_0061_2","gt",161,[(165,185)],[172,177,181,184],"新建身份时排除了旧行人 ID","GT #161 在 B_ref 下始终由 #197 匹配。第 181 帧两组 A2 都将旧 ID 排除，日志记录当前外观距离为 0.699，超过 0.65。"),
    ("07_existing_id_switch","DJI_0061_2","gt",172,[(183,195)],[184,188,192,193],"已有 ID 回到另一个真目标","第 192 帧，新旧预测 ID 的交替发生在高分关联。A2B 的 #250、0.20 的 #243 先前都匹配过 GT #161，此处开始匹配 GT #172。"),
    ("08_continuity_control","DJI_0129_2","gt",1,[(40,49),(70,81)],[42,45,46,78],"短缺口后保留同一身份","该飞机在原 A 诊断中出现过短缺口换号。本次三组对 GT #1 均命中 549/590 帧，各使用一个 ID，均无换号。"),
]
COLORS=("#00765A","#2F61A3","#AF4326","#7246A3")
COLUMNS=("官方 RGB 真值","B_ref","A2B · 0.18","A2B · 0.20")
FONTS=Path("/usr/share/fonts/opentype/noto")


@lru_cache(None)
def font(size,bold=False):
    override = os.environ.get("VTTINY_CJK_BOLD_FONT" if bold else "VTTINY_CJK_FONT") or os.environ.get("VTTINY_CJK_FONT")
    if override:
        return ImageFont.truetype(override,size)
    return ImageFont.truetype(str(FONTS / ("NotoSansCJK-"+("Bold" if bold else "Regular")+".ttc")),size,index=2)


def box(r):
    return tuple(r[k] for k in ("x","y","w","h"))


def iou(a,b):
    x,y,w,h=a;xx,yy,ww,hh=b
    area=max(0,min(x+w,xx+ww)-max(x,xx))*max(0,min(y+h,yy+hh)-max(y,yy))
    return area/(w*h+ww*hh-area)


def outline(draw, rect, color, dotted):
    x,y,r,b=rect
    if not dotted:
        draw.rectangle(rect,outline="#FFFFFF",width=4)
        draw.rectangle(rect,outline=color,width=2)
        return
    for start,end,axis in [(x,r,0),(y,b,1)]:
        for n in range(round(start),round(end)+1,6):
            v=min(n+3,end)
            if v<n:continue
            for const in ([y,b] if not axis else [x,r]):
                pts=(n,const,v,const) if not axis else (const,n,const,v)
                draw.line(pts,fill="#FFFFFF",width=4);draw.line(pts,fill=color,width=2)


def best_near(items, anchor):
    return max(((iou(box(r),anchor),r) for r in items),key=lambda t:t[0],default=(0,None))


def make_panel(frame, rgb, roi, columns, texts, dashed, title):
    W,H=265,126
    x,y,w,h=roi
    crop=rgb.crop((x,y,x+w,y+h)).resize((W,H),Image.Resampling.NEAREST)
    panel=Image.new("RGB",(4*(W+12)+8,199),"white")
    draw=ImageDraw.Draw(panel)
    for i,((coords,_),description,missing) in enumerate(zip(columns,texts,dashed)):
        cx=8+i*(W+12)
        draw.text((cx,0),COLUMNS[i],fill=COLORS[i],font=font(18,True))
        view=crop.copy()
        if coords is not None:
            a,b,c,d=coords
            rect=((a-x)*W/w,(b-y)*H/h,(a+c-x)*W/w,(b+d-y)*H/h)
            outline(ImageDraw.Draw(view),rect,COLORS[i],missing)
        panel.paste(view,(cx,31))
        draw.text((cx,160),description,fill="#18303C",font=font(15,True))
    draw.text((8,181),f"帧 {frame}  ·  {title}",fill="#465B65",font=font(12))
    return panel


def run(full, out, raw):
    dets=Path(__file__).resolve().parents[6]/"results/vttiny/yolo11m_rgb_rgbt/rgbt/detections"
    db=sqlite3.connect(out/"analysis/evidence.sqlite");db.row_factory=sqlite3.Row
    root=out/"cases";root.mkdir(exist_ok=True)
    index=[]
    for name,seq,kind,target,windows,keys,title,note in CASES:
        folder=root/name
        frame_dir=folder/"frames";frame_dir.mkdir(parents=True,exist_ok=True)
        check=set(f for start,end in windows for f in range(start,end+1));assert set(keys)<=check
        allgt=defaultdict(list)
        for g in db.execute("SELECT * FROM gt_frames WHERE sequence=? ORDER BY frame,gt",(seq,)):
            allgt[g['frame']].append(dict(g))
        outputs=defaultdict(list)
        for o in db.execute("SELECT * FROM outputs WHERE sequence=? ORDER BY frame,method,tid",(seq,)):
            outputs[o['frame'],o['method']].append(dict(o))
        with gzip.open(dets/(seq+".json.gz"),"rt") as stream:cache=json.load(stream)['frames']
        watch={};anchors={};source={}
        if kind=="gt":
            for f,gg in allgt.items():
                g=next((x for x in gg if x['gt']==target),None)
                if g is not None:
                    watch[f]=g;anchors[f]=box(g);source[f]='official_gt'
        else:
            events=full/"events/A2B"/(seq+".jsonl")
            with (events.open(encoding="utf-8") if events.exists() else gzip.open(str(events)+".gz","rt",encoding="utf-8")) as stream:
                candidate=next(e['candidate_id'] for line in stream if (e:=json.loads(line))['event']=='birth_committed' and e['track_id']==target)
            with (events.open(encoding="utf-8") if events.exists() else gzip.open(str(events)+".gz","rt",encoding="utf-8")) as stream:
                for line in stream:
                    e=json.loads(line)
                    if e.get('candidate_id')==candidate and e['event'] in ('candidate_hit','candidate_seed'):
                        f=e['frame'];di=e['det_index']
                        a,b,c,d=cache[f-1]['detections'][di][:4];anchors[f]=(a,b,c-a,d-b)
                        source[f]='raw_candidate';watch[f]={"det_idx":di}
            for f in range(1,len(cache)+1):
                o=next((x for x in outputs[f,'A2B'] if x['tid']==target),None)
                if o is not None:
                    watch[f]=o;anchors[f]=box(o);source[f]='A2B_output'
        observed=sorted(anchors)
        median=np.median([anchors[f] for f in observed if f in check],axis=0)
        roi_width=max(62,round(median[2]*3.4),round(median[3]*6))
        roi_height=round(roi_width*126/265)
        capture={};timeline=[];summary=[]
        for fr in cache:
            f=fr['frame'];a=anchors.get(f);focal=watch.get(f)
            # Missing anchors provide only a viewport, never a synthetic output box.
            if a is None:
                i=bisect_left(observed,f);near=min(observed[max(0,i-1):i+1],key=lambda t:abs(t-f))
                center=anchors[near]
            else:center=a
            ax,ay,aw,ah=center
            x=int(np.clip(round(ax+aw/2-roi_width/2),0,640-roi_width))
            y=int(np.clip(round(ay+ah/2-roi_height/2),0,512-roi_height));roi=x,y,roi_width,roi_height
            gtmax,g=best_near(allgt[f],a) if a else (None,None)
            if kind=='gt' and a:g=focal
            elif gtmax is None or gtmax<.1:g=None
            cols=[(box(g) if g else None,g['gt'] if g else None)]
            labels=[f"GT #{g['gt']}  ·  {g['category']}" if g else f"最近 GT IoU {gtmax:.2f}" if a else "参照轨迹本帧缺测"]
            dashed=[False]
            row=dict(sequence=seq,frame=f,rgb=fr['image'],source=source.get(f),crop_x=x,crop_y=y,crop_w=roi_width,crop_h=roi_height,
                     anchor_xywh=json.dumps(a) if a else None,gt=cols[0][1],gt_max_iou=gtmax,
                     best_aligned_raw_score=focal['best_aligned_score'] if kind=='gt' and focal else None)
            for m in METHODS:
                view=outputs[f,m];ob=None;status='no_anchor';v=None
                if a:
                    v,near=best_near(view,a)
                    if kind=='gt':
                        ob=next((o for o in view if o['gid']==target),None)
                        if ob is not None:status='matched_GT'
                        elif v>=.1:ob=near;status='nearby_unmatched_to_focus'
                        else:status='no_match'
                    else:
                        ob=next((o for o in view if o['det_idx']==focal['det_idx']),None)
                        if ob is not None:status='same_raw_row'
                        elif v>=.1:ob=near;status='nearby_different_row'
                        else:status='no_nearby_output'
                prefix=m+'_'
                value=iou(a,box(ob)) if ob is not None else None
                cols.append((box(ob),ob['tid']) if ob is not None else (None,None))
                dashed.append(status.startswith('nearby'))
                if ob is not None:
                    rg=ob['gid'];category='TP' if rg is not None else 'FP'
                    if kind=='gt':
                        txt=(f"命中 #{ob['tid']}" if status=='matched_GT' else f"近框 #{ob['tid']}") + f"  ·  IoU {value:.2f}"
                    else:txt=f"#{ob['tid']}  ·  {category}  ·  "+("同一检测行" if status=='same_raw_row' else "相邻框")
                elif not a:txt="参照轨迹本帧缺测"
                else:txt='无 CLEAR 匹配' if kind=='gt' else '无邻近输出'
                labels.append(txt)
                row.update({prefix+'status':status,prefix+'shown_tid':cols[-1][1],
                            prefix+'matched_focus':int(status=='matched_GT') if kind=='gt' and a else None,
                            prefix+'matched_gt':ob['gid'] if ob else None,prefix+'iou_to_anchor':value,
                            prefix+'max_iou_to_anchor':v,prefix+'stage':ob['stage'] if ob else None,
                            prefix+'det_idx':ob['det_idx'] if ob else None,prefix+'score':ob['score'] if ob else None,
                            prefix+'xywh':json.dumps(box(ob)) if ob else None})
            timeline.append(row)
            if f in check:
                with Image.open(raw/fr['image']) as src:
                    rgb=src.convert("RGB")
                assert rgb.size==(640,512)
                pane=make_panel(f,rgb,roi,cols,labels,dashed,title)
                pane.save(frame_dir/f"{f:04}.jpg",quality=93,subsampling=0)
                if f in keys:capture[f]=pane
                if f==keys[0]:
                    context=rgb.copy()
                    cd=ImageDraw.Draw(context);cd.rectangle((x,y,x+roi_width,y+roi_height),outline="#FFFFFF",width=4)
                    cd.rectangle((x,y,x+roi_width,y+roi_height),outline="#006C9B",width=2)
                    context.save(folder/"context.jpg",quality=92)
                summary.append(dict(frame=f,anchor_source=source.get(f),panels=labels))
        csv_file(folder/"timeline.csv",timeline)
        canvas=Image.new("RGB",(capture[keys[0]].width,len(keys)*205+71),"white");draw=ImageDraw.Draw(canvas)
        draw.text((8,3),title+"   /   "+seq,font=font(26,True),fill="#172F3C")
        draw.text((8,43),"同帧同裁剪。实线为匹配或同检测行，虚线为未与关注目标匹配的邻近框。",font=font(15),fill="#425762")
        for i,f in enumerate(keys):canvas.paste(capture[f],(0,70+i*205))
        canvas.save(folder/"keyframes.jpg",quality=94,subsampling=0)
        doc=dict(folder=name,sequence=seq,focus_type=kind,focus=target,title=title,note=note,
                 windows=windows,keyframes=keys,all_video_frames=len(cache),rendered=len(check),keyframe_source=summary)
        index.append(doc)
    (root/"index.json").write_text(json.dumps(index,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=["# 八组逐帧证据","","每张图的四列依次是官方 RGB 真值、B_ref、A2B 和 A2B_score020。图中方法间的数字 ID 各自独立。点击关键帧图查看转折，打开同目录的 frames 可逐帧比较，timeline.csv 保存整段视频的数字明细。","",
           "真目标案例用 CLEAR 给出的 GT 配对，虚线展示与该 GT 的 IoU 至少 0.10 的邻近预测框。误报案例用 A2B 焦点框及其起轨前的真实检测作为参照。缺测帧仅沿用邻近位置选取裁剪视窗。context.jpg 显示该例首个关键帧在整张 RGB 图上的位置。",""]
    for rec in index:
        windows="、".join(f"{lo}–{hi}" for lo,hi in rec['windows'])
        lines += [f"## {rec['title']}","",rec['note']+f" 连续画面覆盖 MOT 第 {windows} 帧。",
                  f"[关键帧图]({rec['folder']}/keyframes.jpg) · [整段逐帧表]({rec['folder']}/timeline.csv) · [原图位置]({rec['folder']}/context.jpg)",""]
    (root/"案例索引.md").write_text("\n".join(lines),encoding='utf-8')
    db.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(__doc__);p.add_argument("output",type=Path)
    p.add_argument("--run-dir",type=Path,default=Path(__file__).resolve().parents[2]/"full30")
    p.add_argument("--images-root",type=Path,required=True)
    a=p.parse_args();run(a.run_dir,a.output,a.images_root)
