"""Draw aligned RGB crops from the archived outputs; frames are one-based MOT frames."""
import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def read_mot(path):
    return np.loadtxt(path, delimiter=",", ndmin=2)


def iou(box, targets):
    a = np.maximum(box[:2], targets[:, :2])
    b = np.minimum(box[:2] + box[2:], targets[:, :2] + targets[:, 2:])
    inter = np.maximum(b-a, 0).prod(axis=1)
    return inter / (box[2:].prod() + targets[:, 2:].prod(axis=1) - inter)


def main(repo, exp, diag, gt_folder, images):
    folder = diag / "analysis"
    with (folder / "case_frames.csv").open(encoding="utf-8-sig") as stream:
        pairs = {(x["sequence"], int(x["gt"]), int(x["frame"])): x for x in csv.DictReader(stream)}
    fonts = Path("/usr/share/fonts/opentype/noto")
    def font(size, bold=False):
        filename = "NotoSansCJK-" + ("Bold" if bold else "Regular") + ".ttc"
        return ImageFont.truetype(str(fonts / filename), size, index=2)
    cases = [
        ("DJI_0030_2", 265, 5, "真目标补检", "低分框建立 A #39"),
        ("DJI_0129_2", 42, 1, "同一目标换号", "短暂中断前"),
        ("DJI_0129_2", 45, 1, "同一目标换号", "A 在旧 ID 丢失时起轨"),
        ("DJI_0129_2", 47, 1, "同一目标换号", "A 的旧 ID 再次匹配"),
        ("DJI_0327_3", 111, None, "持续误报", "A #212 的起轨帧"),
        ("DJI_0327_3", 218, None, "持续误报", "同一误报的第 108 帧"),
    ]
    colors = ["#00815F", "#285DBA", "#B74019"]
    ink, dim = "#182A37", "#415763"
    w, ch, cw, iw = 1240, 141, 323, 307
    header, ih, left = 99, 98, 247
    page = Image.new("RGB", (w, header + len(cases)*ch + 49), "#FFFFFF")
    draw = ImageDraw.Draw(page)
    draw.text((18, 5), "模块 A 逐帧核对", font=font(30, True), fill=ink)
    draw.text((300, 18), "VT-Tiny-MOT  /  官方测试集  /  RGB 原图的相同局部区域", font=font(17), fill=dim)
    draw.line((18, 52, w-24, 52), fill="#BCCAD1", width=2)
    draw.text((20, 61), "证据所在帧", font=font(20, True), fill=ink)
    for col, title in enumerate(("官方真值", "BoT-SORT 基线", "BoT-SORT + A")):
        x = left + col*cw
        draw.rectangle((x, 69, x+8, 84), fill=colors[col])
        draw.text((x+17, 58), title, font=font(22, True), fill=ink)
    records = []
    for idx, (seq, f, gid, title, subtitle) in enumerate(cases):
        with gzip.open(repo/"results/vttiny/yolo11m_rgb_rgbt/rgbt/detections"/(seq+".json.gz"), "rt") as stream:
            fr = json.load(stream)["frames"][f-1]
        assert fr["frame"] == f
        image = Image.open(images / fr["image"]).convert("RGB")
        files = [gt_folder/seq/"gt/gt.txt", exp/"trackers/baseline/data"/(seq+".txt"),
                 exp/"trackers/A/data"/(seq+".txt")]
        rows = []
        for file in files:
            all_rows = read_mot(file)
            subset = all_rows[all_rows[:,0] == f]
            rows.append(subset)
        if gid is not None:
            pair = pairs[seq, gid, f]
            ids = [gid, int(pair["baseline_id"]) if pair["baseline_id"] else None,
                   int(pair["A_id"]) if pair["A_id"] else None]
            labels = ["GT #"+str(gid)] + ["ID #"+str(t) if t is not None else "无 CLEAR 匹配" for t in ids[1:]]
            target = rows[0][rows[0][:,1] == gid][0,2:6]
        else:
            ids = [None, None, 212]
            target = rows[2][rows[2][:,1] == 212][0,2:6]
            gt_max = iou(target, rows[0][:,2:6]).max(initial=0)
            baseline_max = iou(target, rows[1][:,2:6]).max(initial=0)
            assert gt_max == 0 and baseline_max < 0.5
            labels = ["与 A 框最大 IoU = 0", "无重叠达 0.50 的框", "ID #212"]
        roi_w = min(image.width, max(82, int(target[2]*3.1), int(target[3]*7.2)))
        roi_h = round(roi_w*ih/iw)
        x0 = int(np.clip(target[0] + target[2]/2-roi_w/2, 0, image.width-roi_w))
        y0 = int(np.clip(target[1] + target[3]/2-roi_h/2, 0, image.height-roi_h))
        view = image.crop((x0,y0,x0+roi_w,y0+roi_h)).resize((iw,ih),Image.Resampling.NEAREST)
        top = header + idx*ch
        if idx:
            draw.line((18,top-4,w-24,top-4),fill="#DFE6E9")
        draw.text((20,top+1),title,font=font(21,True),fill=ink)
        draw.text((20,top+36),seq+"  /  帧 "+str(f),font=font(16),fill=dim)
        draw.text((20,top+68),subtitle,font=font(15),fill=dim)
        hit_boxes = []
        for col, tid in enumerate(ids):
            x = left + col*cw
            panel = view.copy()
            pd = ImageDraw.Draw(panel)
            box = rows[col][rows[col][:,1] == tid][0,2:6] if tid is not None else None
            if box is not None:
                xx,yy,ww,hh=box
                rect=[(xx-x0)*iw/roi_w,(yy-y0)*ih/roi_h,(xx+ww-x0)*iw/roi_w,(yy+hh-y0)*ih/roi_h]
                pd.rectangle(rect, outline="#FFFFFF", width=4)
                pd.rectangle(rect, outline=colors[col], width=2)
            hit_boxes.append(box.tolist() if box is not None else None)
            page.paste(panel, (x,top+2))
            draw.text((x, top+ih+4), labels[col], font=font(17,tid is not None), fill=ink)
        records.append(dict(sequence=seq, frame=f, rgb=fr["image"], crop_xywh=[x0,y0,roi_w,roi_h],
                            gt=ids[0], baseline=ids[1], A=ids[2], focused_boxes_xywh=hit_boxes))
    draw.text((20, header+len(cases)*ch+3),
              "前三列为同帧同裁剪。真目标行仅绘该 GT 及其 CLEAR 匹配框。误报行以 A #212 为参照。帧号从 1 开始。",
              font=font(16),fill=dim)
    page.save(folder / "gt_baseline_A_cases.png")
    (folder/"image_cases.json").write_text(json.dumps(records, ensure_ascii=False,indent=2), encoding="utf-8")


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    for name in ("repo", "exp", "diag", "gt-folder", "images"):
        p.add_argument("--"+name,type=Path,required=True)
    main(**vars(p.parse_args()))
