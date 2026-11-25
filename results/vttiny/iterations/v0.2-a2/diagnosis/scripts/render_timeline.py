"""Draw four aligned panels from recorded case timelines, optionally on official RGB frames."""
import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

METHODS = ("B_ref", "A2B", "A2B_score020")
WIDTH, HEIGHT = 232, 112
HEAD, FOOT, GAP = 27, 47, 7
MARGIN = 14
FONT = ImageFont.load_default(size=15)
SMALL = ImageFont.load_default(size=13)
TITLE = ImageFont.load_default(size=19)
INK = "#152a40"
GT = "#14794c"
PRED = "#1456ae"
CANDIDATE = "#ac6200"


def viewbox(row):
    return tuple(int(row["crop_" + field]) for field in ("x", "y", "w", "h"))


def draw_box(draw, xywh, view, offset, scale, color, dashed=False):
    if not xywh:
        return
    x, y, w, h = xywh
    a = offset[0] + (x - view[0]) * scale
    b = offset[1] + (y - view[1]) * scale
    r, t = a + w * scale, b + h * scale
    if not dashed:
        draw.rectangle((a, b, r, t), outline=color, width=3)
        return
    for i in range(round(a), round(r)+1, 7):
        for side in (b, t):
            draw.line((i,side,min(i+4,r),side), fill=color, width=2)
    for i in range(round(b), round(t)+1, 7):
        for side in (a, r):
            draw.line((side,i,side,min(i+4,t)), fill=color, width=2)


def panel(row, method, rgb):
    view = viewbox(row)
    x, y, w, h = view
    scale = min((WIDTH-4)/w, (HEIGHT-4)/h)
    out = Image.new("RGB", (WIDTH, HEIGHT), "#f3f5f7")
    dx, dy = (WIDTH-w*scale)/2, (HEIGHT-h*scale)/2
    if rgb:
        crop = rgb.crop((x,y,x+w,y+h)).resize((round(w*scale),round(h*scale)),Image.Resampling.NEAREST)
        out.paste(crop, (round(dx),round(dy)))
    d = ImageDraw.Draw(out)
    anchor = json.loads(row["anchor_xywh"]) if row["anchor_xywh"] else None
    is_gt = row["source"] == "official_gt"
    color = GT if is_gt else CANDIDATE
    if method is None:
        draw_box(d, anchor, view, (dx,dy), scale, color)
        if is_gt:
            lines = ("GT #" + row["gt"], "official RGB coordinates")
        else:
            lines = (row["source"] or "no anchor", "FP case reference")
    else:
        key = method + "_"
        box = json.loads(row[key+"xywh"]) if row[key+"xywh"] else None
        draw_box(d, box, view, (dx,dy), scale, PRED)
        draw_box(d, anchor, view, (dx,dy), scale, color, dashed=True)
        tid = row[key+"shown_tid"]
        desc = row[key+"status"].replace("_"," ")
        value = row[key+"iou_to_anchor"]
        head = ("#" + tid + "  " if tid else "") + desc
        lines = (head, (row[key+"stage"]+"   " if row[key+"stage"] else "") + ("IoU %.3f" % float(value) if value else ""))
    return out, lines


def render(row, images_root):
    width = 4*WIDTH + 3*GAP + 2*MARGIN
    canvas = Image.new("RGB", (width, HEAD+HEIGHT+FOOT), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((MARGIN,4), "MOT frame " + row["frame"], font=FONT, fill=INK)
    rgb = None
    if images_root:
        with Image.open(images_root/row["rgb"]) as image:
            rgb = image.convert("RGB")
    for index, method in enumerate((None, *METHODS)):
        pic, lines = panel(row, method, rgb)
        left = MARGIN + index*(WIDTH+GAP)
        canvas.paste(pic, (left,HEAD))
        for j, line in enumerate(lines):
            draw.text((left+3,HEAD+HEIGHT+3+18*j), line, font=SMALL, fill=INK)
    return canvas


def sheet(case, rows, images_root):
    panels = [render(rows[k], images_root) for k in case["keyframes"]]
    width = panels[0].width
    header = 78
    out = Image.new("RGB", (width, header+sum(v.height for v in panels)), "white")
    draw = ImageDraw.Draw(out)
    focus = ("GT #" if case["focus_type"] == "gt" else "A2B track #")+str(case["focus"])
    draw.text((MARGIN,7), case["sequence"]+"   |   "+focus, font=TITLE, fill=INK)
    draw.text((MARGIN,34), "Dashed = reference    Blue = tracker output    IoU is relative to the reference box", font=SMALL, fill=INK)
    for j, name in enumerate(("Reference",*METHODS)):
        draw.text((MARGIN+j*(WIDTH+GAP),55), name, font=FONT, fill=INK)
    y = header
    for pic in panels:
        out.paste(pic,(0,y))
        y += pic.height
    return out


def main(cases_dir, output, images_root, all_frames):
    for case in json.loads((cases_dir/"index.json").read_text(encoding="utf-8")):
        name = case["folder"]
        with (cases_dir/name/"timeline.csv").open(encoding="utf-8-sig",newline="") as f:
            rows = {int(row["frame"]):row for row in csv.DictReader(f)}
        dst = output/name
        dst.mkdir(parents=True,exist_ok=True)
        sheet(case, rows, images_root).save(dst/"keyframes.png")
        if all_frames:
            dest = dst/"frames"
            dest.mkdir(exist_ok=True)
            for lo,hi in case["windows"]:
                for frame in range(lo,hi+1):
                    single_frame = dict(case, keyframes=[frame])
                    sheet(single_frame, rows, images_root).save(dest/f"{frame:04}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path(__file__).resolve().parents[1]/"cases")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, help="Official dataset root containing test2017/")
    parser.add_argument("--all-frames", action="store_true")
    args = parser.parse_args()
    main(args.cases,args.output,args.images_root,args.all_frames)
