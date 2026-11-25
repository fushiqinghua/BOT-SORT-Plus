"""Render the first iteration's six diagnostic frames using published box coordinates."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def render(source, output):
    cases = json.loads(source.read_text(encoding="utf-8"))
    image = Image.new("RGB",(860,78+100*len(cases)),"white")
    draw = ImageDraw.Draw(image)
    small, large = ImageFont.load_default(size=14), ImageFont.load_default(size=20)
    draw.text((12,8),"VT-Tiny-MOT  |  First-iteration box comparison",font=large,fill="#183047")
    for x, label in zip((12,204,424,644),("Sequence / MOT frame","Official GT","BoT-SORT baseline","BoT-SORT + A")):
        draw.text((x,50),label,font=small,fill="#183047")
    for i,case in enumerate(cases):
        y = 78+100*i
        draw.text((12,y+7),case["sequence"],font=small,fill="#183047")
        draw.text((12,y+31),f"Frame {case['frame']}",font=small,fill="#183047")
        x0,y0,w,h = case["crop_xywh"]
        scale = min(198/w,56/h)
        for j,(key,box) in enumerate(zip(("gt","baseline","A"),case["focused_boxes_xywh"])):
            left = 202+220*j
            draw.rectangle((left,y,left+204,y+62), fill="#f1f4f6")
            if box:
                x,b,bw,bh=box
                xx = left+(204-w*scale)/2+(x-x0)*scale
                yy = y+(62-h*scale)/2+(b-y0)*scale
                draw.rectangle((xx,yy,xx+bw*scale,yy+bh*scale),outline=("#14794c" if j==0 else "#1456ae"),width=3)
            caption = f"ID {case[key]}" if case[key] is not None else ("No GT for focus" if j==0 else "No matched box")
            draw.text((left+2,y+64),caption,font=small,fill="#183047")
    image.save(output)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    base=Path(__file__).resolve().parent/"analysis"
    parser.add_argument("--source",type=Path,default=base/"image_cases.json")
    parser.add_argument("--output",type=Path,default=base/"box_comparison.png")
    options=parser.parse_args()
    render(options.source,options.output)
