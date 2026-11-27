"""Measured overlays on unchanged RGB frames; evaluation and illustration share the same box coordinates."""
from collections import defaultdict
import csv
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from common_audit import ROOT,SETTINGS,SEQ,SKIP,FULL,read
from clear_audit import source,evaluate_rows
from botsort_plus.io import resolve_image

GREEN="#00f0a4"
PINK="#ff69bf"
YELLOW="#ffd15c"
DARK="#15212c"


def render():
    ds=source();cfg=read(SETTINGS)
    gt=defaultdict(dict)
    with (Path(cfg["gt_root"])/SEQ/"gt/gt.txt").open() as f:
        for row in csv.reader(f):
            if float(row[6])!=0:
                gt[int(row[0])][int(row[1])]=list(map(float,row[2:6]))
    versions=(SKIP,FULL,"dual_low","dual_low_center")
    info={m:evaluate_rows(ROOT/"focus/trackers",m,ds)[0] for m in versions}
    font=lambda n,b=False:ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans"+("-Bold" if b else "")+".ttf",n)
    columns=220; height=159; side=156; top=94; scale=5; cw,ch=40,24
    (ROOT/"figures").mkdir(exist_ok=True)
    def sheet(name,title,targets,versions_used):
        totalw=side+columns*len(targets)+8
        canvas=Image.new("RGB",(totalw,top+height*(1+len(versions_used))+32),"#f4f6f7")
        draw=ImageDraw.Draw(canvas)
        draw.text((14,10),title,font=font(19,True),fill=DARK)
        draw.text((14,40),"Green: official GT     Yellow: input detection     Pink: published track     Original pixels at 5x",font=font(15),fill=DARK)
        for col,(frame,gid) in enumerate(targets):
            x0=side+col*columns
            draw.text((x0,70),f"Frame {frame}  |  GT {gid}",font=font(15,True),fill=DARK)
            x,y,w,h=gt[frame][gid]
            left=int(round(x+w/2-cw/2));upper=int(round(y+h/2-ch/2))
            with Image.open(resolve_image(ds[frame-1]["image"],Path(cfg["images_root"]))) as im:
                crop=im.convert("RGB").crop((left,upper,left+cw,upper+ch)).resize((cw*scale,ch*scale),Image.Resampling.NEAREST)
            def coords(box,offset):
                bx,by,bw,bh=box
                return (x0+(bx-left)*scale,offset+(by-upper)*scale,x0+(bx+bw-left)*scale,offset+(by+bh-upper)*scale)
            def nearest(m):
                items=info[m][frame-1].values()
                return max((v for v in items if v["gt"]==gid or v["near_gt"]==gid or v["raw_near_gt"]==gid),
                           key=lambda v:(v["gt"]==gid,v["max_iou"]),default=None)
            for rr,m in enumerate(("detection",*versions_used)):
                y0=top+rr*height
                if col==0:
                    tag={"detection":"Detection",SKIP:"Skip only",FULL:"Full GMC","dual_low":"Dual low gate",
                         "dual_low_center":"Gate + center"}[m]
                    draw.text((10,y0+48),tag,font=font(15,True),fill=DARK)
                canvas.paste(crop,(x0,y0))
                v=nearest(FULL if m=="detection" else m)
                note="No output here"
                if v is not None:
                    box=v["raw_xywh"] if m=="detection" else v["box"]
                    draw.rectangle(coords(box,y0),outline=YELLOW if m=="detection" else PINK,width=2)
                    note=(f"det {v['det']}  conf {v['score']:.2f}" if m=="detection" else
                          f"ID {v['tid']}  {v['status']}  IoU {v['max_iou']:.3f}")
                draw.rectangle(coords(gt[frame][gid],y0),outline=GREEN,width=2)
                draw.text((x0,y0+ch*scale+5),note,font=font(13),fill=DARK)
        draw.text((12,canvas.height-27),"Detection row shows the frozen detection accepted by Full GMC. Boxes use the original RGB coordinate system.",font=font(12),fill=DARK)
        canvas.save(ROOT/"figures"/name)
    sheet("01_low_score_chain.png","Continuation of track 4 around GT 6",[(f,6) for f in (397,398,400,406,407)],
          (SKIP,FULL,"dual_low"))
    sheet("02_identity_and_localization.png","GT 2 identity continuity and GT 1 localization",[(409,2),(410,2),(412,2),(473,1),(474,1)],
          (SKIP,FULL,"dual_low_center"))
