"""Official-GT/reference-output comparisons on nearest-neighbour crops of original RGB."""
import csv
from collections import defaultdict
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import clear_audit as audit
from botsort_plus.io import resolve_image
from common import OUT,JOINT,B,COMBO,SETTINGS,read,save
from common_audit import sha,csvout

def draw(seq,targets):
    audit.SEQ=seq
    cfg=read(SETTINGS); ds=audit.source()
    gt=defaultdict(dict)
    for row in csv.reader((Path(cfg["gt_root"])/seq/"gt/gt.txt").open()):
        if float(row[6]): gt[int(row[0])][int(row[1])]=list(map(float,row[2:6]))
    data=[audit.evaluate_rows(JOINT/"full/trackers",m,ds)[0] for m in (B,COMBO)]
    gtrows=[]
    for t in range(1,len(ds)+1):
        found=[{v["gt"]:v for v in rows[t-1].values() if v["gt"] is not None} for rows in data]
        for gid,bbox in gt[t].items():
            item=dict(frame=t,gt=gid,x=bbox[0],y=bbox[1],w=bbox[2],h=bbox[3])
            for side,lookup in zip(("B","BG"),found):
                det=lookup.get(gid)
                item[side+"_tid"]=det["tid"] if det else None
                item[side+"_det"]=det["det"] if det else None
                item[side+"_IoU"]=det["max_iou"] if det else None
                item[side+"_status"]="TP" if det else "FN"
            gtrows.append(item)
    csvout(OUT/seq/"official_GT_by_frame.csv",gtrows)
    save(OUT/seq/"GT_provenance.json",dict(gt_sha256=sha(Path(cfg["gt_root"])/seq/"gt/gt.txt"),
         valid_gt_boxes=len(gtrows),frames=len(ds),clear_sha256=sha(audit.TRACK_EVAL/"trackeval/metrics/clear.py")))
    def font(size):
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",size)
    scale=5; w=42; h=26; col=240; margin=91; row_h=202
    canvas=Image.new("RGB",(margin+len(targets)*col+4,99+2*row_h),"#fafafa")
    paint=ImageDraw.Draw(canvas)
    paint.text((12,6),seq+"  |  matched identity and detection row",font=font(22),fill="#182832")
    paint.text((12,40),"Green: official GT     Pink: output for selected detection     Blue: GT owner on another detection",font=font(16),fill="#182832")
    meta=[]
    for j,(t,gid,det) in enumerate(targets):
        x,y,bw,bh=gt[t][gid]; x0=round(x+bw/2-w/2); y0=round(y+bh/2-h/2)
        xcol=margin+col*j
        paint.text((xcol,72),f"f{t}   GT {gid}   det {det}",fill="#24343b",font=font(15))
        with Image.open(resolve_image(ds[t-1]["image"],Path(cfg["images_root"]))) as image:
            crop=image.convert("RGB").crop((x0,y0,x0+w,y0+h)).resize((w*scale,h*scale),Image.Resampling.NEAREST)
        for i,method in enumerate(("B","B + GMC")):
            rows=data[i][t-1]; selected=rows.get(det)
            owner=next((r for r in rows.values() if r["gt"]==gid),None)
            patch=crop.copy();d=ImageDraw.Draw(patch)
            def box(v,color):
                a,b,c,e=v
                d.rectangle([(round((a-x0)*scale),round((b-y0)*scale)),(round((a+c-x0)*scale),round((b+e-y0)*scale))],outline=color,width=2)
            if owner and owner["det"]!=det: box(owner["box"],"#39a8ff")
            if selected: box(selected["box"],"#ff6cab")
            box(gt[t][gid],"#00fb83")
            yy=99+i*row_h; canvas.paste(patch,(xcol,yy))
            if j==0: paint.text((8,yy+62),method,fill="#182832",font=font(15))
            label=(f"ID {selected['tid']}   {selected['status']}" if selected else "No output for this det")
            paint.text((xcol,yy+h*scale+5),label,fill="#182832",font=font(15))
            paint.text((xcol,yy+h*scale+26),"GT owner: "+(f"ID {owner['tid']}, det {owner['det']}" if owner else "unmatched"),fill="#182832",font=font(14))
            meta.append(dict(frame=t,gt=gid,det=det,method=method,selected=selected,owner=owner,official_gt_xywh=gt[t][gid]))
    canvas.save(OUT/seq/"frames.png")
    save(OUT/seq/"illustrated_frames.json",meta)

draw("DJI_0030_2",[(371,4,6),(408,6,4),(427,4,2),(433,4,3),(444,4,3)])
draw("DJI_0169_3",[(108,6,12),(111,6,9),(116,6,11),(174,5,14),(191,5,14)])
