"""Render three spatially aligned views of selected calibration detections."""
import argparse
import os
import csv
import io
import json
from pathlib import Path
import zipfile

from PIL import Image, ImageDraw, ImageFont
from fetch_calibration import HERE, run_on_host

CASES=[('DJI_0067_2',16,32,64),('DJI_0067_2',105,32,68),('DJI_0127_1',71,1,0),
       ('DJI_0067_2',194,61,71),('DJI_0067_2',202,66,74)]


def fetch_pixels():
    code = """
import cv2, gzip, hashlib, io, zipfile, math
import numpy as np
from PIL import Image
images=Path(json.loads((t0/'offline_settings.json').read_bytes())['paths']['images_root']).resolve()
buf=io.BytesIO()
found={}
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 with gzip.open(old/'training_labels/calibration/pairs.runtime.jsonl.gz','rt') as stream:
  for row in map(json.loads,stream):
   key=tuple(row[k] for k in ('sequence','frame','track_id','det_index'))
   if key not in cases: continue
   index=cases.index(key)
   img=(images/row['frame_image']).resolve()
   assert 'train2017' in img.relative_to(images).parts and index not in found
   bgr=cv2.imdecode(np.frombuffer(img.read_bytes(),dtype=np.uint8), cv2.IMREAD_COLOR)
   assert hashlib.sha256(bgr.tobytes()).hexdigest()==row['image_sha256']
   x,y,rgt,btm=row['raw_xyxy']
   left=int(math.floor((x+rgt)/2))-12
   top=int(math.floor((y+btm)/2))-12
   tile=bgr[top:top+25,left:left+25]
   assert tile.shape==(25,25,3)
   out=io.BytesIO()
   Image.fromarray(tile[:,:,::-1]).save(out,format='PNG')
   z.writestr(str(index)+'.png',out.getvalue())
   found[index]=[left,top]
 assert len(found)==len(cases)
 z.writestr('crop_origins.json',json.dumps(found))
sys.stdout.buffer.write(buf.getvalue())
"""
    blob=run_on_host('cases='+repr(CASES)+'\n'+code)
    (HERE/'inputs/case_pixels.zip').write_bytes(blob)


def render():
    with (HERE/'crossings.csv').open(encoding='utf-8-sig') as f:
        rows={(r['sequence'],int(r['frame']),int(r['track_id']),int(r['det_index'])):r for r in csv.DictReader(f)}
    colors=['#B83239','#087CA3','#258047']
    ink='#1C2937'
    font_path=Path(os.environ.get('TCO_CJK_FONT', 'C:/Windows/Fonts/msyh.ttc'))
    def font(size):
        return ImageFont.truetype(str(font_path),size) if font_path.is_file() else ImageFont.load_default(size=size)
    canvas=Image.new('RGB',(1100,1040),'white')
    d=ImageDraw.Draw(canvas)
    d.text((28,17),'同一块原始 RGB 像素上的质量头错判',font=font(25),fill=ink)
    for i,(label,c) in enumerate(zip(['当前 RGB','原检测框','冻结修正框','官方训练真值'],[ink,*colors])):
        d.text((30+175*i,65),label,font=font(16),fill=c)
    d.text((738,65),'同一候选的增量门槛判定',font=font(16),fill=ink)
    with zipfile.ZipFile(HERE/'inputs/case_pixels.zip') as z:
        offsets=json.loads(z.read('crop_origins.json'))
        for i,key in enumerate(CASES):
            row=rows[key]
            left,top=offsets[str(i)]
            tile=Image.open(io.BytesIO(z.read(str(i)+'.png'))).convert('RGB').resize((150,150),Image.Resampling.NEAREST)
            y=103+i*181
            for j,name in enumerate((None,'raw_xyxy','refined_xyxy','official_gt_xyxy')):
                x=30+j*175
                canvas.paste(tile,(x,y))
                if name:
                    l,t,r,b=json.loads(row[name])
                    d.rectangle((x+(l-left)*6,y+(t-top)*6,x+(r-left)*6,y+(b-top)*6),
                                outline=colors[j-1],width=2)
            note='新增放行跌线框' if row['crossing']=='down' and row['transition']=='added' else (
                '撤回放行跌线框' if row['crossing']=='down' else '新增放行升线框')
            for j,val in enumerate([f'{key[0]}   帧 {key[1]}   轨迹 {key[2]}',
                f"真值 IoU   {float(row['raw_iou']):.4f} → {float(row['refined_iou']):.4f}",
                f"绝对监督 Δq   {float(row['abs_gain']):+.5f}",
                f"成对监督 Δq   {float(row['paired_gain']):+.5f}",note]):
                d.text((738,y+27*j),val,font=font(15),fill=ink)
            d.line((28,y+163,1072,y+163), fill='#DBE1E7')
    d.text((30,1011),'每幅取原始 25 × 25 像素区域，最近邻放大 6 倍。升线、跌线均指相对同一真值跨过 IoU 0.50。',font=font(14),fill=ink)
    canvas.save(HERE/'逐框对照.png')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--fetch',action='store_true')
    if p.parse_args().fetch:
        fetch_pixels()
    render()
