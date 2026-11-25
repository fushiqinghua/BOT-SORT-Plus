"""Frame-local infrared output adjustment; measured coordinates and tracker state are untouched."""
from collections import Counter,deque
from pathlib import Path
import cv2
import numpy as np
from .position_localizer import decision,sample

class OnlinePosition:
    def __init__(self,policy,registration,modality="ir"):
        if registration['model']!='identity' or registration['framewise']:
            raise ValueError('This position rule was calibrated on the fixed identity mapping')
        if modality not in ("ir","rgb"):
            raise ValueError("Unsupported position image modality")
        self.modality=modality
        self.profile=policy['profile'];self.gain=policy['gain'];self.registration=registration
        self.images=deque(maxlen=4)
        self.history={}
        self.previous=0
        self.status=Counter()

    def update(self,frame,rgb_path,prediction,events):
        if frame!=self.previous+1:
            raise ValueError('Localization requires contiguous sequence frames')
        self.previous=frame
        rgb=Path(rgb_path)
        if rgb.parent.name!='00':
            raise ValueError('Expected paired dataset layout with RGB 00 and IR 01')
        ir_path=rgb.parent.parent/'01'/rgb.name if self.modality=='ir' else rgb
        img=cv2.imread(str(ir_path),cv2.IMREAD_GRAYSCALE) if ir_path.is_file() else None
        if img is not None and img.shape!=(512,640):
            raise ValueError('Unexpected IR image size')
        self.images.append((frame,img))
        available=dict(self.images)
        boxes={}
        measured={}
        for row in prediction:
            x,y,x2,y2,tid,_,_,det=map(float,row)
            b=np.asarray([x,y,x2-x,y2-y])
            boxes[int(tid)]=b
            if det>=0:measured[int(tid)]=b
        ages={int(e['track_id']):int(e['age']) for e in events if e['event']=='motion_output'}
        if set(ages)!=boxes.keys()-measured.keys():
            raise ValueError('Projected/observed output classification differs')
        offsets={}
        for tid,age in ages.items():
            h=self.history.get(tid,[])
            if len(h)!=2 or h[0][0]!=frame-age-1 or h[1][0]!=frame-age:
                self.status['nonconsecutive_history']+=1
                continue
            stamps=[available.get(f) for f in (h[0][0],h[1][0],frame)]
            if any(im is None for im in stamps):
                self.status['missing_ir']+=1
                continue
            box=boxes[tid]
            others=[b for b in measured.values() if np.all(b[:2]-4<=box[:2]+box[2:]) and np.all(b[:2]+b[2:]+4>=box[:2])]
            raw=sample(stamps,[h[0][1],h[1][1]],box,False,self.registration)
            step,reason=decision(dict(box=box,others=others),raw,self.profile,self.gain)
            self.status[reason]+=1
            if reason=='moved':offsets[tid]=step
        self.history={tid:hist for tid,hist in self.history.items() if hist[-1][0]>=frame-3}
        for tid,box in measured.items():
            history=self.history.setdefault(tid,[])
            self.history[tid]=history[-1:]+[(frame,box)]
        return offsets


class RgbFirstPosition:
    def __init__(self,policy,registration):
        self.rgb=OnlinePosition(policy,registration,"rgb")
        self.ir=OnlinePosition(policy,registration,"ir")
        self.status=Counter()

    def update(self,frame,rgb_path,prediction,events):
        visible=self.rgb.update(frame,rgb_path,prediction,events)
        thermal=self.ir.update(frame,rgb_path,prediction,events)
        merged={**thermal,**visible}
        proposed=sum(e["event"]=="motion_output" for e in events)
        self.status["rgb"]+=len(visible)
        self.status["ir"]+=len(merged)-len(visible)
        self.status["unchanged"]+=proposed-len(merged)
        return merged
