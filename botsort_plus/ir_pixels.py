"""Current-frame IR patches sampled in RGB coordinates. No identity model."""
import cv2
import numpy as np

_GRID=np.arange(28,dtype=np.float32)-13.5
_MX,_MY=np.meshgrid(_GRID,_GRID)
RING=np.ones((28,28),dtype=bool)
RING[6:22,6:22]=False
INVALID={1:'small_roi',2:'edge_roi',3:'registration_uncertain',4:'missing_ir'}

class Extractor:
    def __init__(self,settings):
        self.settings=settings
        self.mat=np.asarray(settings['matrix'],float)
        self.ext=np.asarray(settings['extent'],float)

    def sample(self,ir,box,translation,shift=False):
        if ir is None:return None,4
        box=np.asarray(box,float);size=box[2:4]-box[:2];ctr=box[:2]+size*.5
        if translation is None:
            center=self.mat[:,:2]@ctr+self.mat[:,2];size=(np.abs(self.mat[:,:2])@size)*self.ext
        else:center=ctr+translation
        w,h=size
        if min(w,h)<4:return None,1
        x,y=center
        band=int(min(w,h)>8)+int(min(w,h)>12)
        key=f'{min(3,max(0,int(x//160)))}_{min(2,max(0,int(y//171)))}_{band}'
        count,rate=self.settings['geometry_grid'].get(key,(0,0))
        if count<self.settings['cell_min_count'] or rate<self.settings['center_inside_min']:return None,3
        if shift:x+=48 if x<320 else -48
        width,height=ir.shape[1],ir.shape[0]
        if x-1.16*w<0 or y-1.16*h<0 or x+1.16*w>=width-1 or y+1.16*h>=height-1:return None,2
        patch=cv2.remap(ir,(x+_MX*(w/12)).astype(np.float32),(y+_MY*(h/12)).astype(np.float32),cv2.INTER_LINEAR)
        return patch,0


class Stamp:
    def __init__(self,patch):
        self.patch=patch
        self.template=np.ascontiguousarray(patch[6:22,6:22])
        bg=patch[RING]
        self.contrast=float(patch[10:18,10:18].mean()-np.median(bg))
        self.salience=abs(self.contrast)/(4.+.40*float(bg.std()))
        self.spread=float(self.template.std())
        self.usable=self.spread>=3.5 and abs(self.contrast)>=4 and self.salience>=.55

    def similarity(self,previous):
        # Compare one old tile to a small search window of the current tile.
        # At most a quarter of the original box dimension in either direction.
        return float(cv2.matchTemplate(self.patch[3:25,3:25],previous.template,cv2.TM_CCOEFF_NORMED).max())
