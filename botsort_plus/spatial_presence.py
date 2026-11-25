"""Veto only with positive IR evidence of a displaced response of the same polarity."""
import cv2
import numpy as np

_GRID=np.arange(44,dtype=np.float32)-21.5
_MX,_MY=np.meshgrid(_GRID,_GRID)
def decide(observer,tid,box,others):
    stamp,reason=observer.prediction_stamp(box)
    if stamp is None:return True,reason,None
    hist=observer.past.get(tid)
    if not hist or len(hist)<2:return True,'short_history',None
    (f0,p),(f1,q)=hist
    if (p is None or q is None or f1-f0!=1 or not 1<=observer.frame-f1<=2 or
            not (p.usable and q.usable) or p.contrast*q.contrast<=0 or min(p.salience,q.salience)<.8):
        return True,'weak_history',None
    xy=np.asarray(box[:2]);extent=np.asarray(box[2:])-xy;center=xy+extent*.5
    delta=observer.motion[observer.frame-1]
    if np.isfinite(delta).all():
        c=center+delta;wh=extent
    else:
        mat=observer.extractor.mat;c=mat[:,:2]@center+mat[:,2];wh=(np.abs(mat[:,:2])@extent)*observer.extractor.ext
    if observer.shift:c=c+np.array([48 if c[0]<320 else -48,0])
    image=observer.current_image
    if np.any(c-1.85*wh<0) or np.any(c+1.85*wh>=np.asarray(image.shape[::-1])-1):return True,'edge_roi',None
    tile=cv2.remap(image,(c[0]+_MX*wh[0]/12).astype(np.float32),(c[1]+_MY*wh[1]/12).astype(np.float32),cv2.INTER_LINEAR).astype(np.float32)
    ring=np.r_[tile[:4].ravel(),tile[-4:].ravel(),tile[4:-4,:4].ravel(),tile[4:-4,-4:].ravel()]
    background=float(np.median(ring));noise=max(3.,float(np.median(abs(ring-background))*1.4826))
    sign=np.sign(q.contrast)
    strength=sign*(cv2.blur(tile,(7,7))-background)
    near=(np.abs(_MX)<=5)&(np.abs(_MY)<=5)
    far=(np.maximum(abs(_MX),abs(_MY))>=9)&(np.maximum(abs(_MX),abs(_MY))<=16)
    inner=float(strength[near].max());outer=float(strength[far].max())
    peak_idx=np.argmax(np.where(far,strength,-np.inf));i,j=np.unravel_index(peak_idx,strength.shape)
    reference=min(abs(p.contrast),abs(q.contrast))
    if max(inner,outer)<max(8,2*noise):return True,'low_local_contrast',None
    if inner>=max(.55*reference,2*noise):return True,'support',None
    if outer<max(8.,2*noise,.60*reference) or outer-inner<max(6.,.45*reference):return True,'inconclusive',None
    # A nearby measured object can explain the off-centre response.
    shift=np.array([_MX[i,j]*extent[0]/12,_MY[i,j]*extent[1]/12])
    peak=center+shift
    for ob in others:
        if np.all(ob[:2]-.15*extent<=peak) and np.all(peak<=ob[2:]+.15*extent):
            return True,'occupied_neighbor',None
    # An extended hot/cold contour is ambiguous. Require a separated peak.
    rival=far & (np.maximum(abs(_MX-_MX[i,j]),abs(_MY-_MY[i,j]))>=7)
    if np.any(rival) and outer-float(strength[rival].max())<max(3.,.6*noise):return True,'diffuse_neighbor',None
    return False,'displaced_response',{'inner_response':inner,'outer_response':outer,'offset_xy':shift.tolist()}
