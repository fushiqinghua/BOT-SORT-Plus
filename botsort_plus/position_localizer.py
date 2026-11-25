"""Estimate translations from two past measured patches and one current frame."""
import cv2
import numpy as np

# Specified before reading target labels. Offsets are in fractions of the current box.
PROFILES={
    'permissive':dict(corr=.50,lift=.018,margin=.012,agree=3.5,history=.25,salience=.40),
    'balanced':dict(corr=.62,lift=.035,margin=.025,agree=2.5,history=.40,salience=.65),
    'strict':dict(corr=.72,lift=.055,margin=.045,agree=2.0,history=.55,salience=.85)
}
GAINS=(.5,.85)
MAX_NORM=3.5
MAX_SHORT_FRACTION=.38

def center(box):
    return box[:2]+box[2:]*.5

def ious(box, others):
    others=np.asarray(others,dtype=float).reshape(-1,4)
    inter=np.maximum(0,np.minimum(box[:2]+box[2:],others[:,:2]+others[:,2:])-np.maximum(box[:2],others[:,:2])).prod(axis=1)
    return inter/np.maximum(1e-12,np.prod(box[2:])+others[:,2:].prod(axis=1)-inter)

def roi(image, box, radius, shift):
    wh=box[2:]
    c=center(box)+shift
    edges=(10+radius)*wh/16
    if np.any(c-edges<0) or np.any(c+edges>=np.array(image.shape[::-1])-1):
        return None
    q=np.arange(-10-radius,11+radius,dtype=np.float32)
    x,y=np.meshgrid((c[0]+q*wh[0]/16).astype(np.float32),(c[1]+q*wh[1]/16).astype(np.float32))
    return cv2.remap(image,x,y,cv2.INTER_LINEAR).astype(np.float32)

def quality(t):
    core=t[6:15,6:15]
    ring=np.r_[t[:3].ravel(),t[-3:].ravel(),t[3:-3,:3].ravel(),t[3:-3,-3:].ravel()]
    contrast=float(np.mean(core)-np.median(ring))
    noise=3.+.5*float(np.std(ring))
    return contrast,abs(contrast)/noise

def sample(images,prior,box,shifted,registration):
    wh=box[2:]
    if min(wh)<4 or any(min(b[2:])<4 for b in prior):
        return dict(reason='small_box')
    if any(np.max(abs(np.log(wh/b[2:])))>.5 for b in prior):
        return dict(reason='scale_change')
    c=center(box)
    band=int(min(wh)>8)+int(min(wh)>12)
    cell=f'{int(np.clip(c[0]//160,0,3))}_{int(np.clip(c[1]//171,0,2))}_{band}'
    count,hit=registration['geometry_grid'].get(cell,(0,0))
    if count<registration['cell_min_count'] or hit<registration['center_inside_min']:
        return dict(reason='unreliable_region')
    shift=np.array([(48 if c[0]<320 else -48) if shifted else 0,0])
    old=[roi(im,b,0,shift) for im,b in zip(images[:2],prior)]
    current=roi(images[2],box,7,shift)
    if current is None or any(t is None for t in old):
        return dict(reason='edge')
    qa,qb=map(quality,old)
    if min(float(t.std()) for t in old)<3 or min(abs(qa[0]),abs(qb[0]))<2 or qa[0]*qb[0]<=0:
        return dict(reason='weak_or_inconsistent_contrast')
    history=float(cv2.matchTemplate(old[0],old[1],cv2.TM_CCOEFF_NORMED)[0,0])
    maps=[cv2.matchTemplate(current,t,cv2.TM_CCOEFF_NORMED) for t in old]
    if not all(np.isfinite(m).all() for m in maps):
        return dict(reason='invalid_texture')
    surface=.5*(maps[0]+maps[1])
    y,x=np.unravel_index(np.argmax(surface),surface.shape)
    if x in (0,14) or y in (0,14):
        return dict(reason='peak_at_boundary')
    nearby=np.ones(surface.shape,bool)
    nearby[max(0,y-2):y+3,max(0,x-2):x+3]=False
    peak=float(surface[y,x])
    def subpixel(a,b,c):
        curv=2*b-a-c
        return float(np.clip((c-a)/(2*curv),-.5,.5)) if curv>1e-6 else 0.
    dx=x-7+subpixel(surface[y,x-1],peak,surface[y,x+1])
    dy=y-7+subpixel(surface[y-1,x],peak,surface[y+1,x])
    peaks=[np.asarray(np.unravel_index(np.argmax(m),m.shape)) for m in maps]
    return dict(reason='candidate',dx=float(dx*wh[0]/16),dy=float(dy*wh[1]/16),
          corr=min(float(m[y,x]) for m in maps),lift=peak-float(surface[7,7]),margin=peak-float(surface[nearby].max()),
          agree=float(np.linalg.norm(peaks[0]-peaks[1])),history=history,salience=min(qa[1],qb[1]))

def decision(item,raw,profile,gain):
    if raw['reason']!='candidate':
        return (0.,0.),raw['reason']
    limits=PROFILES[profile]
    for key in ('history','salience','corr','lift','margin'):
        if raw[key]<limits[key]:
            return (0.,0.),key
    if raw['agree']>limits['agree']:
        return (0.,0.),'history_disagrees'
    box=np.asarray(item['box'],float)
    step=np.array([raw['dx'],raw['dy']],float)
    length=float(np.linalg.norm(step))
    if length<.45:
        return (0.,0.),'small_step'
    cap=min(MAX_NORM,MAX_SHORT_FRACTION*min(box[2:]))
    step*=min(1.,cap/length)*gain
    other=np.asarray(item['others'],float).reshape(-1,4)
    moved=box.copy();moved[:2]+=step
    if len(other):
        a=ious(box,other);b=ious(moved,other)
        oldc=center(box);newc=center(moved)
        before=((oldc>=other[:,:2])&(oldc<=other[:,:2]+other[:,2:])).all(axis=1)
        after=((newc>=other[:,:2])&(newc<=other[:,:2]+other[:,2:])).all(axis=1)
        if np.any((b>.22)&(b>a+.04)) or np.any(after&~before):
            return (0.,0.),'measured_neighbor'
    return tuple(map(float,step)),'moved'
