"""Union of overlapping observations relative to the observed target's area."""
import numpy as np


def union_area(rectangles):
    if not len(rectangles):
        return 0.
    if len(rectangles)==1:
        r=rectangles[0]
        return (r[2]-r[0])*(r[3]-r[1])
    xs=np.unique(rectangles[:,[0,2]])
    area=0.
    for left,right in zip(xs[:-1],xs[1:]):
        cover=rectangles[(rectangles[:,0]<right)&(rectangles[:,2]>left)]
        end=-np.inf
        for y0,y1 in sorted(zip(cover[:,1],cover[:,3])):
            area+=(right-left)*max(0.,y1-max(end,y0))
            end=max(end,y1)
    return area


def occlusion_fractions(boxes):
    boxes=np.asarray(boxes,float).reshape(-1,4)
    n=len(boxes)
    all_overlap, bottom=np.zeros(n),np.zeros(n)
    mins=np.maximum(boxes[:,None,:2],boxes[None,:,:2]); maxs=np.minimum(boxes[:,None,2:],boxes[None,:,2:])
    mask=(maxs>mins).all(2); np.fill_diagonal(mask,False)
    for i in np.flatnonzero(mask.any(1)):
        take=mask[i]; rectangles=np.column_stack((mins[i,take],maxs[i,take]))
        area=np.prod(boxes[i,2:]-boxes[i,:2])
        all_overlap[i]=union_area(rectangles)/area
        bottom[i]=union_area(rectangles[boxes[take,3] > boxes[i,3]+5.])/area
    return np.clip(all_overlap,0.,1.),np.clip(bottom,0.,1.)
