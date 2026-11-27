"""Causal RGB background motion selection. Distances are in native image pixels."""
import cv2
import numpy as np

IDENTITY = np.eye(2, 3, dtype=np.float64)
POLICY = dict(grid=4, corners_per_cell=12, min_validation=12, min_cells=4, fb_max=1.,
              median_trigger=1., p80_trigger=2., fit_min_matches=12, min_inliers=10,
              min_inlier_fraction=.50, ransac_px=2., min_gain_px=.35, min_relative_gain=.20,
              max_p80_regression=.25, scale_min=.85, scale_max=1.18, translation_diagonal=.25,
              max_keypoints=512, box_padding=2, validation_scale=.5)


def cell_id(points, shape):
    h, w = shape
    return (np.clip((points[:, 0]*4/w).astype(int), 0, 3) +
            4*np.clip((points[:, 1]*4/h).astype(int), 0, 3))


def even_cells(ids):
    return (ids//4 + ids%4) % 2 == 0


def foreground_free(gray, dets):
    h, w = gray.shape
    mask = np.full((h, w), 255, np.uint8)
    for d in dets:
        x, y = np.floor(d[:2]).astype(int)-2
        u, v = np.ceil(d[2:4]).astype(int)+2
        mask[max(0,y):min(h,v+1),max(0,x):min(w,u+1)] = 0
    return mask


def inside(points, mask):
    h, w = mask.shape
    x, y = np.rint(points).astype(int).T
    return ((x>=0)&(y>=0)&(x<w)&(y<h)&(mask[np.clip(y,0,h-1),np.clip(x,0,w-1)]>0))


def heldout(prev, curr, pmask, cmask):
    a = cv2.resize(prev, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
    b = cv2.resize(curr, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(pmask,(a.shape[1],a.shape[0]),interpolation=cv2.INTER_NEAREST)
    h, w = a.shape
    seeds = []
    for row in range(4):
        for col in range(4):
            if (row+col)%2 == 0:
                continue
            x,y=col*w//4,row*h//4
            roi=np.s_[y:(row+1)*h//4,x:(col+1)*w//4]
            pts=cv2.goodFeaturesToTrack(a[roi],12,.015,4,mask=mask[roi])
            if pts is not None:
                seeds.append(pts.reshape(-1,2)+np.float32([x,y]))
    if not seeds:
        return np.empty((0,2)),np.empty((0,2))
    p=np.concatenate(seeds).reshape(-1,1,2)
    kwargs=dict(winSize=(15,15),maxLevel=2)
    q,ok,_=cv2.calcOpticalFlowPyrLK(a,b,p,None,**kwargs)
    r,back,_=cv2.calcOpticalFlowPyrLK(b,a,q,None,**kwargs)
    p,q,r=[2*v.reshape(-1,2) for v in (p,q,r)]
    good=(ok.ravel()>0)&(back.ravel()>0)&(np.linalg.norm(p-r,axis=1)<=1.)&inside(q,cmask)&inside(p,pmask)
    return p[good],q[good]


def residual(matrix, p, q):
    e=np.linalg.norm(p@matrix[:,:2].T+matrix[:,2]-q,axis=1)
    return [float(np.median(e)),float(np.quantile(e,.8))]


class Matcher:
    def __init__(self):
        import torch
        from lightglue import LightGlue,SuperPoint
        assert torch.cuda.is_available(), "LightGlue experiment requires allocated GPU"
        self.torch=torch
        self.extractor=SuperPoint(max_num_keypoints=POLICY["max_keypoints"]).eval().cuda()
        self.matcher=LightGlue(features="superpoint",flash=False).eval().cuda()

    def features(self, gray):
        image=self.torch.as_tensor(gray,device="cuda").float()[None,None]/255.
        return self.extractor.extract(image, resize=None)

    def pairs(self, a, b):
        pairs=self.matcher(dict(image0=a,image1=b))["matches"][0]
        return (a["keypoints"][0][pairs[:,0]].cpu().numpy(),
                b["keypoints"][0][pairs[:,1]].cpu().numpy())


class Adaptive:
    def __init__(self, matcher):
        self.matcher=matcher
        self.prev=self.pmask=self.features=None

    def apply(self, image, dets, native):
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        mask=foreground_free(gray,dets)
        record=dict(choice="sof",trigger=False)
        chosen=native
        features=None
        if self.prev is not None:
            p,q=heldout(self.prev,gray,self.pmask,mask)
            count=len(p)
            cells=len(set(cell_id(p,gray.shape)))
            record.update(validation=count,validation_cells=cells)
            if count>=12 and cells>=4:
                quality=residual(native,p,q)
                record["native_error"]=quality
                if quality[0]>1. or quality[1]>2.:
                    record["trigger"]=True
                    candidates=[("identity",IDENTITY,residual(IDENTITY,p,q))]
                    with self.matcher.torch.inference_mode():
                        prev=self.features if self.features is not None else self.matcher.features(self.prev)
                        features=self.matcher.features(gray)
                        a,b=self.matcher.pairs(prev,features)
                    cells=cell_id(a,gray.shape)
                    good=even_cells(cells)&inside(a,self.pmask)&inside(b,mask)
                    a,b,cells=a[good],b[good],cells[good]
                    record.update(fit_matches=len(a),fit_cells=len(set(cells)))
                    if len(a)>=12 and len(set(cells))>=4:
                        mat,inliers=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,
                             ransacReprojThreshold=2.,maxIters=1000,confidence=.99,refineIters=10)
                        if mat is not None and np.isfinite(mat).all():
                            count_in=int(inliers.sum()); record["inliers"]=count_in
                            scale=np.sqrt(np.linalg.det(mat[:,:2]))
                            if (count_in>=10 and count_in>=.5*len(a) and
                                len(set(cells[inliers.ravel()>0]))>=4 and .85<=scale<=1.18 and
                                np.linalg.norm(mat[:,2])<=.25*np.hypot(*gray.shape)):
                                candidates.append(("lightglue",mat,residual(mat,p,q)))
                    record["candidate_errors"]={name:score for name,_,score in candidates}
                    qualified=[c for c in candidates if quality[0]-c[2][0]>=max(.35,.2*quality[0])
                               and c[2][1]<=quality[1]+.25]
                    if qualified:
                        name,chosen,score=min(qualified,key=lambda c:(c[2][0],c[2][1]))
                        record.update(choice=name,selected_error=score)
        self.prev,self.pmask,self.features=gray,mask,features
        return chosen,record
