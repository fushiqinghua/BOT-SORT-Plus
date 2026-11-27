"""Causal background registration and symmetric two-view box association in native RGB pixels."""
import cv2
import numpy as np


def project(points, matrix):
    points = np.asarray(points, dtype=float)
    denominator = points @ matrix[2,:2] + matrix[2,2]
    good = np.isfinite(denominator) & (denominator > .1)
    values = (points @ matrix[:2,:2].T + matrix[:2,2]) / np.where(good,denominator,1.)[...,None]
    return values, good & np.isfinite(values).all(axis=-1)


def project_boxes(boxes, matrix):
    b=np.asarray(boxes,float).reshape(-1,4)
    corners=np.stack([b[:,[0,1]],b[:,[2,1]],b[:,[2,3]],b[:,[0,3]]],axis=1)
    mapped,valid=project(corners,matrix)
    result=np.column_stack([mapped.min(axis=1),mapped.max(axis=1)])
    return result, valid.all(axis=1)&((result[:,2:]-result[:,:2])>0).all(axis=1)


def accepted_matrix(matrix, inliers, points, cells, shape):
    if matrix is None or inliers is None or not np.isfinite(matrix).all() or abs(matrix[2,2])<1e-8:
        return None
    m=matrix/matrix[2,2]; take=inliers.ravel().astype(bool)
    height,width=shape
    if take.sum()<12 or take.mean()<.5 or len(set(cells[take]))<4:return None
    hull=cv2.convexHull(points[take].astype(np.float32))
    if cv2.contourArea(hull)<.08*height*width:return None
    corners=np.array([[0,0],[width,0],[width,height],[0,height]],float)
    mapped,good=project(corners,m)
    if not good.all() or np.max(np.linalg.norm(mapped-corners,axis=1))>.25*np.hypot(width,height):return None
    for xy in corners:
        probes=np.array([xy,xy+[1,0],xy+[0,1]])
        p,ok=project(probes,m); jac=np.stack((p[1]-p[0],p[2]-p[0]),axis=1)
        if not ok.all() or np.linalg.det(jac)<=0:return None
        scales=np.linalg.svd(jac,compute_uv=False)
        if scales.min()<.75 or scales.max()>1.33:return None
    return m


def frame_input(image, detections):
    gray=cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),None,fx=.5,fy=.5,interpolation=cv2.INTER_AREA)
    mask=np.full(gray.shape,255,np.uint8)
    for x1,y1,x2,y2,score,*_ in detections:
        if score>.1:cv2.rectangle(mask,(int(x1/2)-1,int(y1/2)-1),(int(np.ceil(x2/2))+1,int(np.ceil(y2/2))+1),0,-1)
    return gray,mask


def correspondences(previous,current):
    a,ma=previous; b,mb=current; h,w=a.shape
    pts=[]; strata=[]
    for i in range(16):
        row,col=divmod(i,4); x=col*w//4; y=row*h//4
        roi=np.s_[y:(row+1)*h//4,x:(col+1)*w//4]
        p=cv2.goodFeaturesToTrack(a[roi],16,.012,3,mask=ma[roi])
        if p is not None:
            pts.append(p.reshape(-1,2)+np.float32([x,y]));strata.extend([i]*len(p))
    if not pts:return np.empty((0,2)),np.empty((0,2)),np.empty(0,int)
    p=np.concatenate(pts); kw=dict(winSize=(15,15),maxLevel=2)
    q,ok,_=cv2.calcOpticalFlowPyrLK(a,b,p,None,**kw)
    back,ok2,_=cv2.calcOpticalFlowPyrLK(b,a,q,None,**kw)
    j=np.rint(np.nan_to_num(q)).astype(int)
    use=(ok.ravel()>0)&(ok2.ravel()>0)&(np.linalg.norm(back-p,axis=1)<=.5)
    use&=(j[:,0]>=0)&(j[:,0]<w)&(j[:,1]>=0)&(j[:,1]<h)&(mb[j[:,1].clip(0,h-1),j[:,0].clip(0,w-1)]>0)
    return 2*p[use],2*q[use],np.asarray(strata)[use]


def improves(a,b):
    return b[0]-a[0]>=max(.35,.20*b[0]) and a[1]<=b[1]+.25


class Registration:
    def __init__(self):self.previous=None

    def step(self,image,dets,native,allow_h):
        curr=frame_input(image,dets); prev=self.previous; self.previous=curr
        record={'choice':'native'}
        if prev is None:return None,record
        p,q,cells=correspondences(prev,curr)
        train=(cells//4+cells%4)%2==0; valid=~train
        if min(train.sum(),valid.sum())<12 or min(len(set(cells[train])),len(set(cells[valid])))<4:return None,record
        ps,qs=p[train],q[train]; pv,qv=p[valid],q[valid]
        shape=image.shape[:2]; native=np.vstack([native,[0.,0.,1.]])
        def errors(m):
            xy,_=project(pv,m)
            return np.quantile(np.linalg.norm(xy-qv,axis=1),[.5,.8]).tolist()
        best=native; err=errors(native); record['native_error']=err
        a,in_a=cv2.estimateAffine2D(ps,qs,method=cv2.RANSAC,ransacReprojThreshold=2.,maxIters=1200,confidence=.99,refineIters=10)
        affine=accepted_matrix(None if a is None else np.vstack([a,[0.,0.,1.]]),in_a,ps,cells[train],shape)
        if affine is not None:
            ae=errors(affine); record['affine_error']=ae
            if improves(ae,err):best,err=affine,ae;record['choice']='affine'
        if allow_h:
            h,in_h=cv2.findHomography(ps,qs,cv2.RANSAC,2.,maxIters=1200,confidence=.99)
            hom=accepted_matrix(h,in_h,ps,cells[train],shape)
            if hom is not None:
                he=errors(hom);record['homography_error']=he
                if improves(he,err):best,err=hom,he;record['choice']='homography'
        if record['choice']=='homography':
            # Signed, held-out residuals by quadrant for the affine-vs-perspective diagnosis.
            quad=(pv[:,0]>=shape[1]/2).astype(int)+2*(pv[:,1]>=shape[0]/2)
            for key,m in [('reference',affine if affine is not None else native),('chosen',best)]:
                delta=project(pv,m)[0]-qv
                record[key+'_quadrants']=[np.median(delta[quad==i],axis=0).tolist() if np.any(quad==i) else None for i in range(4)]
        return (None if record['choice']=='native' else best),record


def association_cost(original,tracks,detections,old_boxes,matrix,inverse):
    base=original(tracks,detections)
    if matrix is None or not base.size:return base
    old=np.array([old_boxes[id(t)] for t in tracks]); boxes=np.array([d.xyxy for d in detections])
    forward,valid=project_boxes(old,matrix)
    cost=original(forward,boxes)
    usable=np.broadcast_to(valid[:,None],base.shape)
    if inverse is not None:
        backward,ok=project_boxes(boxes,inverse)
        cost=(cost+original(old,backward))*.5
        usable=usable&ok[None,:]
    return np.where(usable,cost,base)
