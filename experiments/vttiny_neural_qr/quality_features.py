"""Current-frame observation features. Coordinates and scores use original detection rows."""
import cv2
import numpy as np


FEATURES = ('score', 'log_w', 'log_h', 'visible_frac', 'max_iou', 'max_intersection_fraction',
            'rgb_contrast', 'rgb_spread', 'rgb_gradient', 'ir_contrast', 'ir_spread', 'ir_gradient')
NO_IMAGE = FEATURES[:6]


def gray(image):
    if image is None or image.dtype != np.uint8:
        raise ValueError('Expected decoded uint8 camera pixels')
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError('Expected grayscale or OpenCV BGR')


def sampled_quality(image, x, y, rx, ry):
    # Positions refer to the original sensor grid. For a two-pixel target the
    # gradients remain one native pixel wide; interpolation adds no new detail.
    def sample(sx, sy):
        return cv2.remap(image, sx, sy, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE).astype(np.float32) / 255.
    obj = sample(x, y)
    ring = sample(rx, ry)
    gx = (sample(x + 1., y) - sample(x - 1., y)) * .5
    gy = (sample(x, y + 1.) - sample(x, y - 1.)) * .5
    return np.column_stack((np.abs(obj.mean(1) - ring.mean(1)), obj.std(1),
                            np.sqrt(np.mean(gx*gx + gy*gy, axis=1))))


def frame_features(rgb, ir, dets):
    """Return (N, len(FEATURES)) in unmodified detector row order, including empty frames."""
    vis, thermal = gray(rgb), gray(ir)
    if vis.shape != thermal.shape:
        raise ValueError('Registered visible and infrared frames must have identical dimensions')
    z = np.asarray(dets, dtype=np.float32)
    if z.size == 0:
        return np.empty((0, len(FEATURES)), dtype=np.float32)
    if z.ndim != 2 or z.shape[1] < 5 or not np.isfinite(z[:, :5]).all():
        raise ValueError('Expected finite original-pixel xyxy, score detection rows')
    lo, hi, score = z[:, :2], z[:, 2:4], z[:, 4]
    wh = hi-lo
    if np.any(wh <= 0.) or not np.all((score >= 0.) & (score <= 1.)):
        raise ValueError('Invalid detection size or confidence')
    center = (lo+hi)*.5
    h,w = vis.shape
    area=wh.prod(1)
    visible = np.maximum(0., np.minimum(hi, [w,h])-np.maximum(lo, 0.)).prod(1)/area
    inter = np.maximum(0., np.minimum(hi[:,None],hi)-np.maximum(lo[:,None],lo)).prod(2)
    # The tracker's low-score cutoff is 0.10. Detector exports may include
    # rows below it; those rows cannot act as observed rivals.
    np.fill_diagonal(inter, 0.)
    inter[:,score <= .10]=0.
    iou = inter/np.maximum(area[:,None]+area-inter, 1.e-12)
    base = np.column_stack((score, np.log(wh), visible, iou.max(1), (inter/area[:,None]).max(1)))
    grid = np.linspace(-.4,.4,5,dtype=np.float32)
    gx,gy = np.meshgrid(grid,grid)
    ring=np.linspace(-.75,.75,7,dtype=np.float32)
    rx,ry=np.meshgrid(ring,ring)
    rim=(np.abs(rx)>=.75)|(np.abs(ry)>=.75)
    def xy(x, y):
        return center[:,0,None] + wh[:,0,None]*x.ravel(), center[:,1,None] + wh[:,1,None]*y.ravel()
    x,y=xy(gx,gy)
    xx,yy=xy(rx[rim],ry[rim])
    return np.column_stack((base, sampled_quality(vis,x,y,xx,yy),
                            sampled_quality(thermal,x,y,xx,yy))).astype(np.float32)
