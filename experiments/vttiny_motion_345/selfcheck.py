"""Small geometric and Kalman checks for the research extensions in the pinned redetr environment."""
from types import SimpleNamespace

import cv2
import numpy as np
from boxmot.utils.matching import iou_distance
from occlusion import occlusion_fractions, union_area
from perspective import Registration, accepted_matrix, association_cost, project, project_boxes
from process_noise import UNIT_Q, features, variance
from settings import RESULT, save


def geometry():
    tr = SimpleNamespace(xyxy=np.array([8.,12.,45.,48.]))
    de = SimpleNamespace(xyxy=np.array([12.,15.,38.,43.]))
    def cost(x,y):
        def boxes(items):
            return np.array([p.xyxy if hasattr(p,'xyxy') else p for p in items])
        return iou_distance(boxes(x),boxes(y))
    old={id(tr):tr.xyxy}
    np.testing.assert_array_equal(association_cost(cost,[tr],[de],old,None,None),cost([tr],[de]))
    shift=np.array([[1.,0.,3.],[0.,1.,-2.],[0.,0.,1.]])
    warped,valid=project_boxes([tr.xyxy],shift)
    assert valid.all()
    np.testing.assert_allclose(warped[0],tr.xyxy+[3,-2,3,-2])
    def costs(m):
        f=association_cost(cost,[tr],[de],old,m,None)
        s=association_cost(cost,[tr],[de],old,m,np.linalg.inv(m))
        return f,s
    f,s=costs(shift)
    np.testing.assert_allclose(f,s,atol=1e-7)
    h=np.array([[1.,.08,4.],[.03,1.,1.],[.002,-.001,1.]])
    f,s=costs(h)
    backward,_=project_boxes([de.xyxy],np.linalg.inv(h))
    np.testing.assert_allclose(s,(f+cost([tr],backward))/2)
    assert abs(float(f[0,0]-s[0,0])) > 1.e-4
    horizon=np.eye(3);horizon[2,0]=-.05
    assert not project_boxes([tr.xyxy],horizon)[1][0]
    np.testing.assert_allclose(association_cost(cost,[tr],[de],old,horizon,None),cost([tr],[de]))
    grid=np.array([[x,y] for y in (5.,15.,50.,95.) for x in (5.,20.,55.,95.)])
    assert accepted_matrix(shift,np.ones((16,1)),grid,np.arange(16),(100,100)) is not None
    assert accepted_matrix(horizon,np.ones((16,1)),grid,np.arange(16),(100,100)) is None


def causality():
    rng=np.random.default_rng(142)
    start=rng.integers(0,256,(240,320,3),dtype=np.uint8)
    affine=np.float32([[1.,0.,1.8],[0.,1.,.8]])
    moved=cv2.warpAffine(start,affine,(320,240))
    def replay(frames):
        cv2.setRNGSeed(0); reg=Registration()
        return [reg.step(im,np.empty((0,6)),np.eye(2,3),True) for im in frames]
    a=replay([start,moved,start])
    b=replay([start,moved,np.zeros_like(start)])
    assert a[0][0] is None and a[0][1]=={'choice':'native'}
    assert a[1][1]==b[1][1] and a[1][0] is not None
    np.testing.assert_allclose(a[1][0],b[1][0])


def overlap_and_noise():
    boxes=[[0,0,10,10],[0,0,5,20],[0,0,5,18]]
    area,bottom=occlusion_fractions(boxes)
    assert area[0]==bottom[0]==.5 and bottom[1]==0.
    assert union_area(np.array([[0,0,2,4],[0,0,2,4],[1,0,3,4]]))==12.
    area,bottom=occlusion_fractions([[0,0,6,6],[10,10,15,15]])
    assert not area.any() and not bottom.any()
    assert np.linalg.eigvalsh(UNIT_Q).min() >= -1.e-12 and np.linalg.matrix_rank(UNIT_Q)==2
    assert not UNIT_Q[[2,3,6,7],:].any()
    slow=features(np.array([1.,2.,15.,12.,0.,0.,0.,0.]))
    fast=features(np.array([1.,2.,15.,12.,4.,2.,0.,0.]))
    assert variance(slow,[-2,1,0]) < variance(fast,[-2,1,0])
    assert variance(slow,[-2,0,0]) == variance(fast,[-2,0,0])
    assert 0 < variance(fast,[-20,0,0]) and np.isclose(variance(fast,[20,0,0]),16.)


if __name__=='__main__':
    geometry()
    causality()
    overlap_and_noise()
    save(RESULT/'selfcheck.json',dict(status='PASS',checks=['projective_geometry','fallback','bidirectional_cost',
         'causal_prefix','overlap_union','bottom_order','positive_semidefinite_position_velocity_Q','variance_features']))
    print('Synthetic checks passed')
