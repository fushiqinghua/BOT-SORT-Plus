"""Change the candidate set only. Retain the original causal trigger and acceptance rule."""
import cv2
import numpy as np
from frozen_gmc import IDENTITY,POLICY,cell_id,even_cells,foreground_free,heldout,inside,residual


def fit_lightglue(a,b,shape,previous_mask,mask,record):
    cells=cell_id(a,shape)
    keep=even_cells(cells)&inside(a,previous_mask)&inside(b,mask)
    a,b,cells=a[keep],b[keep],cells[keep]
    record.update(fit_matches=len(a),fit_cells=len(set(cells)))
    if len(a)<POLICY["fit_min_matches"] or len(set(cells))<POLICY["min_cells"]:
        return None
    mat,inliers=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=POLICY["ransac_px"],
                                           maxIters=1000,confidence=.99,refineIters=10)
    if mat is None or not np.isfinite(mat).all():
        return None
    n=int(inliers.sum())
    record["inliers"]=n
    scale=np.sqrt(np.linalg.det(mat[:,:2]))
    if (n>=POLICY["min_inliers"] and n>=POLICY["min_inlier_fraction"]*len(a)
        and len(set(cells[inliers.ravel()>0]))>=POLICY["min_cells"]
        and POLICY["scale_min"]<=scale<=POLICY["scale_max"]
        and np.linalg.norm(mat[:,2])<=POLICY["translation_diagonal"]*np.hypot(*shape)):
        return mat
    return None


def qualified(native, alternative):
    required=max(POLICY["min_gain_px"],POLICY["min_relative_gain"]*native[0])
    return (native[0]-alternative[0]>=required
            and alternative[1]<=native[1]+POLICY["max_p80_regression"])


class Selector:
    def __init__(self, allow_skip, matcher):
        self.allow_skip=allow_skip
        self.matcher=matcher
        self.prev=self.pmask=self.features=None

    def apply(self,image,dets,native):
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        mask=foreground_free(gray,dets)
        chosen=native
        record=dict(choice="sof",trigger=False,model_called=False)
        features=None
        if self.prev is not None:
            p,q=heldout(self.prev,gray,self.pmask,mask)
            cells=len(set(cell_id(p,gray.shape)))
            record.update(validation=len(p),validation_cells=cells)
            if len(p)>=POLICY["min_validation"] and cells>=POLICY["min_cells"]:
                native_error=residual(native,p,q)
                record["native_error"]=native_error
                if native_error[0]>POLICY["median_trigger"] or native_error[1]>POLICY["p80_trigger"]:
                    record["trigger"]=True
                    candidates={}
                    if self.allow_skip:
                        candidates["identity"]=IDENTITY
                    if self.matcher is not None:
                        record["model_called"]=True
                        with self.matcher.torch.inference_mode():
                            prev=self.features if self.features is not None else self.matcher.features(self.prev)
                            features=self.matcher.features(gray)
                            a,b=self.matcher.pairs(prev,features)
                        mat=fit_lightglue(a,b,gray.shape,self.pmask,mask,record)
                        if mat is not None:
                            candidates["lightglue"]=mat
                    errors={key:residual(mat,p,q) for key,mat in candidates.items()}
                    eligible=[key for key,score in errors.items() if qualified(native_error,score)]
                    record.update(candidate_errors=errors,eligible=eligible)
                    if eligible:
                        winner=min(eligible,key=lambda key:(errors[key][0],errors[key][1]))
                        chosen=candidates[winner]
                        record.update(choice=winner,selected_error=errors[winner])
        self.prev,self.pmask,self.features=gray,mask,features
        return chosen,record
