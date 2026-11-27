"""Live RGB GMC and frozen detection/ReID. No GT, sequence or frame gates are used by either fix."""
from pathlib import Path
import os
import sys
import cv2
import torch
import numpy as np
from common_audit import ROOT,FACTOR,REPO,SETTINGS,SEQ,SKIP,FULL,read,write,sha
from guard import LowMatchGuard,decide,dual_low,capture_priors,measured_centers,MARGIN,iou
from frozen_gmc import Matcher,IDENTITY
from lost_camera_margin import LostCameraMargin
from gmc_variants import Selector
import botsort_plus.native_adapter as adapter
import run_botsort_plus as original


def main(method, stage, *, output_dir=None, settings_file=SETTINGS, repository=REPO,
         reference_file=None, sequences=None):
    cv2.setNumThreads(1);torch.set_num_threads(1);torch.manual_seed(0)
    if method!=SKIP:
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
    root = Path(output_dir) if output_dir is not None else ROOT / stage
    settings = read(settings_file)
    config = read(repository / "configs/botsort_plus/B_early_moderate.json")
    if sequences is None:
        sequences = [SEQ] if stage == "focus" else []
    trial=os.environ["BG_LOST_POLICY"]
    if trial not in ("margin_both","evidence_both"):
        raise ValueError(trial)
    config["name"]=trial
    cfg_file=root/"configs"/(method+".json");write(cfg_file,config)
    args=original.parser().parse_args(["--config",str(cfg_file),"--baseline-root",settings["baseline_root"],
        "--features-dir",settings["features_dir"],"--images-root",settings["images_root"],
        "--runtime-root",settings["runtime_root"],"--output-dir",str(root)]+(["--sequences", *sequences] if sequences else []))
    original_prepare=original.prepare
    def prepare(parsed):
        cfg,paths,protocol=original_prepare(parsed)
        assert cfg.adapter_mode == "v2" and cfg.recovery.enabled and cfg.recovery.early_evidence == "moderate"
        assert method == "dual_low"
        prior=read(reference_file or FACTOR/"full/protocols"/(FULL+".json"))
        assert all(sha(FACTOR/k)==prior["gmc_factorial"]["source_sha256"][k] for k in ("frozen_gmc.py","gmc_variants.py"))
        matcher=Matcher() if method!=SKIP else None
        sequence=iter(paths)
        protocol["case_fix"]=dict(method=method,kalman="original",dual_geometry=dual_low(method),
            high_center_frames=2 if method=="dual_low_center" else 0,
            low_iou_native=.5,alternative_min_gain=MARGIN if method=="pixel_margin" else None,
            patch_sha256={k:sha(ROOT/k) for k in ("guard.py","live_arm.py")},
            model=prior["gmc_factorial"]["model"] if matcher else None)
        protocol["lost_camera_policy"]=dict(variant=trial,source_sha256=sha(ROOT/"lost_camera_margin.py"),
            gap_window=cfg.recovery.window,margin=cfg.recovery.ambiguity_margin)
        class Live(LowMatchGuard,adapter.BoTSORTPlus):
            def __init__(self,**kwargs):
                super().__init__(**kwargs)
                self.seq=next(sequence);self.selector=Selector(True,matcher);self.case_policy=method
                self.capture_trace=False;self.original_camera=self.cmc.apply;self.cmc.apply=self.warp
                self.last_lg=-10;self.center_count=0;self.center_frames=0;self.lg_frames=0
                self.blocked_native_edges = 0
                self.margin=LostCameraMargin(self,require_observation=trial=="evidence_both")
                self.recovery_matcher._pair=self.margin.recovery_pair(self.recovery_matcher._pair)
            def _cost(self,tracks,detections,unconfirmed=False):
                cost=super()._cost(tracks,detections,unconfirmed)
                if not unconfirmed: self.margin.high_cost(cost,tracks)
                return cost
            def guard_low_cost(self, cost, tracks, detections):
                if self.matched_camera:
                    for i, track in enumerate(tracks):
                        backup = self.backup_boxes[int(track.id)]
                        for j, detection in enumerate(detections):
                            if cost[i,j] <= .5 and iou(backup,detection[:4]) < .5:
                                cost[i,j] = 1.
                                self.blocked_native_edges += 1
            def warp(self,image,dets):
                native=self.original_camera(image,dets)
                full,record=self.selector.apply(image,dets,native)
                fallback=IDENTITY if "identity" in record.get("eligible",[]) else native
                out=decide(method,full,fallback,record)
                self.margin.observe(self.frame_count,native,out)
                self.matched_camera=record["choice"]=="lightglue" and not np.array_equal(out,fallback)
                if self.matched_camera:
                    self.last_lg=self.frame_count;self.lg_frames+=1
                    if dual_low(method):
                        capture_priors(self,out,fallback)
                return out
            def update(self,dets,img,*args,**kwargs):
                rows=super().update(dets,img,*args,**kwargs)
                if method=="dual_low_center" and self.frame_count-self.last_lg<=1:
                    rows,n=measured_centers(rows,dets,self.track_high_thresh)
                    self.center_count+=n;self.center_frames+=int(n>0)
                return rows
            def finish_sequence_events(self):
                write(root/"actions"/method/(self.seq+".json"),dict(lightglue_frames=self.lg_frames,
                          center_outputs=self.center_count,center_frames=self.center_frames,
                          blocked_native_edges=self.blocked_native_edges,recovery_pair_rejections=self.margin.recovery_rejections,
                          high_pair_rejections=self.margin.high_rejections,unsupported_rejections=self.margin.unsupported_rejections))
                return super().finish_sequence_events()
        adapter.BoTSORTPlus=Live
        return cfg,paths,protocol
    original.prepare=prepare
    return original.run(args)


if __name__=="__main__":
    raise SystemExit(main(*sys.argv[1:]))
