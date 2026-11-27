import gzip
import json
import sys
from settings_factor import ROOT,REPO,PREVIOUS,VENDOR,SETTINGS,FOCUS,ARMS,load,save,sha


def main(method,stage):
    import cv2
    import torch
    import run_botsort_plus as runner
    import botsort_plus.native_adapter as adapter
    from frozen_gmc import Matcher,POLICY
    from gmc_variants import Selector
    skip,use_lg=ARMS[method]
    settings=load(SETTINGS)
    output=ROOT/stage
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    if use_lg:
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
    cfg=load(REPO/"configs/botsort_plus/12_baseline_ref.json")
    cfg["name"]=method
    filename=output/"configs"/(method+".json")
    save(filename,cfg)
    args=runner.parser().parse_args(["--config",str(filename),"--baseline-root",settings["baseline_root"],
        "--features-dir",settings["features_dir"],"--images-root",settings["images_root"],
        "--runtime-root",settings["runtime_root"],"--output-dir",str(output)]+
        (["--sequences",*FOCUS] if stage=="smoke" else []))
    prepare=runner.prepare

    def configure(parsed):
        config,paths,protocol=prepare(parsed)
        assert not config.enabled and protocol["native_config"]["cmc_method"]=="sof" and protocol["native_config"]["with_reid"]
        assert sha(ROOT/"frozen_gmc.py")==sha(PREVIOUS/"adaptive_gmc.py")
        sequence=iter(paths)
        matcher=Matcher() if use_lg else None
        protocol["gmc_factorial"]=dict(skip=skip, lightglue=use_lg, kalman="original_boxmot",
            past_outputs_modified=False, policy=POLICY,source_sha256={p:sha(ROOT/p) for p in
            ("frozen_gmc.py","gmc_variants.py","replay_factor.py","settings_factor.py")},
            model=load(VENDOR/"provenance.json") if use_lg else None)

        class Tracker(adapter.BoTSORTPlus):
            def __init__(self,**kwargs):
                super().__init__(**kwargs)
                self.selector=Selector(skip,matcher) if skip or use_lg else None
                self.original_apply=self.cmc.apply
                self.cmc.apply=self.camera_apply
                self.n_warps=0
                name=next(sequence)
                file=output/"warps"/method/(name+".jsonl.gz")
                file.parent.mkdir(parents=True,exist_ok=True)
                self.capture=gzip.open(file,"xt",encoding="utf-8",compresslevel=3)

            def camera_apply(self,image,dets):
                native=self.original_apply(image,dets)
                if self.selector is None:
                    used,record=native,dict(choice="sof",trigger=False,model_called=False)
                else:
                    used,record=self.selector.apply(image,dets,native)
                self.n_warps+=1
                assert self.n_warps==self.frame_count
                self.capture.write(json.dumps(dict(frame=self.n_warps,native=native.tolist(),used=used.tolist(),**record),
                                              allow_nan=False,separators=(",",":"))+"\n")
                return used

            def finish_sequence_events(self):
                assert self.n_warps==self.frame_count
                self.capture.close()
                return super().finish_sequence_events()

        adapter.BoTSORTPlus=Tracker
        return config,paths,protocol

    runner.prepare=configure
    return runner.run(args)


if __name__=="__main__":
    raise SystemExit(main(*sys.argv[1:]))
