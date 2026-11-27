import hashlib
import sys
from pathlib import Path
from paths import ROOT,REPO,SETTINGS,FOCUS,read,save


def main():
    method,stage=sys.argv[1:]
    import cv2
    import torch
    import run_botsort_plus as runner
    from online import install
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    if method=="adaptive_gmc":
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
    output=ROOT/stage
    settings=read(SETTINGS)
    config=read(REPO/"configs/botsort_plus/12_baseline_ref.json")
    config["name"]=method
    conf=output/"configs"/(method+".json")
    save(conf,config)
    args=runner.parser().parse_args(["--config",str(conf),"--baseline-root",settings["baseline_root"],
         "--features-dir",settings["features_dir"],"--images-root",settings["images_root"],
         "--runtime-root",settings["runtime_root"],"--output-dir",str(output)]+
         (["--sequences",*FOCUS] if stage=="smoke" else []))
    prepare=runner.prepare
    def inspect(parsed):
        cfg,paths,protocol=prepare(parsed)
        assert not cfg.enabled
        install(method,list(paths),output,stage=="smoke")
        protocol["native_ablation"]={"arm":method,"scripts_sha256":{
            x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in
            (ROOT/"adaptive_gmc.py",ROOT/"online.py",ROOT/"replay_arm.py")},"gmc_model":
            read(ROOT/"vendor/provenance.json") if method=="adaptive_gmc" else None,
            "internal_virtual_measurements":method=="oru","past_outputs_modified":False}
        return cfg,paths,protocol
    runner.prepare=inspect
    return runner.run(args)


if __name__=="__main__":
    raise SystemExit(main())
