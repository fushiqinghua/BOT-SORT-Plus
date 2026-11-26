"""Stage the read-only shadow in a separate checkout and gate full replay on byte parity."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

TRACKER = "B_early_moderate"
PILOT = ("DJI_0030_2",)
ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def replace(source, old, new, path):
    if source.count(old) != 1:
        raise RuntimeError(f"{path}: patch anchor is missing or duplicated: {old}")
    return source.replace(old, new, 1)


def patch(work, relative, edits):
    path = work / relative
    source = path.read_text(encoding="utf-8")
    for old, new in edits:
        source = replace(source, old, new, relative)
    compile(source, relative, "exec")
    path.write_text(source, encoding="utf-8")


def stage(repo, work):
    subprocess.run(["git","clone","--quiet","--shared","--no-checkout",str(repo),str(work)],check=True)
    head = subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
    if head != "c3e142e6c11df0f2a7e8a84bede8556c7dabec1e":
        raise ValueError(f"Unexpected v0.4 follow-up head {head}")
    subprocess.run(["git","-C",str(work),"checkout","--quiet","--detach",head],check=True)
    shutil.copyfile(ROOT/"association_shadow.py",work/"botsort_plus/diagnostics/association_shadow.py")
    patch(work,"botsort_plus/native_adapter.py",[
        ('        matches, u_tracks, u_detections = linear_assignment(costs, thresh=threshold)\n',
         '        matches, u_tracks, u_detections = linear_assignment(costs, thresh=threshold)\n'
         '        if stage == "high" and getattr(self, "e1_shadow", None) is not None:\n'
         '            self.e1_shadow.native(self, tracks, detections, threshold, costs, matches)\n'),
    ])
    patch(work,"botsort_plus/adapter_v2.py",[
        ('    frame = tracker.frame_count\n',
         '    frame = tracker.frame_count\n'
         '    e1_shadow = getattr(tracker, "e1_shadow", None)\n'
         '    if e1_shadow is not None:\n        e1_shadow.begin(tracker, frame)\n'),
        ('        decision = tracker.recovery_matcher.match(frame, views, current_candidates, app_cost)\n',
         '        if e1_shadow is not None:\n'
         '            e1_shadow.before_b(tracker.recovery_matcher, views, current_candidates)\n'
         '        decision = tracker.recovery_matcher.match(frame, views, current_candidates, app_cost)\n'
         '        if e1_shadow is not None:\n            e1_shadow.after_b(decision)\n'),
    ])
    patch(work,"botsort_plus/candidates.py",[
        ('        # Highest evidence first, then recent observation and deterministic ID.\n',
         '        if getattr(self, "e1_shadow", None) is not None:\n'
         '            self.e1_shadow.candidate(observations, retained, old, costs, pairs)\n'
         '        # Highest evidence first, then recent observation and deterministic ID.\n'),
    ])
    patch(work,"botsort_plus/recovery.py",[
        ('        pairs, _, _ = assign_with_rejection(eligible, self.config.max_cost, margin=0)\n',
         '        if getattr(self, "e1_shadow", None) is not None:\n'
         '            self.e1_shadow.b_cost(raw_costs, eligible)\n'
         '        pairs, _, _ = assign_with_rejection(eligible, self.config.max_cost, margin=0)\n'),
    ])
    patch(work,"tools/run_botsort_plus.py",[
        ('    from botsort_plus.native_adapter import BoTSORTPlus\n',
         '    from botsort_plus.native_adapter import BoTSORTPlus\n'
         '    from botsort_plus.diagnostics.association_shadow import ShadowRecorder\n'),
        ('            position = position_class(position_policy, registration) if position_policy is not None else None\n',
         '            shadow = ShadowRecorder(root/"shadow"/name/(seq+".jsonl.gz"), seq, len(cache["frames"]))\n'
         '            tracker.e1_shadow = tracker.candidate_pool.e1_shadow = tracker.recovery_matcher.e1_shadow = shadow\n'
         '            position = position_class(position_policy, registration) if position_policy is not None else None\n'),
        ('                        translation = position.update(row["frame"],image_path,prediction,tracker.last_events) if position else {}\n',
         '                        shadow.finish()\n'
         '                        translation = position.update(row["frame"],image_path,prediction,tracker.last_events) if position else {}\n'),
        ('                temp.replace(dest)\n',
         '                temp.replace(dest)\n                shadow.close()\n'),
        ('            except Exception as error:\n',
         '            except Exception as error:\n                shadow.abort()\n'),
    ])
    diffs = subprocess.check_output(["git","-C",str(work),"diff","--binary"],text=True,encoding="utf-8")
    (ROOT/"shadow_probes.patch").write_text(diffs,encoding="utf-8")


def environment(suite, runtime):
    v01 = suite.with_name("botsort_plus_v0_1_0")
    return {
        **os.environ,"CUDA_VISIBLE_DEVICES":"","OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1",
        "MKL_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1","PYTHONDONTWRITEBYTECODE":"1",
        "PYTHONUTF8":"1","PYTHONUNBUFFERED":"1","PYTHONUSERBASE":str(v01/"python_user"),
        "TRACKING_RUNTIME_ROOT":str(runtime),"YOLO_CONFIG_DIR":str(ROOT/"yolo_settings"),
    }


def parity(output, frozen, chosen):
    primary = read(frozen/"verification.json")
    protocol_name = TRACKER+".json"
    observed = read(frozen/"observed/protocols"/protocol_name)
    shadow_protocol = read(output/"protocols"/protocol_name)
    expected = {seq:n for seq,n in observed["sequences"].items() if not chosen or seq in chosen}
    if shadow_protocol["status"] != "COMPLETE" or shadow_protocol["sequences"] != expected:
        raise ValueError("Incomplete association replay")
    for name in ("config","native_config","detector_sha256","rgb_encoder_sha256",
                 "source_protocol_sha256","boxmot_source_hashes"):
        if shadow_protocol[name] != observed[name]:
            raise ValueError(f"Shadow input differs: {name}")
    observed_rows = digest(frozen/"observed/protocols"/protocol_name)
    if observed_rows != primary["observed_protocol_sha256"]:
        raise ValueError("Original frozen protocol changed")
    result = {}
    for seq in expected:
        mot=output/"trackers"/TRACKER/"data"/(seq+".txt")
        events=output/"events"/TRACKER/(seq+".jsonl")
        shadow=output/"shadow"/TRACKER/(seq+".jsonl.gz")
        entry=primary["sequences"][seq]
        for key in ("detection_fingerprints","feature_hashes","decoded_image_hashes"):
            if shadow_protocol[key][seq] != observed[key][seq]:
                raise ValueError(f"Shadow {key} differs for {seq}")
        mot_hash, event_hash = digest(mot), digest(events)
        if mot_hash != entry["sha256"] or event_hash != entry["original_events_sha256"]:
            raise ValueError(f"New replay changed native output or original events for {seq}")
        result[seq] = dict(mot_sha256=mot_hash,event_sha256=event_hash,shadow_sha256=digest(shadow),
                           frames=expected[seq])
    return {"status":"PASS","pilot":bool(chosen),"native_mot_events_byte_exact":True,
            "all_original_recovery_checks_byte_exact":True,"frozen_protocol_sha256":observed_rows,
            "replay_protocol_sha256":digest(output/"protocols"/protocol_name),"sequences":result}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["stage","pilot","full"])
    p.add_argument("--repo",required=True,type=Path)
    args = p.parse_args()
    repo = args.repo.resolve()
    work = ROOT/"shadow_checkout"
    if args.phase == "stage":
        stage(repo, work)
        return
    suite = repo.parent
    original = suite/"BOT-SORT-Plus"
    settings = read(original/"configs/diagnostics/motion_blur_v0.4.local.json")
    frozen=Path(settings["output_root"])
    flow=ROOT.parent
    if not (flow/"P0/verification.json").is_file() or read(ROOT/"cohort/cohort_summary.json")["status"]!="COMPLETE":
        raise RuntimeError("P0 and E1-A gates must complete first")
    if args.phase == "full" and read(ROOT/"pilot_parity.json")["status"]!="PASS":
        raise RuntimeError("Single-sequence parity required before full replay")
    chosen=PILOT if args.phase=="pilot" else ()
    out=ROOT/(args.phase+"_replay")
    base=Path(settings["baseline_root"])
    if not base.is_absolute():
        base=original/base
    runtime=Path(settings["runtime_root"])
    command=[sys.executable,'-B','-X','utf8',str(work/"tools/run_botsort_plus.py"),
             '--baseline-root',str(base),'--features-dir',settings["features_dir"],
             '--images-root',settings["images_root"],'--config',str(work/"configs/botsort_plus/B_early_moderate.json"),
             '--runtime-root',str(runtime),'--output-dir',str(out),'--trace']
    if chosen:
        command += ['--sequences',*chosen]
    subprocess.run(command,cwd=work,env=environment(suite,runtime),check=True)
    report=parity(out,frozen,chosen)
    write(ROOT/(args.phase+"_parity.json"),report)
    print("E1_SHADOW_BYTE_PARITY",args.phase,len(report["sequences"]),flush=True)


if __name__=="__main__":
    main()
