"""Replay first, independently score the focus, then evaluate the locked policies on the full split."""
import hashlib,json,os,shutil,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
INT=ROOT.parent
PREV=INT/"b_gmc_forensics_r1"
ORIG=INT/"botsort_plus_v0_4_motion_blur/BOT-SORT-Plus-followup-c3e142e6"
SETTINGS=INT/"botsort_plus_v0_4_motion_blur/v04_followup_round2/P0/motion_blur_v0.4.local.json"
VENDOR=INT/"native_botsort_gmc_oru_v1/vendor"
EVAL=INT.parent/"external/TrackEval"
DEPLOY=ROOT/"repo"
FOCUS="DJI_0030_2"
REF="evidence_both"
ART=ROOT/"results"
CASE=DEPLOY/"experiments/vttiny_gmc/native_botsort_gmc_0030_fp_v1"
settings=json.loads(SETTINGS.read_bytes())
env=dict(os.environ,PYTHONUSERBASE=str(INT/"botsort_plus_v0_1_0/python_user"),CUDA_VISIBLE_DEVICES="5",
         PYTHONPATH=os.pathsep.join(map(str,[DEPLOY,DEPLOY/"tools",Path(settings["runtime_root"])/"tracking_runtime",
                                              INT/"native_botsort_gmc_0030_fp_v1",EVAL])),
         OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",OMP_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1",
         PYTHONDONTWRITEBYTECODE="1",TRACKEVAL_ROOT=str(EVAL),YOLO_CONFIG_DIR=str(ROOT/"runtime_cfg"),
         PATH=str(Path(sys.executable).parent)+os.pathsep+os.environ["PATH"])
def j(path): return json.loads(path.read_bytes())
def save(path,obj):
    path.parent.mkdir(exist_ok=True,parents=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf8")
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def build(name):
    if not DEPLOY.exists(): shutil.copytree(PREV/"live_repo",DEPLOY)
    shutil.copy2(ROOT/"pending.py",CASE/"pending.py")
    old=PREV/"live_repo"
    def patch(relative,edits):
        code=(old/relative).read_text()
        for a,b in edits:
            if code.count(a)!=1: raise ValueError(f"Unexpected source in {relative}: {a}")
            code=code.replace(a,b)
        (DEPLOY/relative).write_text(code)
    patch("botsort_plus/adapter_v2.py",[('        reserved = decision.reserved_candidate_ids',
        '        if hasattr(tracker,"pending"):\n            tracker.pending.step(frame,warp,eligible,current_candidates,decision,activated,refound,app_cost)\n        reserved = decision.reserved_candidate_ids')])
    patch((CASE/"live_arm.py").relative_to(DEPLOY),[
        ("from lost_camera_margin import LostCameraMargin","from lost_camera_margin import LostCameraMargin\nfrom pending import PendingIdentity"),
        ('("margin_both","evidence_both")','("evidence_both","pending_strict","pending_temporal","pending_probation")'),
        ('require_observation=trial=="evidence_both"','require_observation=True'),
        ("                self.recovery_matcher._pair=self.margin.recovery_pair(self.recovery_matcher._pair)",
         "                self.recovery_matcher._pair=self.margin.recovery_pair(self.recovery_matcher._pair)\n                if trial!=\"evidence_both\": self.pending=PendingIdentity(self,temporal=trial!=\"pending_strict\",probation=trial==\"pending_probation\")"),
        ("                if self.matched_camera:\n                    for i, track", "                if hasattr(self,'pending'): self.pending.guard_low(cost,tracks,detections)\n                if self.matched_camera:\n                    for i, track"),
        ("                return rows","                if hasattr(self,'pending'): self.pending.record_outputs(rows,dets)\n                return rows"),
        ("                return super().finish_sequence_events()",
         "                if hasattr(self,'pending'):\n                    write(root/'pending'/trial/(self.seq+'.json'),dict(events=self.pending.events,counts=dict(self.pending.counts)))\n                return super().finish_sequence_events()"),
        ('protocol["lost_camera_policy"]=dict(variant=trial','protocol["pending_identity"]=dict(policy=trial,source=sha(ROOT/"pending.py"),adapter=sha(repository/"botsort_plus/adapter_v2.py"),live=sha(ROOT/"live_arm.py"))\n        protocol["lost_camera_policy"]=dict(variant=trial')])
    if name in ("pending_guarded", "pending_shadow"):
        shutil.copy2(ROOT/"pending_refined.py", CASE/"pending_refined.py")
        path=CASE/"live_arm.py"
        s=path.read_text()
        for a,b in (
            ("from pending import PendingIdentity", "from pending_refined import DeferredEvidence as PendingIdentity"),
            ('("evidence_both","pending_strict","pending_temporal","pending_probation")', '("pending_guarded","pending_shadow")'),
            ('PendingIdentity(self,temporal=trial!="pending_strict",probation=trial=="pending_probation")','PendingIdentity(self,shadow=trial=="pending_shadow")'),
            ('protocol["lost_camera_policy"]=dict(variant=trial', 'protocol["pending_identity"]["refined_source"]=sha(ROOT/"pending_refined.py")\n        protocol["lost_camera_policy"]=dict(variant=trial')):
            assert s.count(a)==1,a
            s=s.replace(a,b)
        path.write_text(s)

def run(cmd,overrides=None):
    subprocess.run([sys.executable,"-Bu",*map(str,cmd)],env=dict(env,**(overrides or {})),cwd=DEPLOY,check=True)
def attach(directory,src,m):
    for sub,filename in (("protocols",m+".json"),("trackers",m)):
        dest=directory/sub/filename
        dest.parent.mkdir(exist_ok=True,parents=True)
        if not dest.exists(): dest.symlink_to(src/sub/filename)
def execute(stage,names):
    focus=stage.startswith("focus")
    aggregate=ROOT/"scores"/stage
    baseline=PREV/"full_runs"/REF
    reference=j(baseline/"protocols"/(REF+".json"))
    checks={}
    methods=([REF]+names)
    for name in methods:
        work=(ROOT/"runs"/("focus" if focus else "full")/name) if focus or name!=REF else baseline
        protocol=work/"protocols"/(name+".json")
        if not protocol.exists():
            build(name)
            print(stage,name,flush=True)
            run([DEPLOY/"tools/run_vttiny_gmc.py","--settings",SETTINGS,"--vendor",VENDOR,"--output-dir",work,
                 *(["--sequences",FOCUS] if focus else [])],{"BG_LOST_POLICY":name})
        p=j(protocol)
        assert p["status"]=="COMPLETE" and p["lost_camera_policy"]["variant"]==name
        assert set(p["sequences"])==({FOCUS} if focus else set(reference["sequences"]))
        for k in ("detector_sha256","rgb_encoder_sha256","native_config","encoder_contracts","boxmot_source_hashes"):
            assert p[k]==reference[k],(k,name)
        for s in p["sequences"]:
            for k in ("detection_fingerprints","feature_hashes","decoded_image_hashes","sequences"):
                assert p[k][s]==reference[k][s],(name,s,k)
            assert digest(work/"trackers"/name/"data"/(s+".txt"))==p["output_prediction_hashes"][s]
        if name in (REF,"pending_shadow"): assert p["output_prediction_hashes"][FOCUS]==reference["output_prediction_hashes"][FOCUS]
        if not focus and name!=REF:
            focused=j(ROOT/"runs/focus"/name/"protocols"/(name+".json"))
            assert p["output_prediction_hashes"][FOCUS]==focused["output_prediction_hashes"][FOCUS]
            assert p["pending_identity"]==focused["pending_identity"]
        checks[name]=dict(focus_sha256=p["output_prediction_hashes"][FOCUS],frames=sum(p["sequences"].values()),sequences=len(p["sequences"]))
        attach(aggregate,work,name)
    run([ORIG/"tools/evaluate_botsort_plus.py","--run-dir",aggregate,"--trackers",*methods,"--reference-tracker",REF,
         "--gt-root",settings["gt_root"],"--trackeval-root",EVAL,"--output-dir",ART/stage/"metrics"])
    save(ART/stage/"parity.json",checks)
    run([ROOT/"audit.py",stage,*methods])
    print("COMPLETE",stage,flush=True)

if __name__=="__main__":
    stage=sys.argv[1]
    lock=j(ROOT/("plan_"+stage+".json"))
    execute(stage,lock["methods"])
