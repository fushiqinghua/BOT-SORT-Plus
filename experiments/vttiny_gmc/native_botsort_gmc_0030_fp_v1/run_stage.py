import os
import sys
from common_audit import ROOT,env

STAGES={"inspect":"clear_audit.py","trace":"replay_case.py","counterfactual":"replay_case.py",
        "focus":"candidate_suite.py","full":"candidate_suite.py","package":"report_case.py"}

if __name__=="__main__":
    stage=sys.argv[1]
    os.execve(sys.executable,[sys.executable,"-Bu",str(ROOT/STAGES[stage]),stage],env())
