"""Score the unmodified server predictions and align them by original detection row."""
import sys
import clear_audit as audit
from common import JOINT,B,COMBO,OUT,read,save,EVAL
from common_audit import sha

seq=audit.SEQ=sys.argv[1]
assert sha(EVAL/"trackeval/metrics/clear.py")=="da78468866f61ecf84e197446e630382e0c976ca1f47b70e1dd4904ab7ba598e"
dets=audit.source()
report=read(JOINT/"full/metrics/summary.json")
official={row["tracker"]:row for row in report["results"] if row["sequence"]==seq}
res={}
for m in ("native","dual_low",B,COMBO):
    result=audit.evaluate_rows(JOINT/"full/trackers",m,dets)
    assert all(result[-1][k]==official[m][k] for k in ("FP","FN","IDSW"))
    res[m]=result
for left,right in [(B,COMBO),("native","dual_low")]:
    directory=OUT/seq/("with_B" if left==B else "without_B")
    audit.compare(res[left],res[right],left,right,directory)
save(OUT/seq/"official.json",official)
