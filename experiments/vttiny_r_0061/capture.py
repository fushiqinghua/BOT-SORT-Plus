"""Unchanged tracker interventions with complete native association/update instrumentation."""
import sys
from rf_settings import RF,ROOT,CASE,RUNS,read,sha
sys.path.append(str(RF))
import branch_replay
if __name__=='__main__':
    m=sys.argv[1]
    assert m in RUNS
    branch_replay.CASE=CASE
    branch_replay.ROOT=ROOT
    branch_replay.main('pilot',m)
    src=RUNS[m]/'trackers'/m/'data'/(CASE+'.txt')
    dst=ROOT/'pilot/trackers'/m/'data'/(CASE+'.txt')
    assert sha(src)==sha(dst)==read(RUNS[m]/'protocols'/(m+'.json'))['output_prediction_hashes'][CASE]
    print('Exact historical MOT parity',m,sha(src),flush=True)
