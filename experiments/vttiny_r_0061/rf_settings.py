"""Freeze both historical run directories. New traces and analyses go under ROOT."""
from pathlib import Path
import os
import sys
ROOT=Path(__file__).resolve().parent
QR=ROOT.parent/'native_neural_qr_v1'
RF=ROOT.parent/'native_r_forensics_v1'
OUT=ROOT/'results'
CASE='DJI_0061_2'
RUNS={'native':QR/'full','low_r':RF/'full'}
sys.path.append(str(QR))
from qr_settings import read,write,table,sha,ORIG,SETTINGS,EVAL
from qr_settings import environment as qr_environment

def environment():
    env=qr_environment()
    env['PYTHONPATH']=os.pathsep.join([str(ROOT),env['PYTHONPATH'],str(RF)])
    env['YOLO_CONFIG_DIR']=str(ROOT/'runtime_config')
    return env
