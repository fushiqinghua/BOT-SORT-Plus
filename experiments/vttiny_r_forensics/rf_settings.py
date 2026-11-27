"""Isolated output alongside the previous frozen four-arm experiment."""
import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('VT_RF_ROOT', SRC)).resolve()
INT = Path(os.environ.get('VT_INTEGRATIONS', ROOT.parent)).resolve()
QR = Path(os.environ.get('VT_RF_QR', INT/'native_neural_qr_v1')).resolve()
OUT = ROOT/'results'
CASE = 'DJI_0309_4'
sys.path.insert(0,str(QR))
from qr_settings import ORIG, SETTINGS, EVAL, PROJECT, read, write, table, sha
from qr_settings import environment as original_environment

def environment():
    env=original_environment()
    env.update(VT_RF_ROOT=str(ROOT),VT_RF_QR=str(QR),VT_QR_ROOT=str(QR),VT_INTEGRATIONS=str(INT))
    env['PYTHONPATH']=os.pathsep.join([str(SRC),str(QR),env['PYTHONPATH']])
    env['YOLO_CONFIG_DIR']=str(ROOT/'runtime_config')
    return env
