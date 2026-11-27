"""Execute the archived diagnostic reader in the configured experiment environment."""
import os
from pathlib import Path
import subprocess
import sys


def run_on_host(code):
    run_root = Path(os.environ['TCO_RUN_ROOT']).expanduser().resolve(strict=True)
    for name in ('t01_v1', 't23_v1', 'q2_v1'):
        if not (run_root / name).is_dir():
            raise ValueError(f'Missing experiment directory {name} under TCO_RUN_ROOT')
    setup = (
        'from pathlib import Path\nimport os, sys, json, subprocess\n'
        f'r=Path({str(run_root)!r})\n'
        "root=r/'q2_v1'\nold=r/'t23_v1'\nt0=r/'t01_v1'\nsrc=root/'src'\n"
    )
    return subprocess.run([sys.executable, '-Bu', '-'], input=(setup + code).encode('utf-8'),
                          stdout=subprocess.PIPE, check=True).stdout
