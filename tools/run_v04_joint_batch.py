#!/usr/bin/env python3
"""Portable J0-J3 B/GMC diagnostics and E3 human-review closure."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.vttiny_gmc.round3_joint.suite import main

if __name__ == "__main__":
    raise SystemExit(main())
