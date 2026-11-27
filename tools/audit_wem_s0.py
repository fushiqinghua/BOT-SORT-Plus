#!/usr/bin/env python3
"""CLI for the read-only W v0.1 S0 cache inventory."""
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.research.wem_v01.cache_audit import main


if __name__ == "__main__":
    raise SystemExit(main())
