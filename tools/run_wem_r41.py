#!/usr/bin/env python3
"""Entry point for the W R4.1 complete online benchmark."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.research.wem_r41.batch import main


if __name__ == "__main__":
    raise SystemExit(main())
