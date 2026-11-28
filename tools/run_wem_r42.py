"""CLI entry for the frozen W0 stabilization batch."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.research.wem_r42.batch import main

if __name__ == "__main__":
    raise SystemExit(main())
