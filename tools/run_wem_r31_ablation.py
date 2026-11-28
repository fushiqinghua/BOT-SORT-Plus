"""Run the fixed W0--W3 cache experiments from any working directory."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research.wem_r31.batch import main


if __name__ == "__main__":
    raise SystemExit(main())
