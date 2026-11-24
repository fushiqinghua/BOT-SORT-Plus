"""CLI entry point for real RGB/thermal feature extraction."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from botsort_plus.feature_export import main

if __name__ == "__main__":
    main()
