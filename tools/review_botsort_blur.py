#!/usr/bin/env python3
"""D4 offline image-quality and human-reviewed motion-blur diagnostics.

Run D1-D3 first. This reads their GT-assisted detection audit; it never changes
tracking output. Use a new output directory when adding completed manual labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from botsort_plus.diagnostics.blur import analyze_blur


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--tracker-name", required=True)
    p.add_argument("--baseline-root", type=Path, required=True)
    p.add_argument("--features-dir", type=Path, required=True)
    p.add_argument("--images-root", type=Path, required=True)
    p.add_argument("--audit-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--manual-labels", type=Path)
    p.add_argument("--max-review", type=int, default=120)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    result = analyze_blur(**vars(args))
    print(json.dumps({"status": result["status"], "observations": result["overall"]["observations"],
                      "manual_review_rows": result["manual_review_rows"],
                      "manual_label_status": result["manual_label_status"],
                      "output_dir": str(args.output_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
