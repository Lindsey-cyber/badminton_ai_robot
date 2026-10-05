#!/usr/bin/env python3
"""Fail on FPS, p95 latency or RSS regression between comparable runs."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from badminton_ai.regression import compare_reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument("--max-fps-drop", type=float, default=0.15)
    parser.add_argument("--max-p95-rise", type=float, default=0.20)
    parser.add_argument("--max-rss-rise", type=float, default=0.20)
    args = parser.parse_args()
    try:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        current = json.loads(args.current.read_text(encoding="utf-8"))
        report = compare_reports(baseline, current, args.max_fps_drop,
                                 args.max_p95_rise, args.max_rss_rise)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Benchmark comparison unavailable: {exc}\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
