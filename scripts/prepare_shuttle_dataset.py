#!/usr/bin/env python3
"""Validate labeled YOLO shuttlecock images and write their dataset YAML."""

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from badminton_ai.dataset import validate_dataset, write_dataset_yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    try:
        summary = validate_dataset(args.dataset)
        yaml_path = write_dataset_yaml(args.dataset)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Dataset validation failed: {exc}\n")
    for split, counts in summary.items():
        print(f"{split}: {counts.images} images, {counts.boxes} shuttlecock boxes")
    print(f"Wrote {yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
