#!/usr/bin/env python3
"""Evaluate a selected one-class .pt model; use test only after validation selection."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from badminton_ai.dataset import validate_dataset, write_dataset_yaml


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.batch < 1 or not args.weights.is_file():
        parser.error("Positive batch and existing fine-tuned weights required")
    counts = validate_dataset(args.dataset)
    yaml = write_dataset_yaml(args.dataset)
    manifest = args.dataset / "manifest.json"
    if not manifest.is_file():
        parser.error("Dataset manifest is required")

    import torch
    import ultralytics
    from ultralytics import YOLO

    model = YOLO(str(args.weights.resolve()))
    if len(model.names) != 1:
        parser.error("Expected one-class fine-tuned weights")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics = model.val(data=str(yaml), split=args.split, imgsz=640,
                        batch=args.batch, workers=0, device="cpu", plots=False,
                        project=str(args.output.parent.resolve()),
                        name=f"{args.output.stem}_ultralytics")
    report = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "split": args.split, "images": counts[args.split].images,
              "boxes": counts[args.split].boxes,
              "dataset_manifest_sha256": digest(manifest),
              "dataset_sha256": json.loads(manifest.read_text())["dataset_sha256"],
              "weights_sha256": digest(args.weights), "weights_size_bytes": args.weights.stat().st_size,
              "python": platform.python_version(), "torch": torch.__version__,
              "ultralytics": ultralytics.__version__,
              "imgsz": 640, "batch": args.batch, "device": "cpu",
              "precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
              "map50": float(metrics.box.map50), "map50_95": float(metrics.box.map),
              "validation_speed_ms": metrics.speed,
              "limitations": ["Rights of the underlying broadcast footage remain unverified.",
                              "Fixed-size annotation boxes may not trace actual shuttle extent."]}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{args.split}: P={report['precision']:.4f} R={report['recall']:.4f} "
          f"mAP50={report['map50']:.4f} mAP50-95={report['map50_95']:.4f}")


if __name__ == "__main__":
    main()
