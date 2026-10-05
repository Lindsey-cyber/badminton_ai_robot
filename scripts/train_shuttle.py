#!/usr/bin/env python3
"""Fine-tune YOLOv8n on real labeled shuttlecocks, evaluate, export ONNX."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from badminton_ai.dataset import validate_dataset, write_dataset_yaml
from badminton_ai.model_compat import check_onnx_file


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--weights", default="yolov8n.pt",
                        help="Starting pretrained weights; Ultralytics may download these")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Fixed export size; the existing ONNX detector uses 640")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu", help="Ultralytics device, e.g. cpu or 0")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "outputs/training")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.imgsz != 640:
        parser.error("epochs and batch must be positive; current ONNX detector requires imgsz=640")

    try:
        splits = validate_dataset(args.dataset)
        yaml_path = write_dataset_yaml(args.dataset)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Dataset validation failed: {exc}\n")

    try:
        import torch
        import ultralytics
        from ultralytics import YOLO
    except ImportError as exc:
        parser.exit(1, f"Install requirements-training.txt on the training machine: {exc}\n")

    model = YOLO(args.weights)
    model.train(data=str(yaml_path), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=args.device, project=str(args.runs_dir.resolve()),
                name="shuttle_yolov8n", seed=42, deterministic=True,
                mosaic=0.0, scale=0.1, fliplr=0.5, flipud=0.0)
    best = Path(model.trainer.best)
    if not best.is_file():
        raise RuntimeError(f"Training finished without best weights: {best}")
    trained = YOLO(str(best))
    metrics = trained.val(data=str(yaml_path), split="test", imgsz=args.imgsz,
                          device=args.device)
    exported = Path(trained.export(format="onnx", imgsz=args.imgsz,
                                   dynamic=False, nms=False, simplify=False))
    check_onnx_file(exported, args.imgsz)

    report = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_yaml": str(yaml_path),
        "splits": {name: vars(counts) for name, counts in splits.items()},
        "starting_weights": args.weights,
        "epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch,
        "device": args.device, "seed": 42,
        "augmentations": {"mosaic": 0.0, "scale": 0.1, "fliplr": 0.5, "flipud": 0.0},
        "ultralytics": ultralytics.__version__, "torch": torch.__version__,
        "test_precision": float(metrics.box.mp),
        "test_recall": float(metrics.box.mr),
        "test_map50": float(metrics.box.map50),
        "test_map50_95": float(metrics.box.map),
        "best_pt": str(best.resolve()), "best_pt_sha256": file_sha256(best),
        "onnx": str(exported.resolve()), "onnx_sha256": file_sha256(exported),
        "onnx_size_bytes": exported.stat().st_size,
        "inference_latency_ms": None,
        "latency_note": "Run scripts/benchmark.py with this ONNX file on the target hardware.",
    }
    destination = best.parent.parent / "training_report.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved held-out test metrics and model hashes: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
