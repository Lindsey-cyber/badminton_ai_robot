#!/usr/bin/env python3
"""Fine-tune YOLOv8n; select settings on validation, keeping test untouched."""

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
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="Ultralytics device, e.g. cpu or 0")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "outputs/training")
    parser.add_argument("--name", default="shuttle_yolov8n")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-epochs", type=float, default=3.0,
                        help="Keep below total epochs for short CPU experiments")
    parser.add_argument("--mosaic", type=float, default=0.0)
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument("--fliplr", type=float, default=0.5)
    args = parser.parse_args()
    if (args.epochs < 1 or args.batch < 1 or args.workers < 0 or args.imgsz != 640 or args.seed < 0 or
            not 0 <= args.warmup_epochs < args.epochs or
            not 0 <= args.mosaic <= 1 or not 0 <= args.scale <= 1 or
            not 0 <= args.fliplr <= 1):
        parser.error("Invalid epochs, warmup, batch, seed or augmentation; current ONNX detector requires imgsz=640")

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
                batch=args.batch, workers=args.workers, device=args.device,
                project=str(args.runs_dir.resolve()),
                name=args.name, seed=args.seed, deterministic=True,
                warmup_epochs=args.warmup_epochs,
                mosaic=args.mosaic, scale=args.scale, fliplr=args.fliplr, flipud=0.0)
    best = Path(model.trainer.best)
    if not best.is_file():
        raise RuntimeError(f"Training finished without best weights: {best}")
    trained = YOLO(str(best))
    metrics = trained.val(data=str(yaml_path), split="val", imgsz=args.imgsz,
                          batch=args.batch, workers=args.workers, device=args.device,
                          plots=False, project=str((best.parent.parent / "evaluation").resolve()),
                          name="val")
    exported = Path(trained.export(format="onnx", imgsz=args.imgsz,
                                   dynamic=False, nms=False, simplify=False))
    check_onnx_file(exported, args.imgsz)

    manifest = args.dataset / "manifest.json"
    manifest_data = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else {}
    train_args = best.parent.parent / "args.yaml"
    report = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_yaml": str(yaml_path),
        "dataset_manifest_sha256": file_sha256(manifest) if manifest.is_file() else None,
        "dataset_sha256": manifest_data.get("dataset_sha256"),
        "splits": {name: vars(counts) for name, counts in splits.items()},
        "starting_weights": args.weights,
        "starting_weights_sha256": (file_sha256(Path(args.weights))
                                    if Path(args.weights).is_file() else None),
        "epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch,
        "workers": args.workers, "run_name": args.name,
        "device": args.device, "seed": args.seed,
        "warmup_epochs": args.warmup_epochs,
        "augmentations": {"mosaic": args.mosaic, "scale": args.scale,
                          "fliplr": args.fliplr, "flipud": 0.0},
        "ultralytics": ultralytics.__version__, "torch": torch.__version__,
        "validation_precision": float(metrics.box.mp),
        "validation_recall": float(metrics.box.mr),
        "validation_map50": float(metrics.box.map50),
        "validation_map50_95": float(metrics.box.map),
        "best_pt": str(best.resolve()), "best_pt_sha256": file_sha256(best),
        "train_args_yaml_sha256": file_sha256(train_args) if train_args.is_file() else None,
        "onnx": str(exported.resolve()), "onnx_sha256": file_sha256(exported),
        "onnx_size_bytes": exported.stat().st_size,
        "inference_latency_ms": None,
        "latency_note": "Run scripts/benchmark.py with this ONNX file on the target hardware.",
    }
    destination = best.parent.parent / "training_report.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved validation metrics and model hashes: {destination}; held-out test untouched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
