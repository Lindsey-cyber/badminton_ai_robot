#!/usr/bin/env python3
"""Evaluate COCO sports-ball class 32 against one-class shuttle annotations.

The temporary label copy maps target 0 to COCO class 32. Original ground-truth
labels remain unchanged. This is only a pretrained baseline, not a trained
shuttlecock detector or evidence of clean footage rights.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from badminton_ai.dataset import validate_dataset


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def create_mapped_evaluation(dataset: Path, workspace: Path, split: str,
                             names: dict[int, str]) -> Path:
    """Hard-link images so Ultralytics looks up the mapped labels, not source labels."""
    workspace.mkdir(parents=True, exist_ok=True)
    images = workspace / "images" / split
    images.mkdir(parents=True)
    labels = workspace / "labels" / split
    labels.mkdir(parents=True)
    for original in sorted((dataset / "images" / split).glob("*.jpg")):
        try:
            os.link(original, images / original.name)
        except OSError:
            shutil.copy2(original, images / original.name)
    for original in sorted((dataset / "labels" / split).glob("*.txt")):
        rows = original.read_text(encoding="utf-8").splitlines()
        (labels / original.name).write_text("".join("32 " + row.split(maxsplit=1)[1] + "\n"
                                                    for row in rows), encoding="utf-8")
    yaml = workspace / "data.yaml"
    yaml.write_text(f"path: {json.dumps(str(workspace))}\n"
                    "train: images/train\nval: images/val\ntest: images/test\n"
                    "names: " + json.dumps([names[i] for i in range(80)]) + "\n",
                    encoding="utf-8")
    return yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path, help="Official COCO YOLOv8n .pt")
    parser.add_argument("--workspace", required=True, type=Path,
                        help="Empty scratch directory for mapped labels and evaluation output")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    dataset, workspace = args.dataset.resolve(), args.workspace.resolve()
    if args.batch < 1 or not args.weights.is_file() or (workspace.exists() and any(workspace.iterdir())):
        parser.error("Positive batch, existing weights and an empty workspace are required")
    validate_dataset(dataset)
    manifest = dataset / "manifest.json"
    if not manifest.is_file():
        parser.error("Dataset manifest is required for experiment provenance")
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))

    import torch
    import ultralytics
    from ultralytics import YOLO

    model = YOLO(str(args.weights.resolve()))
    if len(model.names) != 80 or model.names[32] != "sports ball":
        parser.error("Weights must use the 80-class COCO layout with sports ball at class 32")

    yaml = create_mapped_evaluation(dataset, workspace, args.split, model.names)

    metrics = model.val(data=str(yaml), split=args.split, imgsz=640,
                        batch=args.batch, device="cpu", workers=0,
                        project=str(workspace), name="pretrained_evaluation", exist_ok=True,
                        plots=False, verbose=False)
    report = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "pretrained_coco_sports_ball_baseline",
        "split": args.split,
        "ground_truth_original_class": 0,
        "prediction_and_mapped_class": 32,
        "dataset_manifest_sha256": digest(manifest),
        "dataset_sha256": manifest_data["dataset_sha256"],
        "source_group_to_split": manifest_data["source_group_to_split"],
        "weights_sha256": digest(args.weights),
        "weights_size_bytes": args.weights.stat().st_size,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "device": "cpu", "imgsz": 640, "batch": args.batch,
        "precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50), "map50_95": float(metrics.box.map),
        "validation_speed_ms": metrics.speed,
        "limitations": [
            "COCO sports ball is not a shuttlecock class.",
            "Label source footage rights and fixed-size box quality are not verified.",
            "Only one filename-derived video group is used for this split.",
        ],
    }
    (workspace / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                           encoding="utf-8")
    print(f"{args.split}: precision={report['precision']:.4f} recall={report['recall']:.4f} "
          f"mAP50={report['map50']:.4f} mAP50-95={report['map50_95']:.4f}")


if __name__ == "__main__":
    main()
