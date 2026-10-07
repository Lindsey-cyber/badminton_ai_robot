#!/usr/bin/env python3
"""Compare raw one-class YOLO predictions from .pt and exported ONNX on real images."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from badminton_ai.model_compat import check_onnx_file
from demo_badminton_detection import YOLOv8Detector


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True, help="One source-group image directory")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--max-abs-tolerance", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (not args.weights.is_file() or not args.onnx.is_file() or args.count < 1 or
            args.max_abs_tolerance <= 0):
        parser.error("Existing model files, positive count and tolerance are required")
    paths = sorted(args.images.glob("*.jpg"))
    if not paths:
        parser.error("No JPG images found")
    sampled = [paths[i * len(paths) // min(args.count, len(paths))]
               for i in range(min(args.count, len(paths)))]

    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO

    check_onnx_file(args.onnx, 640)
    detector = YOLOv8Detector(str(args.onnx), target_classes=[0])
    model = YOLO(str(args.weights))
    if len(model.names) != 1:
        parser.error("Expected one-class fine-tuned weights")
    model.model.eval()
    samples = []
    for image_path in sampled:
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Cannot decode {image_path}")
        tensor, *geometry = detector.preprocess(image)
        with torch.inference_mode():
            native = model.model(torch.from_numpy(tensor))
            if isinstance(native, tuple):
                native = native[0]
            native = native.cpu().numpy()
        exported = detector.session.run(detector.output_names,
                                        {detector.input_name: tensor})[0]
        if native.shape != exported.shape:
            raise RuntimeError(f"Model output shape mismatch: {native.shape} vs {exported.shape}")
        diff = np.abs(native - exported)
        source_boxes = detector.postprocess([native], *geometry)
        onnx_boxes = detector.postprocess([exported], *geometry)
        paired = zip(source_boxes, onnx_boxes)
        box_differences = [max(abs(a - b) for a, b in zip(source["bbox"], converted["bbox"]))
                           for source, converted in paired]
        score_differences = [abs(source["score"] - converted["score"])
                             for source, converted in zip(source_boxes, onnx_boxes)]
        samples.append({"image": image_path.name,
                        "max_abs": float(diff.max()), "mean_abs": float(diff.mean()),
                        "source_detections": len(source_boxes),
                        "onnx_detections": len(onnx_boxes),
                        "class_ids_match": ([d["class_id"] for d in source_boxes] ==
                                            [d["class_id"] for d in onnx_boxes]),
                        "max_paired_bbox_abs_px": max(box_differences, default=0.0),
                        "max_paired_score_abs": max(score_differences, default=0.0)})
    worst = max(row["max_abs"] for row in samples)
    report = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "pt_sha256": digest(args.weights), "onnx_sha256": digest(args.onnx),
              "images": str(args.images), "sample_count": len(samples),
              "max_abs_tolerance": args.max_abs_tolerance,
              "worst_max_abs": worst,
              "median_mean_abs": statistics.median(row["mean_abs"] for row in samples),
              "final_count_mismatch_images": sum(row["source_detections"] != row["onnx_detections"]
                                                 for row in samples),
              "final_class_mismatch_images": sum(not row["class_ids_match"] for row in samples),
              "max_paired_bbox_abs_px": max(row["max_paired_bbox_abs_px"] for row in samples),
              "max_paired_score_abs": max(row["max_paired_score_abs"] for row in samples),
              "samples": samples,
              "comparison": "Same existing detector preprocessing and postprocessing; raw [1,5,N] output and final boxes at detector default confidence 0.20",
              "pass": worst <= args.max_abs_tolerance}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Raw PT→ONNX worst absolute difference: {worst:.6f}; pass={report['pass']}")
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
