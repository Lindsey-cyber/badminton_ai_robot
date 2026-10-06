#!/usr/bin/env python3
"""List validation false negatives/positives for manual shuttlecock review.

The confidence/IoU operating point is for case triage, not the Ultralytics mAP
calculation. Blur, occlusion, background and lighting require visual review.
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from badminton_ai.dataset import validate_dataset


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / (area_a + area_b - intersection) if area_a + area_b > intersection else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()
    if not args.weights.is_file() or not 0 < args.confidence < 1:
        parser.error("Existing weights and confidence in (0, 1) are required")
    validate_dataset(args.dataset)

    from ultralytics import YOLO

    labels = args.dataset / "labels" / "val"
    images = args.dataset / "images" / "val"
    model = YOLO(str(args.weights))
    if len(model.names) != 1:
        parser.error("Error analysis requires a fine-tuned one-class model")
    counts: Counter[str] = Counter()
    cases = []
    for result in model.predict(source=str(images), stream=True, imgsz=640,
                                conf=args.confidence, device="cpu", verbose=False):
        image = Path(result.path)
        height, width = result.orig_shape
        rows = (labels / (image.stem + ".txt")).read_text(encoding="utf-8").splitlines()
        gt = []
        for row in rows:
            cls, cx, cy, bw, bh = map(float, row.split())
            if cls != 0:
                raise ValueError(f"Unexpected label class for {image}")
            gt.append(((cx - bw / 2) * width, (cy - bh / 2) * height,
                       (cx + bw / 2) * width, (cy + bh / 2) * height))
        predicted = [(tuple(map(float, box)), float(score)) for box, score in
                     zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist())]
        matched: set[int] = set()
        missed = []
        for target in gt:
            available = [(iou(target, box), index) for index, (box, _) in enumerate(predicted)
                         if index not in matched]
            best, index = max(available, default=(0.0, -1))
            if best >= 0.5:
                matched.add(index)
                counts["tp"] += 1
            else:
                target_center = ((target[0] + target[2]) / 2,
                                 (target[1] + target[3]) / 2)
                nearest = min((math.dist(target_center, ((box[0] + box[2]) / 2,
                                                         (box[1] + box[3]) / 2))
                               for box, _ in predicted), default=None)
                missed.append({"box": [round(v, 2) for v in target],
                               "max_iou": round(best, 4),
                               "nearest_prediction_center_px": (round(nearest, 2)
                                                                if nearest is not None else None),
                               "width_px": round(target[2] - target[0], 2),
                               "height_px": round(target[3] - target[1], 2)})
                counts["fn"] += 1
                counts["fn_width_under_8px"] += target[2] - target[0] < 8
                counts["fn_area_under_100px2"] += ((target[2] - target[0]) *
                                                    (target[3] - target[1]) < 100)
        extras = [dict(box=[round(v, 2) for v in box], confidence=round(score, 4))
                  for i, (box, score) in enumerate(predicted) if i not in matched]
        counts["fp"] += len(extras)
        counts["images"] += 1
        if missed or extras:
            cases.append({"image": image.name, "missed": missed, "extra": extras})

    report = {"weights": str(args.weights), "dataset": str(args.dataset),
              "split": "val", "confidence": args.confidence, "match_iou": 0.5,
              "counts": dict(counts), "cases": cases,
              "limitations": ["Not mAP; one fixed operating threshold for case triage.",
                              "Blur, occlusion, background and lighting need visual review.",
                              "Review missed/extra boxes visually before changing V2 training."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(dict(counts))


if __name__ == "__main__":
    main()
