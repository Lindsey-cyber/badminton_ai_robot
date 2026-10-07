#!/usr/bin/env python3
"""Copy a small, verified YOLO validation subset for human label correction."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).resolve().parents[1] /
                        "outputs/experiments/label_review_manifest.json")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    selection = json.loads(args.manifest.read_text(encoding="utf-8"))
    dataset_manifest = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))
    if selection["dataset_sha256"] != dataset_manifest["dataset_sha256"]:
        parser.error("Dataset version differs from the selected review manifest")

    pairs = []
    for item in selection["items"]:
        name = item["image"]
        if Path(name).name != name or not name.endswith(".jpg"):
            parser.error(f"Unsafe or unsupported image filename: {name}")
        image = args.dataset / "images/val" / name
        label = args.dataset / "labels/val" / f"{Path(name).stem}.txt"
        if sha256(image) != item["image_sha256"] or sha256(label) != item["label_sha256"]:
            parser.error(f"Image or label hash mismatch: {name}")
        pairs.append((item, image, label))

    (args.output / "images/val").mkdir(parents=True)
    (args.output / "labels/val").mkdir(parents=True)
    for _, image, label in pairs:
        shutil.copy2(image, args.output / "images/val" / image.name)
        shutil.copy2(label, args.output / "labels/val" / label.name)
    with (args.output / "review.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("image", "category", "decision", "notes"))
        writer.writerows((item["image"], item["category"], "", "") for item, _, _ in pairs)
    (args.output / "README.md").write_text(
        "# Human shuttlecock label review\n\n"
        "Open each image with its existing YOLO label. In review.csv, record "
        "verified, corrected, absent, or uncertain; describe ambiguous frames. "
        "Edit only the copied labels/val/*.txt files. Do not treat uncertain "
        "frames as ground truth. Return review.csv and corrected label TXT files "
        "privately; the broadcast images should not be redistributed. "
        "This validation audit does not retroactively change the reported "
        "test result.\n", encoding="utf-8")
    print(f"Copied {len(pairs)} image/label pairs to {args.output}")


if __name__ == "__main__":
    main()
