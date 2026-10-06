#!/usr/bin/env python3
"""Regroup Mathieu Cartron's Roboflow v1 export by source video.

Its published random split mixes neighboring frames from each of three video
groups. This importer uses video 2 for training, video 1 for validation and
video 3 for the held-out test. It never generates or alters annotations.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from badminton_ai.dataset import validate_dataset, write_dataset_yaml


UPSTREAM = "https://universe.roboflow.com/mathieu-cartron/shuttlecock-cqzy3/dataset/1"
GROUPS = {"2": "train", "1": "val", "3": "test"}
NAME = re.compile(r"video_label_(\d+)_(\d+)_jpg\.rf\.([0-9a-f]+)\.jpg$")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare(source: Path, output: Path, mirror_revision: str) -> dict:
    source, output = source.resolve(), output.resolve()
    if not mirror_revision:
        raise ValueError("An exact mirror commit or source archive hash is required")
    metadata = source / "README.dataset.txt"
    roboflow = source / "README.roboflow.txt"
    if not metadata.is_file() or not roboflow.is_file():
        raise ValueError("Expected the Roboflow export README files")
    if ("License: CC BY 4.0" not in metadata.read_text(encoding="utf-8") or
            "The dataset includes 8053 images." not in roboflow.read_text(encoding="utf-8")):
        raise ValueError("Source license or export version differs from the inspected v1 release")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty to prevent stale labels")

    entries = []
    used_names: set[str] = set()
    for original_split in ("train", "valid", "test"):
        images = source / original_split / "images"
        labels = source / original_split / "labels"
        if not images.is_dir() or not labels.is_dir():
            raise ValueError(f"Missing source split: {original_split}")
        for image in sorted(images.glob("*")):
            match = NAME.fullmatch(image.name)
            if match is None or match.group(1) not in GROUPS:
                raise ValueError(f"Cannot determine source video for {image}")
            label = labels / (image.stem + ".txt")
            if not label.is_file() or image.name in used_names:
                raise ValueError(f"Missing label or duplicate filename: {image}")
            used_names.add(image.name)
            entries.append((image, label, match.group(1), GROUPS[match.group(1)]))
        if len(list(labels.glob("*.txt"))) != len(list(images.glob("*"))):
            raise ValueError(f"Unmatched annotations in source {original_split}")
    if len(entries) != 8053 or set(group for _, _, group, _ in entries) != set(GROUPS):
        raise ValueError("Expected exactly 8053 frames across source videos 1, 2 and 3")

    # Copying preserves an independently usable dataset when the source clone is removed.
    counts: dict[str, Counter] = {s: Counter() for s in GROUPS.values()}
    fingerprint = hashlib.sha256()
    for image, label, group, split in sorted(entries, key=lambda row: row[0].name):
        image_dest = output / "images" / split / image.name
        label_dest = output / "labels" / split / label.name
        image_dest.parent.mkdir(parents=True, exist_ok=True)
        label_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, image_dest)
        shutil.copy2(label, label_dest)
        for path in (image_dest, label_dest):
            fingerprint.update(str(path.relative_to(output)).encode("utf-8") + b"\0")
            fingerprint.update(bytes.fromhex(sha256(path)))
        counts[split][group] += 1

    summary = validate_dataset(output)
    write_dataset_yaml(output)
    manifest = {
        "schema_version": 1,
        "upstream": UPSTREAM,
        "license_declared_by_publisher": "CC BY 4.0",
        "attribution": "Mathieu Cartron, Shuttlecock Dataset (Roboflow Universe, 2022), version 1",
        "source_mirror_revision": mirror_revision,
        "source_metadata_sha256": {name: sha256(source / name) for name in
                                   ("README.dataset.txt", "README.roboflow.txt")},
        "split_policy": "Whole video_label_N source groups; published random splits discarded",
        "source_group_to_split": GROUPS,
        "split_groups_and_images": {split: dict(groups) for split, groups in counts.items()},
        "splits": {split: vars(item) for split, item in summary.items()},
        "dataset_sha256": fingerprint.hexdigest(),
        "limitations": [
            "Three source-video groups only; test estimates have high domain variance.",
            "Box sizes are nearly constant within each group; review label quality before accuracy claims.",
            "Published frames show professional broadcasts; the upstream media rights are not independently verified.",
            "Do not redistribute the images or labels from this repository.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Extracted Roboflow v1 YOLO export")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mirror-revision", required=True, help="Exact Git commit or source archive SHA256")
    args = parser.parse_args()
    try:
        manifest = prepare(args.source, args.output, args.mirror_revision)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Dataset preparation failed: {exc}\n")
    print(f"Prepared {manifest['splits']} at {args.output}")


if __name__ == "__main__":
    main()
