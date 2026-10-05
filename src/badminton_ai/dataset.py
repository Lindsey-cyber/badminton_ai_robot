"""Validate a one-class YOLO shuttlecock dataset before training.

All three splits are held separately. Exact duplicate images across splits are
rejected, but near-duplicate video frames require grouping by recording during
annotation; hashes cannot detect that leakage.
"""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class SplitSummary:
    images: int
    boxes: int


def validate_dataset(root: Path) -> dict[str, SplitSummary]:
    from PIL import Image, UnidentifiedImageError

    root = root.resolve()
    summaries: dict[str, SplitSummary] = {}
    image_hashes: dict[str, str] = {}
    for split in SPLITS:
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise ValueError(f"Missing images/{split} or labels/{split} directory")
        images = sorted(path for path in image_dir.rglob("*")
                        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            raise ValueError(f"images/{split} is empty")
        expected_labels = {label_dir / path.relative_to(image_dir).with_suffix(".txt")
                           for path in images}
        actual_labels = set(label_dir.rglob("*.txt"))
        if expected_labels != actual_labels:
            missing = sorted(str(path) for path in expected_labels - actual_labels)
            extra = sorted(str(path) for path in actual_labels - expected_labels)
            raise ValueError(f"Mismatched labels/{split}: missing={missing[:3]}, extra={extra[:3]}")

        box_count = 0
        for image in images:
            try:
                with Image.open(image) as opened:
                    opened.verify()
            except (OSError, UnidentifiedImageError) as exc:
                raise ValueError(f"Unreadable image: {image}") from exc
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            if digest in image_hashes and image_hashes[digest] != split:
                raise ValueError(f"Exact image reused across {image_hashes[digest]} and {split}: {image}")
            image_hashes[digest] = split

            label = label_dir / image.relative_to(image_dir).with_suffix(".txt")
            for line_number, row in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
                tokens = row.split()
                try:
                    values = [float(token) for token in tokens]
                except ValueError as exc:
                    raise ValueError(f"Invalid YOLO annotation: {label}:{line_number}") from exc
                if (len(values) != 5 or tokens[0] != "0" or
                        not all(math.isfinite(value) for value in values) or
                        not all(0 <= value <= 1 for value in values[1:]) or
                        values[3] <= 0 or values[4] <= 0 or
                        values[1] - values[3] / 2 < 0 or values[1] + values[3] / 2 > 1 or
                        values[2] - values[4] / 2 < 0 or values[2] + values[4] / 2 > 1):
                    raise ValueError(f"Invalid one-class YOLO box: {label}:{line_number}")
                box_count += 1
        if box_count == 0:
            raise ValueError(f"labels/{split} has no shuttlecock boxes")
        summaries[split] = SplitSummary(len(images), box_count)
    return summaries


def write_dataset_yaml(root: Path) -> Path:
    """Write an absolute path after validate_dataset has succeeded."""
    path = root.resolve() / "data.yaml"
    path.write_text(f"path: {json.dumps(str(root.resolve()))}\n"
                    "train: images/train\nval: images/val\ntest: images/test\n"
                    "names:\n  0: shuttlecock\n", encoding="utf-8")
    return path
