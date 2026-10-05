"""Dataset validation catches annotation errors and exact split leakage."""

from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from badminton_ai.dataset import SPLITS, validate_dataset, write_dataset_yaml

try:
    from PIL import Image
except ImportError:
    Image = None


@unittest.skipIf(Image is None, "Pillow is required for dataset validation")
class DatasetTests(unittest.TestCase):
    def make_dataset(self, root: Path) -> None:
        for index, split in enumerate(SPLITS):
            images = root / "images" / split
            labels = root / "labels" / split
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            Image.new("RGB", (10, 10), (index * 50, 0, 0)).save(images / "a.png")
            (labels / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    def test_valid_splits_and_yaml(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_dataset(root)
            summary = validate_dataset(root)
            self.assertEqual(summary["test"].boxes, 1)
            self.assertIn("0: shuttlecock", write_dataset_yaml(root).read_text())

    def test_bad_box_missing_label_and_split_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_dataset(root)
            label = root / "labels/train/a.txt"
            label.write_text("0 0.95 0.5 0.2 0.2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid one-class"):
                validate_dataset(root)
            label.unlink()
            with self.assertRaisesRegex(ValueError, "Mismatched labels"):
                validate_dataset(root)
            label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            (root / "images/val/a.png").write_bytes((root / "images/train/a.png").read_bytes())
            with self.assertRaisesRegex(ValueError, "reused across"):
                validate_dataset(root)


if __name__ == "__main__":
    unittest.main()
