"""The COCO baseline must not accidentally evaluate original class-0 labels."""

import importlib.util
from pathlib import Path
import tempfile
import unittest


class BaselineMappingTests(unittest.TestCase):
    def test_images_use_mapped_label_path(self):
        script = Path(__file__).resolve().parents[1] / "scripts/evaluate_pretrained_shuttle.py"
        spec = importlib.util.spec_from_file_location("pretrained_mapping", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            image = source / "images/val/frame.jpg"
            label = source / "labels/val/frame.txt"
            image.parent.mkdir(parents=True)
            label.parent.mkdir(parents=True)
            image.write_bytes(b"frame bytes")
            label.write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
            workspace = Path(temp) / "mapped"
            yaml = module.create_mapped_evaluation(source, workspace, "val",
                                                    {i: str(i) for i in range(80)})
            mapped_image = workspace / "images/val/frame.jpg"
            self.assertFalse(mapped_image.is_symlink())
            self.assertEqual((workspace / "labels/val/frame.txt").read_text(),
                             "32 0.5 0.5 0.1 0.1\n")
            self.assertEqual(label.read_text(), "0 0.5 0.5 0.1 0.1\n")
            self.assertIn("val: images/val", yaml.read_text())


if __name__ == "__main__":
    unittest.main()
