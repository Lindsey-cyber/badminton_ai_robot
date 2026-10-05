"""Reject incompatible exported layouts before deployment."""

from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from badminton_ai.model_compat import check_detection_layout


class ModelCompatTests(unittest.TestCase):
    def test_one_class_raw_layout(self):
        check_detection_layout([1, 3, 640, 640], [1, 5, 8400], 640)
        with self.assertRaisesRegex(ValueError, "one-class"):
            check_detection_layout([1, 3, 640, 640], [1, 84, 8400], 640)
        with self.assertRaisesRegex(ValueError, "one-class"):
            check_detection_layout([1, 3, 640, 640], [1, 300, 6], 640)
        with self.assertRaisesRegex(ValueError, "fixed BCHW"):
            check_detection_layout([1, 3, "height", "width"], [1, 5, 8400], 640)


if __name__ == "__main__":
    unittest.main()
