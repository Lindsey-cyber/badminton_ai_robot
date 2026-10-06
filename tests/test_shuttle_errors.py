"""Greedy error triage relies on correct box overlap at tiny scales."""

import importlib.util
from pathlib import Path
import unittest


class ShuttleErrorTests(unittest.TestCase):
    def test_iou_handles_overlap_and_disjoint_small_boxes(self):
        script = Path(__file__).resolve().parents[1] / "scripts/analyze_shuttle_errors.py"
        spec = importlib.util.spec_from_file_location("shuttle_errors", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.iou((10, 10, 15, 15), (10, 10, 15, 15)), 1)
        self.assertEqual(module.iou((10, 10, 15, 15), (20, 20, 25, 25)), 0)
        self.assertAlmostEqual(module.iou((0, 0, 4, 4), (2, 2, 6, 6)), 4 / 28)


if __name__ == "__main__":
    unittest.main()
