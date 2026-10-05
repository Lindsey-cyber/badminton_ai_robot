"""Triangulation units and rejection of synthetic calibration for measurement."""

from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

try:
    import numpy as np
    import stereo_distance_demo as stereo
except ImportError:
    stereo = None


@unittest.skipIf(stereo is None, "Vision dependencies required for stereo tests")
class StereoTests(unittest.TestCase):
    def test_disparity_depth_and_coordinate_units(self):
        self.assertAlmostEqual(stereo.disparity_to_depth(48, 800, 0.12), 2)
        self.assertEqual(stereo.depth_to_real_coords(680, 360, 2, 640, 360, 800),
                         (0.1, 0.0, 2))
        self.assertEqual(stereo.disparity_to_depth(0, 800, 0.12), float("inf"))

    def test_rejects_example_calibration_for_physical_measurement(self):
        if not stereo.CV2_AVAILABLE:
            self.skipTest("OpenCV required")
        image = np.zeros((720, 1280), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "Synthetic calibration"):
            stereo.rectify_pair(image, image, stereo.EXAMPLE_CALIB)


if __name__ == "__main__":
    unittest.main()
