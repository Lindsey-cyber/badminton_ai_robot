"""Tests for the extracted image-plane tracking logic."""

import math
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from badminton_ai.tracking import BallKalmanFilter, SpeedEstimator


class TrackingTests(unittest.TestCase):
    def test_missing_detection_advances_uncertainty(self):
        tracker = BallKalmanFilter()
        self.assertEqual(tracker.predict_next(), [])
        tracker.update((10, 20))
        tracker.update((12, 20))
        previous_covariance = tracker.P.copy()
        previous_x = tracker.state[0]
        tracker.advance_without_measurement()
        self.assertGreater(tracker.state[0], previous_x)
        self.assertGreater(tracker.P[0, 0], previous_covariance[0, 0])
        self.assertEqual(len(tracker.predict_next(3)), 3)

    def test_speed_uses_elapsed_frames_and_does_not_claim_3d_speed(self):
        estimator = SpeedEstimator(fps=30)
        estimator.update((0, 0), frame_index=0)
        result = estimator.update((10, 0), frame_index=3)
        self.assertAlmostEqual(result["pixel_speed"], 100)
        self.assertIsNone(result["speed_kmh"])

        estimator.calibrate_from_court(610, 6.1)
        converted = estimator.update((20, 0), frame_index=6)
        self.assertAlmostEqual(converted["speed_ms"], 1.0)
        self.assertAlmostEqual(converted["speed_kmh"], 3.6)

    def test_rejects_invalid_time_and_positions(self):
        with self.assertRaises(ValueError):
            SpeedEstimator(fps=0)
        estimator = SpeedEstimator(fps=30)
        estimator.update((0, 0), frame_index=2)
        with self.assertRaises(ValueError):
            estimator.update((1, 0), frame_index=2)
        with self.assertRaises(ValueError):
            estimator.update((math.nan, 0), frame_index=3)
        tracker = BallKalmanFilter()
        with self.assertRaises(RuntimeError):
            tracker.advance_without_measurement()
        with self.assertRaises(ValueError):
            tracker.update((float("inf"), 0))


if __name__ == "__main__":
    unittest.main()
