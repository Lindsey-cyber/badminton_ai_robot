"""Tests of baseline measurement semantics; no camera or ONNX runtime required."""

import argparse
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import benchmark
from scripts import video_batch_infer


class FakeCapture:
    def __init__(self, frames):
        self.frames = iter(frames)
        self.released = False

    def isOpened(self):
        return True

    def get(self, _property):
        return 30.0

    def read(self):
        try:
            return True, next(self.frames)
        except StopIteration:
            return False, None

    def release(self):
        self.released = True


class FakeDetector:
    session = types.SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])

    def inference(self, frame):
        return [], 2.0


class BenchmarkTests(unittest.TestCase):
    def test_percentiles_interpolate(self):
        self.assertEqual(benchmark.percentile([1.0, 2.0, 3.0, 4.0], 50), 2.5)
        self.assertAlmostEqual(benchmark.percentile([1.0, 2.0, 3.0, 4.0], 95), 3.85)
        with self.assertRaises(ValueError):
            benchmark.summarize([])

    def test_video_end_is_not_a_dropped_frame_or_read_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.onnx"
            model.write_bytes(b"model fixture")
            capture = FakeCapture([1, 2, 3])  # First frame is warmup.
            fake_cv2 = types.SimpleNamespace(VideoCapture=lambda _: capture, CAP_PROP_FPS=5)
            args = argparse.Namespace(mode="pose", model=model, input=Path("video.mp4"),
                                      camera=None, warmup=1, frames=10)
            with patch.dict(sys.modules, {"cv2": fake_cv2}), patch.object(benchmark, "make_detector", return_value=FakeDetector()):
                result = benchmark.run(args)
            self.assertTrue(capture.released)
            self.assertEqual(result["processed_frames"], 2)
            self.assertEqual(result["read_failures"], 0)
            self.assertIsNone(result["dropped_frame_rate"])
            self.assertIsNone(result["camera_capture_to_result_ms"])
            self.assertEqual(result["ort_run_ms"]["p95"], 2.0)
            self.assertGreaterEqual(result["read_start_to_result_ms"]["p95"], 0)

    def test_legacy_batch_does_not_publish_fps_after_inference_error(self):
        capture = FakeCapture([object()])
        fake_cv2 = types.SimpleNamespace(VideoCapture=lambda _: capture, CAP_PROP_FPS=5,
                                         CAP_PROP_FRAME_WIDTH=6, CAP_PROP_FRAME_HEIGHT=7,
                                         CAP_PROP_FRAME_COUNT=8)

        class BrokenDetector:
            def inference(self, _frame):
                raise RuntimeError("bad output shape")

        with patch.dict(sys.modules, {"cv2": fake_cv2}):
            result = video_batch_infer.process_single_video(
                Path("video.mp4"), BrokenDetector(), None, "pose", 1
            )
        self.assertEqual(result["error"], "bad output shape")
        self.assertNotIn("avg_fps", result)
        self.assertTrue(capture.released)


if __name__ == "__main__":
    unittest.main()
