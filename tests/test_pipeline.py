"""Concurrency and failure tests without camera, model or recorded video."""

from pathlib import Path
import sys
import time
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from badminton_ai.pipeline import VisionPipeline


class BurstSource:
    def __init__(self, total: int, fail_at: int | None = None):
        self.total = total
        self.fail_at = fail_at
        self.index = 0
        self.closed = False

    def read(self):
        if self.fail_at == self.index:
            raise OSError("decoder failed")
        if self.closed or self.index == self.total:
            return False, None
        result = self.index
        self.index += 1
        return True, result

    def close(self):
        self.closed = True


class PipelineTests(unittest.TestCase):
    def test_slow_inference_drops_old_frames_and_keeps_latest(self):
        source = BurstSource(100)
        seen = []

        def infer(value):
            time.sleep(0.002)
            return value

        metrics = VisionPipeline(source, infer,
                                 on_result=lambda frame, result: seen.append((frame.sequence, result)),
                                 queue_size=2).run()
        self.assertTrue(source.closed)
        self.assertGreater(metrics["dropped_frames"], 0)
        self.assertEqual(metrics["captured_frames"], 100)
        self.assertEqual(metrics["processed_frames"] + metrics["dropped_frames"], 100)
        self.assertLessEqual(metrics["max_queue_depth"], 2)
        self.assertEqual(seen[-1], (99, 99))
        self.assertIsNone(metrics["camera_exposure_to_output_ms"])
        self.assertGreater(metrics["read_to_output_ms"]["p95"], 0)

    def test_capture_failure_surfaces_and_closes_source(self):
        source = BurstSource(100, fail_at=3)
        with self.assertRaisesRegex(RuntimeError, "Capture worker failed") as raised:
            VisionPipeline(source, lambda value: value).run()
        self.assertIsInstance(raised.exception.__cause__, OSError)
        self.assertTrue(source.closed)

    def test_inference_failure_shuts_down_capture(self):
        source = BurstSource(50)

        def fail(_value):
            raise ValueError("bad inference output")

        with self.assertRaisesRegex(ValueError, "bad inference output"):
            VisionPipeline(source, fail).run()
        self.assertTrue(source.closed)


if __name__ == "__main__":
    unittest.main()
