"""Concurrency and failure tests without camera, model or recorded video."""

from pathlib import Path
import json
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from badminton_ai.pipeline import OpenCVCameraSource, VisionPipeline


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
    def test_camera_source_reports_negotiated_settings_and_fails_on_read_error(self):
        class Capture:
            def __init__(self, _index):
                self.read_count = 0
                self.released = False
                self.requested = {}

            def isOpened(self):
                return True

            def set(self, key, value):
                self.requested[key] = value

            def get(self, key):
                return {1: 640, 2: 480, 3: 24}[key]

            def read(self):
                self.read_count += 1
                return True, self.read_count

            def release(self):
                self.released = True

        capture = Capture(0)
        fake_cv2 = types.SimpleNamespace(VideoCapture=lambda _: capture,
                                         CAP_PROP_FRAME_WIDTH=1,
                                         CAP_PROP_FRAME_HEIGHT=2, CAP_PROP_FPS=3)
        with patch.dict(sys.modules, {"cv2": fake_cv2}):
            source = OpenCVCameraSource(0, max_frames=2, width=1280, height=720,
                                        requested_fps=30)
            self.assertEqual((source.reported_width, source.reported_height, source.fps),
                             (640, 480, 24))
            self.assertEqual(capture.requested, {1: 1280, 2: 720, 3: 30})
            self.assertEqual(source.read(), (True, 1))
            self.assertEqual(source.read(), (True, 2))
            self.assertEqual(source.read(), (False, None))
            source.close()
            self.assertTrue(capture.released)

            capture = Capture(0)
            capture.read = lambda: (False, None)
            source = OpenCVCameraSource(0, max_frames=1)
            with self.assertRaisesRegex(RuntimeError, "Camera read failed"):
                source.read()
            source.close()

    def test_camera_benchmark_excludes_warmup_and_writes_real_report_fields(self):
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "scripts"))
        from scripts import benchmark_pipeline

        class Capture:
            def __init__(self, _index):
                self.read_count = 0

            def isOpened(self):
                return True

            def set(self, _key, _value):
                return True

            def get(self, key):
                return {1: 640, 2: 480, 3: 24}[key]

            def read(self):
                self.read_count += 1
                return True, self.read_count

            def release(self):
                pass

        capture = Capture(0)
        fake_cv2 = types.SimpleNamespace(VideoCapture=lambda _: capture,
                                         CAP_PROP_FRAME_WIDTH=1,
                                         CAP_PROP_FRAME_HEIGHT=2, CAP_PROP_FPS=3)
        class Detector:
            session = types.SimpleNamespace(get_providers=lambda: ["fake-provider"])

            def inference(self, frame):
                return frame

        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.onnx"
            model.write_bytes(b"test-only")
            output = Path(directory) / "report.json"
            argv = ["benchmark_pipeline.py", "--camera", "0", "--model", str(model),
                    "--frames", "3", "--warmup", "1", "--width", "1280",
                    "--output", str(output)]
            with patch.dict(sys.modules, {"cv2": fake_cv2}), \
                    patch.object(sys, "argv", argv), \
                    patch.object(benchmark_pipeline, "make_detector", return_value=Detector()):
                self.assertEqual(benchmark_pipeline.main(), 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(capture.read_count, 4)
            self.assertEqual(report["captured_frames"], 3)
            self.assertEqual(report["warmup_frames"], 1)
            self.assertEqual(report["input_kind"], "camera")
            self.assertIsNone(report["input_sha256"])
            self.assertEqual(report["camera_settings"]["reported_width"], 640)

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

    def test_window_metrics_report_queue_and_latency(self):
        source = BurstSource(30)
        windows = []
        pipeline = VisionPipeline(source, lambda value: (time.sleep(0.002), value)[1],
                                  queue_size=2, on_metrics=windows.append,
                                  metrics_interval_s=0.001)
        result = pipeline.run()
        self.assertTrue(windows)
        self.assertEqual(windows[-1]["dropped_frames_total"], result["dropped_frames"])
        self.assertGreater(windows[-1]["inference_p95_ms"], 0)
        self.assertLessEqual(windows[-1]["queue_depth"], 2)
        self.assertGreater(windows[-1]["process_rss_mb"], 0)

    def test_metrics_sink_failure_stops_capture(self):
        source = BurstSource(30)

        def fail(_snapshot):
            raise RuntimeError("metrics sink failed")

        with self.assertRaisesRegex(RuntimeError, "metrics sink failed"):
            VisionPipeline(source, lambda value: value, on_metrics=fail,
                           metrics_interval_s=0.000001).run()
        self.assertTrue(source.closed)


if __name__ == "__main__":
    unittest.main()
