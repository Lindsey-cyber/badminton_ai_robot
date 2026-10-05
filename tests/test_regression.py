"""A change in measurement boundary must not silently pass a benchmark gate."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from badminton_ai.regression import compare_reports


class RegressionTests(unittest.TestCase):
    def baseline(self):
        return {"platform": "pi", "python": "3.12", "runtime_versions": {"onnxruntime": "x"},
                "mode": "pose", "model_sha256": "abc", "input": "video.mp4",
                "source_reported_fps": 24, "queue_capacity": 2, "warmup_frames": 10,
                "captured_frames": 200, "processed_fps": 20,
                "read_to_output_ms": {"p95": 100}, "process_rss_mb": {"p95": 200}}

    def test_detects_regressions_and_allows_small_variation(self):
        old = self.baseline()
        new = deepcopy(old)
        new["processed_fps"] = 18
        new["read_to_output_ms"]["p95"] = 115
        self.assertTrue(compare_reports(old, new)["passed"])
        new["processed_fps"] = 16
        new["read_to_output_ms"]["p95"] = 125
        new["process_rss_mb"]["p95"] = 245
        result = compare_reports(old, new)
        self.assertFalse(result["passed"])
        self.assertTrue(all(metric["regressed"] for metric in result["metrics"].values()))

    def test_rejects_environment_and_boundary_mismatch(self):
        old = self.baseline()
        new = deepcopy(old)
        new["model_sha256"] = "other"
        with self.assertRaisesRegex(ValueError, "model_sha256"):
            compare_reports(old, new)
        new = deepcopy(old)
        new.pop("read_to_output_ms")
        new["read_start_to_result_ms"] = {"p95": 100}
        with self.assertRaisesRegex(ValueError, "latency boundaries"):
            compare_reports(old, new)
        new = deepcopy(old)
        new["captured_frames"] = 100
        with self.assertRaisesRegex(ValueError, "frame count"):
            compare_reports(old, new)


if __name__ == "__main__":
    unittest.main()
