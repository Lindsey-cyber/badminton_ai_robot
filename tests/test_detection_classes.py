"""Guard against confusing COCO person (0) with a custom shuttle class (0)."""

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np


class DetectionClassTests(unittest.TestCase):
    def test_generic_model_filters_people_out(self):
        script = Path(__file__).resolve().parents[1] / "scripts/demo_badminton_detection.py"
        spec = importlib.util.spec_from_file_location("demo_badminton_detection_test", script)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"onnxruntime": types.ModuleType("onnxruntime")}):
            spec.loader.exec_module(module)

        self.assertEqual(module.TARGET_CLASS_IDS, [32])
        self.assertNotEqual(module.BALL_CLASS_IDS.get(0), "shuttlecock")
        detector = module.YOLOv8Detector.__new__(module.YOLOv8Detector)
        detector.conf_threshold = 0.2
        detector.target_classes = module.TARGET_CLASS_IDS

        output = np.zeros((1, 84, 2), dtype=np.float32)
        output[0, :4, :] = np.array([[50, 100], [50, 100], [10, 10], [10, 10]])
        output[0, 4, 0] = 0.99   # High-confidence COCO person.
        output[0, 4 + 32, 1] = 0.6  # Lower-confidence COCO sports ball.
        detections = detector.postprocess([output], 1.0, 0, 0, 200, 200)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["class_id"], 32)


if __name__ == "__main__":
    unittest.main()
