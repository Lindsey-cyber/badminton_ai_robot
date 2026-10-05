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
        detector.iou_threshold = 0.45
        detector.target_classes = module.TARGET_CLASS_IDS

        output = np.zeros((1, 84, 2), dtype=np.float32)
        output[0, :4, :] = np.array([[50, 100], [50, 100], [10, 10], [10, 10]])
        output[0, 4, 0] = 0.99   # High-confidence COCO person.
        output[0, 4 + 32, 1] = 0.6  # Lower-confidence COCO sports ball.
        detections = detector.postprocess([output], 1.0, 0, 0, 200, 200)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["class_id"], 32)

        # A one-class raw export uses class 0 only when explicitly selected.
        detector.target_classes = [0]
        one_class = np.array([[[50], [50], [10], [10], [0.8]]], dtype=np.float32)
        shuttle = detector.postprocess([one_class], 1.0, 0, 0, 200, 200)
        self.assertEqual(len(shuttle), 1)
        self.assertEqual(shuttle[0]["class_id"], 0)

    def test_raw_predictions_suppress_same_class_overlap(self):
        script = Path(__file__).resolve().parents[1] / "scripts/demo_badminton_detection.py"
        spec = importlib.util.spec_from_file_location("ball_nms_test", script)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"onnxruntime": types.ModuleType("onnxruntime")}):
            spec.loader.exec_module(module)
        detector = module.YOLOv8Detector.__new__(module.YOLOv8Detector)
        detector.conf_threshold = 0.2
        detector.iou_threshold = 0.45
        detector.target_classes = [32]
        output = np.zeros((1, 84, 4), dtype=np.float32)
        output[0, :4, :] = np.array([[50, 51, 150, 300], [50, 51, 150, 300],
                                      [20, 20, 20, 20], [20, 20, 20, 20]])
        output[0, 4 + 32, :] = [0.9, 0.8, 0.7, 0.99]
        output[0, 4, 0] = 0.98  # A person score must not hide the ball score.
        results = detector.postprocess([output], 1.0, 0, 0, 200, 200)
        self.assertEqual([round(d["score"], 1) for d in results], [0.9, 0.7])
        self.assertEqual(results[0]["center"], (50.0, 50.0))
        with self.assertRaisesRegex(ValueError, "outside"):
            detector.target_classes = [84]
            detector.postprocess([output], 1.0, 0, 0, 200, 200)


if __name__ == "__main__":
    unittest.main()
