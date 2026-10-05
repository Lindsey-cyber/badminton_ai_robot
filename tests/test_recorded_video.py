"""Exercise the checked-in recording and pose model through the bounded pipeline."""

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from badminton_ai.pipeline import VideoFileSource, VisionPipeline


class RecordedVideoTests(unittest.TestCase):
    def test_checked_in_video_replays_with_real_pose_model(self):
        video = ROOT / "assets/demo_inputs/badminton_sample.mp4"
        model = ROOT / "src/pose/yolov8n-pose.onnx"
        if not video.is_file() or not model.is_file():
            self.skipTest("Checked-in video/model are unavailable in this source checkout")
        from benchmark import make_detector

        detector = make_detector("pose", model)
        seen = []
        pipeline = VisionPipeline(
            VideoFileSource(video, max_frames=24),
            lambda frame: detector.inference(frame)[0],
            on_result=lambda frame, people: seen.append((frame.sequence, len(people))),
            queue_size=2,
        )
        metrics = pipeline.run()
        self.assertEqual(metrics["captured_frames"], 24)
        self.assertGreater(metrics["processed_frames"], 0)
        self.assertEqual(metrics["processed_frames"], len(seen))
        self.assertEqual(metrics["processed_frames"] + metrics["dropped_frames"], 24)
        self.assertEqual([sequence for sequence, _ in seen],
                         sorted(sequence for sequence, _ in seen))
        self.assertLessEqual(metrics["max_queue_depth"], 2)
        self.assertGreater(metrics["read_to_output_ms"]["p95"], 0)


if __name__ == "__main__":
    unittest.main()
