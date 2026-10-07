"""Ground-truth scoring rejects incomplete reviews and duplicate matching."""

from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from badminton_ai.event_evaluation import evaluate_candidates


class EventEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.predictions = {
            "input_sha256": "a" * 64,
            "source_reported_fps": 10,
            "processed_frames": 20,
            "events": [{"type": "ShotCandidate", "payload": {"frame_index": frame}}
                       for frame in (2, 10, 11)],
        }
        self.annotations = {
            "annotation_schema_version": 1,
            "input_sha256": "a" * 64,
            "source_reported_fps": 10,
            "review_status": "complete",
            "reviewer": "human-reviewer",
            "reviewed_frames": [0, 19],
            "shot_frame_indices": [3, 10, 12],
        }

    def test_one_to_one_matching_and_timing(self):
        result = evaluate_candidates(self.predictions, self.annotations, 0.1)
        self.assertEqual((result["true_positives"], result["false_positives"],
                          result["false_negatives"]), (3, 0, 0))
        self.assertEqual([m["signed_error_ms"] for m in result["matches"]], [-100, 0, -100])
        self.assertEqual(result["absolute_timing_error_p50_ms"], 100)

    def test_match_count_precedes_closest_individual_match(self):
        self.predictions["events"] = [
            {"type": "ShotCandidate", "payload": {"frame_index": frame}} for frame in (1, 2)]
        self.annotations["shot_frame_indices"] = [2, 3]
        result = evaluate_candidates(self.predictions, self.annotations, 0.1)
        self.assertEqual(result["true_positives"], 2)
        self.assertEqual([(m["candidate_frame"], m["shot_frame"]) for m in result["matches"]],
                         [(1, 2), (2, 3)])

    def test_mismatch_and_incomplete_review_fail_closed(self):
        for key, value in (("input_sha256", "b" * 64), ("reviewed_frames", [0, 18]),
                           ("review_status", "partial"), ("reviewer", ""),
                           ("shot_frame_indices", [3, 3])):
            with self.subTest(key=key):
                changed = dict(self.annotations, **{key: value})
                with self.assertRaises(ValueError):
                    evaluate_candidates(self.predictions, changed, 0.1)

    def test_empty_positive_class_is_undefined(self):
        self.predictions["events"] = []
        self.annotations["shot_frame_indices"] = []
        result = evaluate_candidates(self.predictions, self.annotations, 0.1)
        self.assertIsNone(result["precision"])
        self.assertIsNone(result["recall"])
        self.assertIsNone(result["f1"])
        self.assertIsNone(result["absolute_timing_error_p95_ms"])

    def test_cli_template_requires_human_completion_before_scoring(self):
        script = Path(__file__).resolve().parents[1] / "scripts/evaluate_shot_candidates.py"
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            predicted = directory / "predictions.json"
            template = directory / "review.json"
            score = directory / "score.json"
            predicted.write_text(json.dumps(self.predictions), encoding="utf-8")
            created = subprocess.run([sys.executable, str(script), "--predictions", str(predicted),
                                      "--template", "--output", str(template)], capture_output=True)
            self.assertEqual(created.returncode, 0, created.stderr.decode())
            self.assertEqual(json.loads(template.read_text())["candidate_frames_for_review"], [2, 10, 11])
            rejected = subprocess.run([sys.executable, str(script), "--predictions", str(predicted),
                                       "--annotations", str(template), "--output", str(score)],
                                      capture_output=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertFalse(score.exists())


if __name__ == "__main__":
    unittest.main()
