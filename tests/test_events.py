"""Shot candidates and local event ordering without labeled video claims."""

from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from badminton_ai.events import (CandidateEventProcessor, EventDispatcher, EventType,
                                 SessionEventEmitter, ShotCandidateDetector, Wrist)


class EventTests(unittest.TestCase):
    def test_local_peak_is_reported_one_frame_later(self):
        detector = ShotCandidateDetector(fps=10, min_wrist_speed_px_s=30, min_gap_s=0.5)
        positions = [0, 1, 5, 6, 7, 11, 12]
        candidates = [detector.observe(i, Wrist(x, 0, 1), Wrist(0, 0, 1))
                      for i, x in enumerate(positions)]
        self.assertIsNone(candidates[2])  # A future frame is needed to confirm the peak.
        self.assertEqual(candidates[3].frame_index, 2)
        self.assertAlmostEqual(candidates[3].wrist_speed_px_s, 40)
        self.assertIsNone(candidates[6])  # Frame 5 is within the five-frame gap.

    def test_missing_pose_breaks_speed_continuity(self):
        detector = ShotCandidateDetector(10, 30)
        detector.observe(0, Wrist(0, 0, 1), Wrist(0, 0, 1))
        detector.observe(1, None, None)
        self.assertIsNone(detector.observe(2, Wrist(1000, 0, 1), Wrist(0, 0, 1)))
        with self.assertRaises(ValueError):
            detector.observe(2, Wrist(0, 0, 1), Wrist(0, 0, 1))
        with self.assertRaises(ValueError):
            Wrist(float("nan"), 0, 1)

    def test_session_order_and_handler_failure(self):
        received = []
        dispatcher = EventDispatcher()
        dispatcher.subscribe(EventType.SHOT_CANDIDATE, received.append)
        emitter = SessionEventEmitter("session-1", dispatcher)
        first = emitter.emit(EventType.SHOT_CANDIDATE, {"frame_index": 2})
        second = emitter.emit(EventType.SHOT_CANDIDATE, {"frame_index": 8})
        self.assertEqual([event.sequence for event in received], [0, 1])
        self.assertNotEqual(first.event_id, second.event_id)
        self.assertIsNotNone(first.timestamp_utc.utcoffset())

        dispatcher.subscribe(EventType.SHOT_CANDIDATE, lambda _event: (_ for _ in ()).throw(RuntimeError("sink failed")))
        with self.assertRaisesRegex(RuntimeError, "sink failed"):
            emitter.emit(EventType.SHOT_CANDIDATE, {"frame_index": 12})
        self.assertEqual(received[-1].sequence, 2)
        with self.assertRaises(RuntimeError):
            emitter.emit(EventType.SHOT_CANDIDATE, {"frame_index": 15})
        self.assertEqual(received[-1].sequence, 3)

    def test_pose_candidates_are_emitted_with_source_frame_numbers(self):
        received = []
        dispatcher = EventDispatcher()
        dispatcher.subscribe(EventType.SHOT_CANDIDATE, received.append)
        processor = CandidateEventProcessor(
            ShotCandidateDetector(10, 30), SessionEventEmitter("s", dispatcher))

        def person(x):
            keypoints = [(0, 0, 1)] * 17
            keypoints[9] = (x, 0, 1)
            return {"bbox": [0, 0, 100, 100], "keypoints": keypoints}

        for frame, x in ((0, 0), (1, 1), (2, 5), (3, 6)):
            processor.process(frame, [person(x)])
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].payload["frame_index"], 2)
        self.assertEqual(received[0].sequence, 0)


if __name__ == "__main__":
    unittest.main()
