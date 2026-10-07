"""A candidate flows through a conservative policy into the existing event log."""

from pathlib import Path
import sys
import tempfile
import time
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from badminton_ai.events import (CandidateEventProcessor, EventDispatcher, EventType,
                                 SessionEventEmitter, ShotCandidateDetector)
from badminton_ai.pipeline import VisionPipeline
from badminton_ai.robot import RobotState, SimulatedRobot
from badminton_ai.storage import EventStore
from badminton_ai.training_policy import SimulatedTrainingPolicy

class PolicyTests(unittest.TestCase):
    def test_bounded_pipeline_candidate_policy_simulator_store(self):
        class PacedFrames:
            def __init__(self):
                self.frame = 0

            def read(self):
                if self.frame >= 5:
                    return False, None
                time.sleep(0.02)
                position = [0, 1, 5, 6, 7][self.frame]
                self.frame += 1
                return True, position

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            with EventStore(Path(directory) / "events.sqlite3") as store:
                store.create_session("s")
                dispatcher = EventDispatcher()
                for kind in (EventType.SHOT_CANDIDATE, EventType.ROBOT_COMMAND_ISSUED,
                             EventType.ROBOT_STATE_CHANGED):
                    dispatcher.subscribe(kind, store.append)
                emitter = SessionEventEmitter("s", dispatcher)
                policy = SimulatedTrainingPolicy(SimulatedRobot(), emitter, fps=10)
                dispatcher.subscribe(EventType.SHOT_CANDIDATE, policy.on_candidate)
                processor = CandidateEventProcessor(ShotCandidateDetector(10, 30), emitter)

                def infer(position):
                    points = [(0, 0, 1)] * 17
                    points[9] = (position, 0, 1)
                    return [{"bbox": [0, 0, 100, 100], "keypoints": points}]

                def on_result(frame, detections):
                    policy.on_frame(frame.sequence)
                    processor.process(frame.sequence, detections)

                metrics = VisionPipeline(PacedFrames(), infer, on_result, queue_size=2).run()
                policy.finish()
                self.assertEqual(metrics["captured_frames"], 5)
                self.assertEqual(metrics["dropped_frames"], 0)
                self.assertEqual([e.type for e in store.read_events("s")][:2],
                                 [EventType.SHOT_CANDIDATE, EventType.ROBOT_COMMAND_ISSUED])
                self.assertEqual(policy.command_count, 2)

    def test_candidate_move_launch_and_ordered_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            with EventStore(Path(directory) / "events.sqlite3") as store:
                store.create_session("s")
                dispatcher = EventDispatcher()
                for kind in (EventType.SHOT_CANDIDATE, EventType.ROBOT_COMMAND_ISSUED,
                             EventType.ROBOT_STATE_CHANGED):
                    dispatcher.subscribe(kind, store.append)
                emitter = SessionEventEmitter("s", dispatcher)
                robot = SimulatedRobot(move_speed_m_s=4.0)
                policy = SimulatedTrainingPolicy(robot, emitter, fps=10)
                dispatcher.subscribe(EventType.SHOT_CANDIDATE, policy.on_candidate)

                policy.on_frame(0)
                candidate = emitter.emit(EventType.SHOT_CANDIDATE, {"hand": "left", "frame_index": 0})
                self.assertEqual(robot.get_state().state, RobotState.MOVING)
                policy.on_frame(1)
                policy.finish()
                self.assertEqual(robot.get_state().state, RobotState.READY)
                events = store.read_events("s")
                self.assertEqual([e.sequence for e in events], list(range(len(events))))
                commands = [e for e in events if e.type == EventType.ROBOT_COMMAND_ISSUED]
                self.assertEqual([e.payload["kind"] for e in commands], ["MOVE", "LAUNCH"])
                self.assertEqual([e.payload["source_event_id"] for e in commands],
                                 [candidate.event_id] * 2)
                self.assertTrue(all(e.payload["accepted"] == 1 for e in commands))
                self.assertAlmostEqual(robot.get_state().position_m[0], 6.1 * .75)
                self.assertIn("LAUNCHING", [e.payload["state"] for e in events
                                              if e.type == EventType.ROBOT_STATE_CHANGED])
                policy.on_candidate(candidate)
                self.assertEqual(policy.command_count, 2)  # Same event cannot replay the drill.

    def test_invalid_robot_and_frame_order_fail(self):
        emitter = SessionEventEmitter("s", EventDispatcher())
        with self.assertRaises(TypeError):
            SimulatedTrainingPolicy(object(), emitter, fps=24)
        policy = SimulatedTrainingPolicy(SimulatedRobot(), emitter, fps=24)
        policy.on_frame(0)
        with self.assertRaises(ValueError):
            policy.on_frame(0)


if __name__ == "__main__":
    unittest.main()
