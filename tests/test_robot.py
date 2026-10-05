"""Deterministic command and safety behavior of the software robot."""

from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from badminton_ai.robot import RobotState, SimulatedRobot


class RobotTests(unittest.TestCase):
    def test_queued_move_then_launch_advances_in_time(self):
        robot = SimulatedRobot(move_speed_m_s=2, launch_duration_s=0.25)
        self.assertFalse(robot.launch("early").accepted)
        self.assertTrue(robot.move_to(2, 0, "move").accepted)
        self.assertTrue(robot.launch("launch").accepted)
        self.assertEqual(robot.get_state().queue_depth, 2)

        snapshot = robot.advance(0.5)
        self.assertEqual(snapshot.state, RobotState.MOVING)
        self.assertEqual(snapshot.position_m, (1, 0))
        self.assertEqual(snapshot.active_command_id, "move")

        snapshot = robot.advance(0.6)
        self.assertEqual(snapshot.position_m, (2, 0))
        self.assertEqual(snapshot.state, RobotState.LAUNCHING)
        self.assertEqual(snapshot.active_command_id, "launch")
        self.assertEqual(robot.advance(0.15).state, RobotState.READY)

    def test_duplicate_and_conflicting_id(self):
        robot = SimulatedRobot()
        original = robot.move_to(1, 1, "move")
        duplicate = robot.move_to(1, 1, "move")
        self.assertTrue(duplicate.duplicate)
        self.assertEqual(duplicate.accepted, original.accepted)
        self.assertEqual(robot.get_state().queue_depth, 1)
        with self.assertRaisesRegex(ValueError, "reused"):
            robot.move_to(2, 1, "move")

    def test_queue_full_and_stop_preempts_active_motion(self):
        robot = SimulatedRobot(queue_capacity=1)
        robot.move_to(3, 0, "moving")
        robot.advance(0.5)
        robot.move_to(4, 0, "queued")
        self.assertFalse(robot.move_to(5, 0, "overflow").accepted)
        position = robot.get_state().position_m
        self.assertTrue(robot.stop("stop").accepted)
        self.assertEqual(robot.get_state().state, RobotState.IDLE)
        self.assertEqual(robot.get_state().queue_depth, 0)
        self.assertEqual(robot.advance(100).position_m, position)

    def test_emergency_stop_latches_and_clears_commands(self):
        robot = SimulatedRobot()
        robot.move_to(3, 0, "move")
        robot.advance(0.2)
        robot.launch("launch")
        self.assertTrue(robot.emergency_stop("e1").accepted)
        self.assertTrue(robot.emergency_stop("e2").accepted)
        self.assertEqual(robot.advance(100).state, RobotState.EMERGENCY_STOP)
        self.assertEqual(robot.get_state().queue_depth, 0)
        self.assertFalse(robot.move_to(1, 0, "blocked").accepted)
        self.assertFalse(robot.stop("cannot-clear-emergency").accepted)
        robot.reset_emergency_stop()
        self.assertEqual(robot.get_state().state, RobotState.IDLE)
        self.assertTrue(robot.move_to(1, 0, "new-command").accepted)

    def test_invalid_inputs_do_not_enqueue(self):
        robot = SimulatedRobot()
        with self.assertRaises(ValueError):
            robot.move_to(float("nan"), 1, "bad")
        with self.assertRaises(ValueError):
            robot.move_to(7, 1, "outside")
        with self.assertRaises(ValueError):
            robot.advance(-1)
        with self.assertRaises(ValueError):
            robot.stop("")
        self.assertEqual(robot.get_state().queue_depth, 0)


if __name__ == "__main__":
    unittest.main()
