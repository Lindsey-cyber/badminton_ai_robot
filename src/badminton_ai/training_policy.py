"""Small simulation-only rule connecting pose candidates to robot commands.

Candidate wrist peaks are unverified. This policy must never control hardware.
Video frame time, rather than host scheduling, advances the robot simulation.
"""

import math

from .events import Event, EventType, SessionEventEmitter
from .robot import RobotState, SimulatedRobot


class SimulatedTrainingPolicy:
    """Move to the side opposite the observed wrist, then simulate one launch."""

    def __init__(self, robot: SimulatedRobot, emitter: SessionEventEmitter, fps: float) -> None:
        if not isinstance(robot, SimulatedRobot):
            raise TypeError("This unverified policy only accepts SimulatedRobot")
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be positive and finite")
        self.robot = robot
        self.emitter = emitter
        self.fps = fps
        self._last_frame = -1
        self._last_state = robot.get_state().state
        self._pending_launch_for: str | None = None
        self._seen_candidates: set[str] = set()
        self.command_count = 0

    def _state_event(self) -> None:
        snapshot = self.robot.get_state()
        if snapshot.state == self._last_state:
            return
        self._last_state = snapshot.state
        self.emitter.emit(EventType.ROBOT_STATE_CHANGED, {
            "state": snapshot.state.value,
            "x_m": round(snapshot.position_m[0], 4),
            "y_m": round(snapshot.position_m[1], 4),
            "queue_depth": snapshot.queue_depth,
            "active_command_id": snapshot.active_command_id or "",
        })

    def _command_event(self, source_event_id: str, kind: str, command_id: str,
                       accepted: bool, reason: str = "", x_m: float = 0.0,
                       y_m: float = 0.0) -> None:
        self.emitter.emit(EventType.ROBOT_COMMAND_ISSUED, {
            "source_event_id": source_event_id,
            "command_id": command_id,
            "kind": kind,
            "accepted": int(accepted),
            "reason": reason,
            "target_x_m": x_m,
            "target_y_m": y_m,
        })
        self.command_count += 1

    def on_candidate(self, event: Event) -> None:
        if event.type != EventType.SHOT_CANDIDATE or event.session_id != self.emitter.session_id:
            raise ValueError("Expected a ShotCandidate in the active session")
        if event.event_id in self._seen_candidates:
            return
        self._seen_candidates.add(event.event_id)
        state = self.robot.get_state()
        if self._pending_launch_for or state.queue_depth or state.state not in (
                RobotState.IDLE, RobotState.READY):
            return  # A previous drill is still in progress; no stale command queue.
        hand = event.payload.get("hand")
        if hand not in ("left", "right"):
            raise ValueError("Candidate hand must be left or right")
        x_m = self.robot.width_m * (0.75 if hand == "left" else 0.25)
        y_m = self.robot.length_m * 0.25
        command_id = f"{event.event_id}:move"
        ack = self.robot.move_to(x_m, y_m, command_id)
        self._command_event(event.event_id, "MOVE", command_id, ack.accepted,
                            ack.reason, x_m, y_m)
        if ack.accepted:
            self._pending_launch_for = event.event_id
            self.robot.advance(0)
            self._state_event()

    def _launch_if_ready(self) -> None:
        if self._pending_launch_for is None or self.robot.get_state().state != RobotState.READY:
            return
        source_event_id = self._pending_launch_for
        self._pending_launch_for = None
        command_id = f"{source_event_id}:launch"
        ack = self.robot.launch(command_id)
        self._command_event(source_event_id, "LAUNCH", command_id, ack.accepted, ack.reason)
        if ack.accepted:
            self.robot.advance(0)
            self._state_event()

    def on_frame(self, frame_index: int) -> None:
        if frame_index <= self._last_frame:
            raise ValueError("frame_index must strictly increase")
        if self._last_frame >= 0:
            self.robot.advance((frame_index - self._last_frame) / self.fps)
            self._state_event()
            self._launch_if_ready()
        self._last_frame = frame_index

    def finish(self) -> None:
        """Finish outstanding simulator work after replay, using virtual time."""
        if self._pending_launch_for:
            self.robot.advance(math.hypot(self.robot.width_m, self.robot.length_m)
                               / self.robot.move_speed_m_s)
            self._state_event()
            self._launch_if_ready()
        if self.robot.get_state().state == RobotState.LAUNCHING:
            self.robot.advance(self.robot.launch_duration_s)
            self._state_event()
