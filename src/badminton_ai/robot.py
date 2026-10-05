"""Deterministic robot simulator for software development and CI.

ACK means a command was accepted into the simulator, not that it completed.
No transport or physical motor behavior is implied.
"""

from collections import deque
from dataclasses import dataclass
from enum import Enum
import math
from typing import Protocol


class RobotState(str, Enum):
    IDLE = "IDLE"
    MOVING = "MOVING"
    READY = "READY"
    LAUNCHING = "LAUNCHING"
    EMERGENCY_STOP = "EMERGENCY_STOP"


class CommandKind(str, Enum):
    MOVE = "MOVE"
    LAUNCH = "LAUNCH"
    STOP = "STOP"
    EMERGENCY_STOP = "EMERGENCY_STOP"


@dataclass(frozen=True)
class RobotCommand:
    command_id: str
    kind: CommandKind
    target_m: tuple[float, float] | None = None


@dataclass(frozen=True)
class Ack:
    command_id: str
    accepted: bool
    state_at_ack: RobotState
    reason: str = ""
    duplicate: bool = False


@dataclass(frozen=True)
class RobotSnapshot:
    state: RobotState
    position_m: tuple[float, float]
    queue_depth: int
    active_command_id: str | None


class Robot(Protocol):
    def move_to(self, x_m: float, y_m: float, command_id: str) -> Ack: ...
    def launch(self, command_id: str) -> Ack: ...
    def stop(self, command_id: str) -> Ack: ...
    def emergency_stop(self, command_id: str) -> Ack: ...
    def get_state(self) -> RobotSnapshot: ...


class SimulatedRobot:
    """Advance time explicitly; commands update position and state, not GPIO."""

    def __init__(self, width_m: float = 6.1, length_m: float = 13.4,
                 move_speed_m_s: float = 1.0, launch_duration_s: float = 0.3,
                 queue_capacity: int = 16) -> None:
        if not all(math.isfinite(x) and x > 0 for x in
                   (width_m, length_m, move_speed_m_s, launch_duration_s)):
            raise ValueError("Dimensions, speed and launch duration must be positive and finite")
        if queue_capacity < 1:
            raise ValueError("queue_capacity must be positive")
        self.width_m = width_m
        self.length_m = length_m
        self.move_speed_m_s = move_speed_m_s
        self.launch_duration_s = launch_duration_s
        self.queue_capacity = queue_capacity
        self.state = RobotState.IDLE
        self.position_m = (0.0, 0.0)
        self._queue: deque[RobotCommand] = deque()
        self._active: RobotCommand | None = None
        self._launch_remaining_s = 0.0
        self._seen: dict[str, tuple[RobotCommand, Ack]] = {}

    def get_state(self) -> RobotSnapshot:
        return RobotSnapshot(self.state, self.position_m, len(self._queue),
                             self._active.command_id if self._active else None)

    def _submit(self, command: RobotCommand) -> Ack:
        if not command.command_id:
            raise ValueError("command_id is required")
        prior = self._seen.get(command.command_id)
        if prior is not None:
            if prior[0] != command:
                raise ValueError("Command ID reused with different contents")
            return Ack(prior[1].command_id, prior[1].accepted, prior[1].state_at_ack,
                       prior[1].reason, duplicate=True)

        if self.state == RobotState.EMERGENCY_STOP and command.kind == CommandKind.EMERGENCY_STOP:
            ack = Ack(command.command_id, True, self.state)
        elif self.state == RobotState.EMERGENCY_STOP:
            ack = Ack(command.command_id, False, self.state, "Emergency stop is active")
        elif command.kind in (CommandKind.STOP, CommandKind.EMERGENCY_STOP):
            self._queue.clear()
            self._active = None
            self._launch_remaining_s = 0.0
            self.state = (RobotState.EMERGENCY_STOP if command.kind == CommandKind.EMERGENCY_STOP
                          else RobotState.IDLE)
            ack = Ack(command.command_id, True, self.state)
        elif len(self._queue) >= self.queue_capacity:
            ack = Ack(command.command_id, False, self.state, "Command queue is full")
        elif command.kind == CommandKind.LAUNCH and not (
                self.state in (RobotState.READY, RobotState.MOVING, RobotState.LAUNCHING)
                or any(pending.kind == CommandKind.MOVE for pending in self._queue)):
            ack = Ack(command.command_id, False, self.state, "Move to a ready position before launch")
        else:
            self._queue.append(command)
            ack = Ack(command.command_id, True, self.state)
        self._seen[command.command_id] = (command, ack)
        return ack

    def move_to(self, x_m: float, y_m: float, command_id: str) -> Ack:
        if not all(math.isfinite(v) for v in (x_m, y_m)):
            raise ValueError("Target coordinates must be finite")
        if not (0 <= x_m <= self.width_m and 0 <= y_m <= self.length_m):
            raise ValueError("Target is outside the simulated court")
        return self._submit(RobotCommand(command_id, CommandKind.MOVE, (x_m, y_m)))

    def launch(self, command_id: str) -> Ack:
        return self._submit(RobotCommand(command_id, CommandKind.LAUNCH))

    def stop(self, command_id: str) -> Ack:
        return self._submit(RobotCommand(command_id, CommandKind.STOP))

    def emergency_stop(self, command_id: str) -> Ack:
        return self._submit(RobotCommand(command_id, CommandKind.EMERGENCY_STOP))

    def reset_emergency_stop(self) -> None:
        """Simulation-only reset; real hardware requires its own safety procedure."""
        if self.state != RobotState.EMERGENCY_STOP:
            raise RuntimeError("Emergency stop is not active")
        self.state = RobotState.IDLE

    def advance(self, seconds: float) -> RobotSnapshot:
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Elapsed time must be nonnegative and finite")
        remaining = seconds
        while self.state != RobotState.EMERGENCY_STOP:
            if self._active is None:
                if not self._queue:
                    break
                self._active = self._queue.popleft()
                if self._active.kind == CommandKind.MOVE:
                    self.state = RobotState.MOVING
                else:
                    self.state = RobotState.LAUNCHING
                    self._launch_remaining_s = self.launch_duration_s

            if self.state == RobotState.MOVING:
                target = self._active.target_m
                distance = math.dist(self.position_m, target)
                duration = distance / self.move_speed_m_s
                if duration > remaining and not math.isclose(
                        duration, remaining, rel_tol=0.0, abs_tol=1e-12):
                    ratio = remaining / duration
                    self.position_m = tuple(a + (b - a) * ratio for a, b in zip(self.position_m, target))
                    break
                self.position_m = target
                remaining = max(0.0, remaining - duration)
                self._active = None
                self.state = RobotState.READY
            elif self.state == RobotState.LAUNCHING:
                if self._launch_remaining_s > remaining and not math.isclose(
                        self._launch_remaining_s, remaining, rel_tol=0.0, abs_tol=1e-12):
                    self._launch_remaining_s -= remaining
                    break
                remaining = max(0.0, remaining - self._launch_remaining_s)
                self._launch_remaining_s = 0.0
                self._active = None
                self.state = RobotState.READY
            if remaining <= 0 and self._queue:
                break
        return self.get_state()
