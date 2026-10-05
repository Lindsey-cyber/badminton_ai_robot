"""In-process events and conservative wrist-speed shot candidates.

A wrist-speed peak is not evidence of racket contact. Ground-truth rallies
are required before candidate counts can be called detected shots.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import math
from typing import Callable, Mapping
from uuid import uuid4


class EventType(str, Enum):
    SHOT_CANDIDATE = "ShotCandidate"


@dataclass(frozen=True)
class Event:
    event_id: str
    session_id: str
    sequence: int
    timestamp_utc: datetime
    type: EventType
    payload: Mapping[str, str | int | float]


class EventDispatcher:
    """Synchronous local dispatch; handler failure propagates to the caller."""

    def __init__(self) -> None:
        self._handlers: dict[EventType, list[Callable[[Event], None]]] = {}

    def subscribe(self, event_type: EventType, handler: Callable[[Event], None]) -> None:
        self._handlers.setdefault(event_type, []).append(handler)

    def publish(self, event: Event) -> None:
        for handler in self._handlers.get(event.type, ()):
            handler(event)


class SessionEventEmitter:
    """Assign monotonic sequence numbers within one session."""

    def __init__(self, session_id: str, dispatcher: EventDispatcher,
                 starting_sequence: int = 0) -> None:
        if not session_id:
            raise ValueError("session_id is required")
        if starting_sequence < 0:
            raise ValueError("starting_sequence must be nonnegative")
        self.session_id = session_id
        self.dispatcher = dispatcher
        self._next_sequence = starting_sequence

    def emit(self, event_type: EventType, payload: Mapping[str, str | int | float]) -> Event:
        event = Event(str(uuid4()), self.session_id, self._next_sequence,
                      datetime.now(timezone.utc), event_type, dict(payload))
        self._next_sequence += 1
        self.dispatcher.publish(event)
        return event


@dataclass(frozen=True)
class Wrist:
    x: float
    y: float
    confidence: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in (self.x, self.y, self.confidence)):
            raise ValueError("Wrist coordinates and confidence must be finite")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Wrist confidence must be between zero and one")


@dataclass(frozen=True)
class ShotCandidate:
    frame_index: int
    video_time_s: float
    hand: str
    wrist_speed_px_s: float


class ShotCandidateDetector:
    """Emit a local wrist-speed peak after seeing the following frame.

    Thresholds are in image pixels/second. Camera scale, player selection and
    pose accuracy affect these values; no default accuracy is implied.
    """

    def __init__(self, fps: float, min_wrist_speed_px_s: float,
                 min_gap_s: float = 0.25, min_confidence: float = 0.25) -> None:
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be positive and finite")
        if not math.isfinite(min_wrist_speed_px_s) or min_wrist_speed_px_s <= 0:
            raise ValueError("min_wrist_speed_px_s must be positive and finite")
        if not math.isfinite(min_gap_s) or min_gap_s < 0:
            raise ValueError("min_gap_s must be nonnegative and finite")
        if not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be between zero and one")
        self.fps = fps
        self.min_speed = min_wrist_speed_px_s
        self.min_gap_frames = math.ceil(min_gap_s * fps)
        self.min_confidence = min_confidence
        self._last_frame = -1
        self._last_wrists: tuple[Wrist, Wrist] | None = None
        self._previous_speed: tuple[int, float, str] | None = None
        self._before_previous_speed: float | None = None
        self._last_candidate_frame = -self.min_gap_frames - 1

    def observe(self, frame_index: int, left: Wrist | None,
                right: Wrist | None) -> ShotCandidate | None:
        if frame_index <= self._last_frame:
            raise ValueError("frame_index must strictly increase")
        delta_frames = frame_index - self._last_frame
        self._last_frame = frame_index
        current = (left, right) if left is not None and right is not None else None
        if current is None or self._last_wrists is None:
            self._last_wrists = current
            self._previous_speed = None
            self._before_previous_speed = None
            return None

        speeds = []
        for hand, old, new in zip(("left", "right"), self._last_wrists, current):
            if old.confidence >= self.min_confidence and new.confidence >= self.min_confidence:
                speeds.append((math.dist((old.x, old.y), (new.x, new.y)) * self.fps / delta_frames, hand))
        self._last_wrists = current
        if not speeds:
            self._previous_speed = None
            self._before_previous_speed = None
            return None

        speed, hand = max(speeds)
        candidate = None
        previous = self._previous_speed
        if previous is not None and self._before_previous_speed is not None:
            peak_frame, peak_speed, peak_hand = previous
            if (peak_speed >= self.min_speed and peak_speed >= self._before_previous_speed
                    and peak_speed > speed
                    and peak_frame - self._last_candidate_frame >= self.min_gap_frames):
                candidate = ShotCandidate(peak_frame, peak_frame / self.fps, peak_hand, peak_speed)
                self._last_candidate_frame = peak_frame
        self._before_previous_speed = previous[1] if previous is not None else None
        self._previous_speed = (frame_index, speed, hand)
        return candidate


class CandidateEventProcessor:
    """Turn pose detections into unverified, ordered candidate events."""

    def __init__(self, detector: ShotCandidateDetector,
                 emitter: SessionEventEmitter) -> None:
        self.detector = detector
        self.emitter = emitter

    def process(self, frame_index: int, detections: list[dict]) -> Event | None:
        if detections:
            # One visible player is assumed; the selected identity may switch.
            person = max(detections, key=lambda d: (d["bbox"][2] - d["bbox"][0]) *
                         (d["bbox"][3] - d["bbox"][1]))
            keypoints = person["keypoints"]
            wrists = (Wrist(*keypoints[9]), Wrist(*keypoints[10]))
        else:
            wrists = (None, None)
        candidate = self.detector.observe(frame_index, *wrists)
        if candidate is None:
            return None
        return self.emitter.emit(EventType.SHOT_CANDIDATE, {
            "frame_index": candidate.frame_index,
            "video_time_s": candidate.video_time_s,
            "hand": candidate.hand,
            "wrist_speed_px_s": round(candidate.wrist_speed_px_s, 3),
            "threshold_px_s": self.detector.min_speed,
        })
