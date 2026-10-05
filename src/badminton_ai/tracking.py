"""Single-camera point tracking and apparent two-dimensional speed estimates.

Positions are pixels. A court-width scale does not calibrate the 3D flight of
an airborne shuttlecock, so the converted speed is only an approximation.
"""

from collections import deque
import logging
import math

import numpy as np


logger = logging.getLogger(__name__)


class BallKalmanFilter:
    """Constant-velocity 2D Kalman filter with state [x, y, vx, vy]."""

    def __init__(self) -> None:
        self.initialized = False
        self.state = np.zeros(4, dtype=float)
        self.P = np.eye(4) * 100
        self.Q = np.eye(4)
        self.R = np.eye(2) * 10
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)

    def update(self, measurement: tuple[float, float]) -> np.ndarray:
        z = np.asarray(measurement, dtype=float)
        if z.shape != (2,) or not np.all(np.isfinite(z)):
            raise ValueError("Measurement must be a finite (x, y) pixel position")
        if not self.initialized:
            self.state[:2] = z
            self.initialized = True
            return self.state[:2].copy()

        predicted = self.F @ self.state
        predicted_cov = self.F @ self.P @ self.F.T + self.Q
        residual = z - self.H @ predicted
        residual_cov = self.H @ predicted_cov @ self.H.T + self.R
        gain = np.linalg.solve(residual_cov.T, (predicted_cov @ self.H.T).T).T
        self.state = predicted + gain @ residual
        self.P = (np.eye(4) - gain @ self.H) @ predicted_cov
        return self.state[:2].copy()

    def advance_without_measurement(self) -> np.ndarray:
        """Advance position and uncertainty after a frame without a detection."""
        if not self.initialized:
            raise RuntimeError("Filter has no initial measurement")
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[:2].copy()

    def predict_next(self, n_steps: int = 5) -> list[np.ndarray]:
        if n_steps < 0:
            raise ValueError("n_steps must be nonnegative")
        if not self.initialized:
            return []
        state = self.state.copy()
        positions = []
        for _ in range(n_steps):
            state = self.F @ state
            positions.append(state[:2].copy())
        return positions


class SpeedEstimator:
    """Smoothed image-plane speed; conversion assumes a constant pixel scale."""

    def __init__(self, fps: float, pixels_per_meter: float | None = None) -> None:
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be positive and finite")
        if pixels_per_meter is not None and (not math.isfinite(pixels_per_meter) or pixels_per_meter <= 0):
            raise ValueError("pixels_per_meter must be positive and finite")
        self.fps = fps
        self.pixels_per_meter = pixels_per_meter
        self.prev_pos: tuple[float, float] | None = None
        self.prev_frame_index: int | None = None
        self.speeds: deque[float] = deque(maxlen=10)

    def calibrate_from_court(self, court_width_px: float, court_width_m: float = 6.1) -> None:
        if not all(math.isfinite(x) and x > 0 for x in (court_width_px, court_width_m)):
            raise ValueError("Court width must be positive and finite")
        self.pixels_per_meter = court_width_px / court_width_m
        logger.info("Apparent court-plane scale: %.1f pixels/meter", self.pixels_per_meter)

    def update(self, position: tuple[float, float], frame_index: int | None = None) -> dict[str, float | None]:
        if len(position) != 2 or not all(math.isfinite(float(v)) for v in position):
            raise ValueError("Position must be a finite (x, y) pixel position")
        current = (float(position[0]), float(position[1]))
        if frame_index is not None and frame_index < 0:
            raise ValueError("frame_index must be nonnegative")
        if self.prev_pos is None:
            self.prev_pos = current
            self.prev_frame_index = frame_index
            return {"pixel_speed": 0.0, "speed_kmh": None, "speed_ms": None}

        frame_delta = 1
        if frame_index is not None and self.prev_frame_index is not None:
            frame_delta = frame_index - self.prev_frame_index
            if frame_delta <= 0:
                raise ValueError("frame_index must increase between positions")
        pixel_speed = math.dist(current, self.prev_pos) * self.fps / frame_delta
        self.prev_pos = current
        self.prev_frame_index = frame_index
        self.speeds.append(pixel_speed)
        averaged = float(np.mean(self.speeds))
        if self.pixels_per_meter is None:
            return {"pixel_speed": averaged, "speed_ms": None, "speed_kmh": None}
        meters_per_second = averaged / self.pixels_per_meter
        return {"pixel_speed": averaged, "speed_ms": meters_per_second,
                "speed_kmh": meters_per_second * 3.6}
