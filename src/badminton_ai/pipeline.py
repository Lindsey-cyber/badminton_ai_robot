"""One capture worker and a bounded, latest-frame inference loop.

The producer keeps the newest frame when inference falls behind. Timestamps
start after source.read() returns; camera exposure and driver buffering are
outside this software boundary.
"""

from dataclasses import dataclass
import os
from pathlib import Path
import queue
import resource
import statistics
import sys
import threading
import time
from typing import Any, Callable, Protocol


class FrameSource(Protocol):
    def read(self) -> tuple[bool, Any]: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class CapturedFrame:
    sequence: int
    read_done_ns: int
    image: Any


def _percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * percent / 100
    lower = int(index)
    return ordered[lower] + (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]) * (index - lower)


def _distribution(values: list[float]) -> dict[str, float]:
    return {"p50": round(_percentile(values, 50), 3),
            "p95": round(_percentile(values, 95), 3),
            "mean": round(statistics.fmean(values), 3)}


def _rss_mb() -> float:
    try:
        with open("/proc/self/statm", encoding="ascii") as statm:
            pages = int(statm.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / 1_000_000
    except (OSError, IndexError, ValueError):
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / (1_000_000 if sys.platform == "darwin" else 1000)


class LatestFrameBuffer:
    """Thread-safe bounded queue; only the capture thread inserts frames."""

    def __init__(self, capacity: int = 2) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self._queue: queue.Queue[CapturedFrame] = queue.Queue(maxsize=capacity)
        self.dropped = 0
        self.max_depth = 0

    def put(self, frame: CapturedFrame) -> None:
        while True:
            try:
                self._queue.put_nowait(frame)
                self.max_depth = max(self.max_depth, self._queue.qsize())
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                    self.dropped += 1
                except queue.Empty:
                    # The consumer freed the slot after put_nowait saw Full.
                    pass

    def get(self, timeout: float) -> CapturedFrame:
        return self._queue.get(timeout=timeout)

    def empty(self) -> bool:
        return self._queue.empty()

    def depth(self) -> int:
        return self._queue.qsize()


class VisionPipeline:
    """Run capture in one worker; inference and output on the calling thread."""

    def __init__(self, source: FrameSource, infer: Callable[[Any], Any],
                 on_result: Callable[[CapturedFrame, Any], None] | None = None,
                 queue_size: int = 2,
                 on_metrics: Callable[[dict[str, float | int]], None] | None = None,
                 metrics_interval_s: float = 1.0) -> None:
        if not 0 < metrics_interval_s < float("inf"):
            raise ValueError("metrics_interval_s must be positive and finite")
        self.source = source
        self.infer = infer
        self.on_result = on_result or (lambda _frame, _result: None)
        self.on_metrics = on_metrics
        self.metrics_interval_s = metrics_interval_s
        self.buffer = LatestFrameBuffer(queue_size)
        self._stop = threading.Event()
        self._producer_done = threading.Event()
        self._capture_error: Exception | None = None
        self._captured = 0
        self._capture_start_ns = 0
        self._capture_end_ns = 0

    def stop(self) -> None:
        self._stop.set()
        self.source.close()

    def _capture(self) -> None:
        self._capture_start_ns = time.perf_counter_ns()
        try:
            while not self._stop.is_set():
                ok, image = self.source.read()
                if not ok:
                    break
                stamp = time.perf_counter_ns()
                self.buffer.put(CapturedFrame(self._captured, stamp, image))
                self._captured += 1
        except Exception as exc:
            self._capture_error = exc
        finally:
            self._capture_end_ns = time.perf_counter_ns()
            self._producer_done.set()

    def run(self) -> dict[str, Any]:
        """Return measured software metrics or raise on source/inference failure."""
        inference_ms: list[float] = []
        read_to_output_ms: list[float] = []
        rss_mb: list[float] = []
        started = time.perf_counter()
        started_cpu = time.process_time()
        window_started = started
        window_cpu = started_cpu
        window_inference: list[float] = []
        window_latency: list[float] = []
        last_dropped = 0

        def publish_window(now: float) -> None:
            nonlocal window_started, window_cpu, last_dropped
            if self.on_metrics is None or not window_inference:
                return
            duration = max(now - window_started, 1e-9)
            current_cpu = time.process_time()
            inference_summary = _distribution(window_inference)
            latency_summary = _distribution(window_latency)
            snapshot = {
                "window_s": round(duration, 3),
                "processed_fps": round(len(window_inference) / duration, 3),
                "inference_p50_ms": inference_summary["p50"],
                "inference_p95_ms": inference_summary["p95"],
                "read_to_output_p50_ms": latency_summary["p50"],
                "read_to_output_p95_ms": latency_summary["p95"],
                "queue_depth": self.buffer.depth(),
                "dropped_frames": self.buffer.dropped - last_dropped,
                "dropped_frames_total": self.buffer.dropped,
                "process_cpu_percent_one_core_100": round((current_cpu - window_cpu) / duration * 100, 2),
                "process_rss_mb": rss_mb[-1],
            }
            self.on_metrics(snapshot)
            window_started, window_cpu, last_dropped = now, current_cpu, self.buffer.dropped
            window_inference.clear()
            window_latency.clear()

        worker = threading.Thread(target=self._capture, name="frame-capture", daemon=True)
        worker.start()
        try:
            while not self._producer_done.is_set() or not self.buffer.empty():
                if self._capture_error is not None:
                    raise RuntimeError("Capture worker failed") from self._capture_error
                try:
                    frame = self.buffer.get(timeout=0.05)
                except queue.Empty:
                    continue
                infer_start = time.perf_counter_ns()
                result = self.infer(frame.image)
                infer_done = time.perf_counter_ns()
                self.on_result(frame, result)
                output_done = time.perf_counter_ns()
                inference_sample = (infer_done - infer_start) / 1_000_000
                latency_sample = (output_done - frame.read_done_ns) / 1_000_000
                inference_ms.append(inference_sample)
                read_to_output_ms.append(latency_sample)
                window_inference.append(inference_sample)
                window_latency.append(latency_sample)
                rss_mb.append(_rss_mb())
                if time.perf_counter() - window_started >= self.metrics_interval_s:
                    publish_window(time.perf_counter())
            if self._capture_error is not None:
                raise RuntimeError("Capture worker failed") from self._capture_error
            publish_window(time.perf_counter())
        finally:
            self.stop()
            worker.join(timeout=2.0)
            if worker.is_alive():
                raise RuntimeError("Capture worker did not stop within 2 seconds")

        wall_s = time.perf_counter() - started
        if not inference_ms:
            raise RuntimeError("No frames were processed")
        capture_s = (self._capture_end_ns - self._capture_start_ns) / 1_000_000_000
        return {
            "captured_frames": self._captured,
            "processed_frames": len(inference_ms),
            "dropped_frames": self.buffer.dropped,
            "dropped_frame_rate": round(self.buffer.dropped / self._captured, 4),
            "max_queue_depth": self.buffer.max_depth,
            "captured_fps": round(self._captured / capture_s, 3),
            "processed_fps": round(len(inference_ms) / wall_s, 3),
            "inference_ms": _distribution(inference_ms),
            "read_to_output_ms": _distribution(read_to_output_ms),
            "process_cpu_percent_one_core_100": round((time.process_time() - started_cpu) / wall_s * 100, 2),
            "process_rss_mb": _distribution(rss_mb),
            "wall_s": round(wall_s, 3),
            "camera_exposure_to_output_ms": None,
        }


class VideoFileSource:
    """Replay a recorded video at its metadata FPS for repeatable load tests."""

    def __init__(self, path: Path, max_frames: int | None = None) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required for video replay") from exc
        if max_frames is not None and max_frames < 1:
            raise ValueError("max_frames must be positive")
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            self._cap.release()
            raise RuntimeError(f"Cannot open video: {path}")
        self.fps = self._cap.get(cv2.CAP_PROP_FPS)
        if not self.fps or self.fps <= 0:
            self._cap.release()
            raise RuntimeError("Video has no valid FPS metadata")
        self.max_frames = max_frames
        self._frames_read = 0
        self._started: float | None = None
        self._closed = threading.Event()

    def read(self) -> tuple[bool, Any]:
        if self._closed.is_set():
            return False, None
        if self.max_frames is not None and self._frames_read >= self.max_frames:
            return False, None
        if self._started is None:
            self._started = time.perf_counter()
        target = self._started + self._frames_read / self.fps
        delay = target - time.perf_counter()
        if delay > 0 and self._closed.wait(delay):
            return False, None
        if self._closed.is_set():
            return False, None
        ok, frame = self._cap.read()
        if ok:
            self._frames_read += 1
        return ok, frame

    def close(self) -> None:
        self._closed.set()
        self._cap.release()


class OpenCVCameraSource:
    """Bounded OpenCV camera capture; a failed read aborts the benchmark."""

    def __init__(self, index: int, max_frames: int, width: int | None = None,
                 height: int | None = None, requested_fps: float | None = None) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required for camera capture") from exc
        if index < 0 or max_frames < 1:
            raise ValueError("Camera index must be nonnegative and max_frames positive")
        if ((width is not None and width < 1) or
                (height is not None and height < 1)):
            raise ValueError("Camera dimensions must be positive")
        if requested_fps is not None and not 0 < requested_fps < float("inf"):
            raise ValueError("Requested camera FPS must be positive and finite")
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            self._cap.release()
            raise RuntimeError(f"Cannot open camera: {index}")
        if width is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if requested_fps is not None:
            self._cap.set(cv2.CAP_PROP_FPS, requested_fps)
        self.reported_width = self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.reported_height = self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or None
        self.max_frames = max_frames
        self._frames_read = 0
        self._closed = False

    def read(self) -> tuple[bool, Any]:
        if self._closed or self._frames_read >= self.max_frames:
            return False, None
        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError("Camera read failed during warmup or the measured run")
        self._frames_read += 1
        return True, frame

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._cap.release()
