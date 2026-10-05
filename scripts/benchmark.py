#!/usr/bin/env python3
"""Measure the existing sequential ONNX demo on a video or camera.

This is a baseline for the current code, not the future real-time pipeline.
No frames are intentionally dropped, and camera capture timestamps are unavailable.
"""

import argparse
import hashlib
from importlib import metadata
import json
import os
import platform
import resource
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def report_path(path: Path) -> str:
    """Use a repository-relative path when the input belongs to this checkout."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_versions() -> dict[str, str | None]:
    versions = {}
    for package in ("numpy", "opencv-python-headless", "onnxruntime"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def percentile(values: list[float], percent: float) -> float:
    """Linear interpolation, matching NumPy's default percentile convention."""
    if not values:
        raise ValueError("Cannot calculate a percentile from zero samples")
    ordered = sorted(values)
    index = (len(ordered) - 1) * percent / 100
    low = int(index)
    return ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (index - low)


def current_rss_mb() -> float:
    """Current process RSS on Linux; peak RSS fallback on other Unix systems."""
    try:
        with open("/proc/self/statm", encoding="ascii") as statm:
            resident_pages = int(statm.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / 1_000_000
    except (OSError, IndexError, ValueError):
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / (1_000_000 if sys.platform == "darwin" else 1000)


def summarize(samples: list[float]) -> dict[str, float]:
    if not samples:
        raise ValueError("No frames were measured")
    return {
        "p50": round(percentile(samples, 50), 3),
        "p95": round(percentile(samples, 95), 3),
        "mean": round(statistics.fmean(samples), 3),
    }


def make_detector(mode: str, model_path: Path):
    # Reuse the demo's preprocessing and postprocessing so this measures the
    # code that actually exists. Do not download a model during a benchmark.
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    sys.path.insert(0, str(ROOT / "scripts"))
    if mode == "pose":
        from demo_pose_inference import YOLOv8PoseInference
        return YOLOv8PoseInference(str(model_path))
    from demo_badminton_detection import YOLOv8Detector
    return YOLOv8Detector(str(model_path), target_classes=[32])


def run(args: argparse.Namespace) -> dict:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required: pip install -r requirements.txt") from exc

    model_path = Path(args.model).resolve()
    detector = make_detector(args.mode, model_path)
    capture = cv2.VideoCapture(args.camera if args.camera is not None else str(args.input))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open input: {args.camera if args.camera is not None else args.input}")

    core_ms: list[float] = []
    inference_total_ms: list[float] = []
    iteration_ms: list[float] = []
    rss_mb: list[float] = []
    read_failures = 0
    source_fps = capture.get(cv2.CAP_PROP_FPS) or None
    start_wall = None
    start_cpu = None
    try:
        # Warmup is excluded from all samples and elapsed throughput time.
        for _ in range(args.warmup):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Input ended or camera read failed during warmup")
            detector.inference(frame)

        start_wall = time.perf_counter()
        start_cpu = time.process_time()
        for _ in range(args.frames):
            read_start = time.perf_counter()
            ok, frame = capture.read()
            if not ok:
                if args.camera is None:
                    break  # End of a recorded video is normal.
                read_failures += 1
                raise RuntimeError("Camera read failed; no reconnect exists in the baseline demo")
            read_done = time.perf_counter()
            _, ort_ms = detector.inference(frame)
            done = time.perf_counter()
            core_ms.append(ort_ms)
            inference_total_ms.append((done - read_done) * 1000)
            iteration_ms.append((done - read_start) * 1000)
            rss_mb.append(current_rss_mb())
        wall_s = time.perf_counter() - start_wall
        cpu_s = time.process_time() - start_cpu
    finally:
        capture.release()

    if not core_ms:
        raise RuntimeError("No frames measured; check input and --warmup")
    return {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "runtime_versions": runtime_versions(),
        "mode": args.mode,
        "model": report_path(model_path),
        "model_size_bytes": model_path.stat().st_size,
        "model_sha256": file_sha256(model_path),
        "providers": detector.session.get_providers(),
        "input": report_path(args.input) if args.camera is None else f"camera:{args.camera}",
        "input_kind": "video" if args.camera is None else "camera",
        "source_reported_fps": source_fps,
        "warmup_frames": args.warmup,
        "processed_frames": len(core_ms),
        "read_failures": read_failures,
        "measured_wall_s": round(wall_s, 3),
        "processed_fps": round(len(core_ms) / wall_s, 3),
        "ort_run_ms": summarize(core_ms),
        "inference_with_pre_post_ms": summarize(inference_total_ms),
        "read_start_to_result_ms": summarize(iteration_ms),
        "process_cpu_percent_one_core_100": round(cpu_s / wall_s * 100, 2),
        "process_rss_mb": summarize(rss_mb),
        "dropped_frames": None,
        "dropped_frame_rate": None,
        "camera_capture_to_result_ms": None,
        "limitations": [
            "Sequential capture/inference; no producer-consumer queue exists yet.",
            "OpenCV provides no camera exposure timestamp or driver drop counter here.",
            "Recorded video throughput is not real-time camera input FPS.",
            "CPU percent is process CPU time divided by wall time; 100% means one core.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Recorded video path")
    source.add_argument("--camera", type=int, help="OpenCV camera index")
    parser.add_argument("--mode", choices=("pose", "ball"), default="pose")
    parser.add_argument("--model", type=Path, help="Existing ONNX model (no auto download)")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/benchmarks/baseline.json")
    args = parser.parse_args()
    if args.frames <= 0 or args.warmup < 0:
        parser.error("--frames must be positive and --warmup must be nonnegative")
    if args.model is None:
        args.model = ROOT / ("src/pose/yolov8n-pose.onnx" if args.mode == "pose" else "src/perception/yolov8n.onnx")
    if args.input is not None and not args.input.is_file():
        parser.error(f"Input not found: {args.input}")
    try:
        result = run(args)
    except (FileNotFoundError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"Benchmark failed: {exc}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved {args.output}: {result['processed_fps']} processed FPS, "
          f"p95 read-start-to-result {result['read_start_to_result_ms']['p95']} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
