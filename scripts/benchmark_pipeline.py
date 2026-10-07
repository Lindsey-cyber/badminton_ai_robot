#!/usr/bin/env python3
"""Replay video at its recorded rate through a bounded inference queue.

This measures software backpressure, not Raspberry Pi camera exposure latency.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from badminton_ai.pipeline import OpenCVCameraSource, VideoFileSource, VisionPipeline
from benchmark import file_sha256, make_detector, report_path, runtime_versions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--input", type=Path, help="Recorded video")
    source_group.add_argument("--camera", type=int, help="OpenCV camera index")
    parser.add_argument("--width", type=int, help="Requested camera width")
    parser.add_argument("--height", type=int, help="Requested camera height")
    parser.add_argument("--camera-fps", type=float, help="Requested camera FPS")
    parser.add_argument("--mode", choices=("pose", "ball"), default="pose")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--class-id", type=int, default=32,
                        help="Detection class: COCO sports ball 32; fine-tuned shuttlecock 0")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--queue-size", type=int, default=2)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/benchmarks/pipeline_replay.json")
    args = parser.parse_args()
    if args.frames < 1 or args.warmup < 0 or args.queue_size < 1 or args.class_id < 0:
        parser.error("--frames and --queue-size must be positive; --warmup must be nonnegative")
    if args.input is not None and not args.input.is_file():
        parser.error(f"Video not found: {args.input}")
    if args.camera is not None and args.camera < 0:
        parser.error("--camera must be nonnegative")
    if args.input is not None and any(x is not None for x in
                                      (args.width, args.height, args.camera_fps)):
        parser.error("Camera settings require --camera")
    model = args.model or ROOT / ("src/pose/yolov8n-pose.onnx" if args.mode == "pose" else "src/perception/yolov8n.onnx")
    source = None
    try:
        detector = make_detector(args.mode, model.resolve(), args.class_id)
        if args.camera is not None:
            source = OpenCVCameraSource(args.camera, args.frames + args.warmup,
                                        args.width, args.height, args.camera_fps)
        elif args.warmup:
            warmup = VideoFileSource(args.input, max_frames=args.warmup)
            try:
                for _ in range(args.warmup):
                    ok, frame = warmup.read()
                    if not ok:
                        raise RuntimeError("Video ended during warmup")
                    detector.inference(frame)
            finally:
                warmup.close()
        if args.camera is not None and args.warmup:
            for _ in range(args.warmup):
                _, frame = source.read()
                detector.inference(frame)
        if source is None:
            source = VideoFileSource(args.input, max_frames=args.frames)
        report = VisionPipeline(source, detector.inference, queue_size=args.queue_size).run()
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Pipeline benchmark failed: {exc}\n")
    finally:
        if source is not None:
            source.close()

    report.update({
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "runtime_versions": runtime_versions(),
        "mode": args.mode,
        "class_id": args.class_id if args.mode == "ball" else None,
        "input_kind": "camera" if args.camera is not None else "video",
        "input": f"camera:{args.camera}" if args.camera is not None else report_path(args.input),
        "input_sha256": None if args.camera is not None else file_sha256(args.input),
        "source_reported_fps": source.fps,
        "camera_settings": ({
            "requested_width": args.width, "requested_height": args.height,
            "requested_fps": args.camera_fps,
            "reported_width": source.reported_width,
            "reported_height": source.reported_height,
            "reported_fps": source.fps,
        } if args.camera is not None else None),
        "model": report_path(model),
        "model_size_bytes": model.stat().st_size,
        "model_sha256": file_sha256(model.resolve()),
        "providers": detector.session.get_providers(),
        "warmup_frames": args.warmup,
        "queue_capacity": args.queue_size,
        "limitations": [
            "Read-to-output begins after source.read() returns, not at camera exposure.",
            "Drops count frames removed from this software queue, not camera-driver drops.",
            ("Requested camera settings may differ from negotiated OpenCV values."
             if args.camera is not None else
             "Recorded video is paced by metadata FPS, not a physical camera."),
        ],
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}: {report['processed_fps']} processed FPS, "
          f"{report['dropped_frame_rate']:.1%} software drops")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
