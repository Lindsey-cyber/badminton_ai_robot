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

from badminton_ai.pipeline import VideoFileSource, VisionPipeline
from benchmark import make_detector


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--mode", choices=("pose", "ball"), default="pose")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--queue-size", type=int, default=2)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/benchmarks/pipeline_replay.json")
    args = parser.parse_args()
    if args.frames < 1 or args.warmup < 0 or args.queue_size < 1:
        parser.error("--frames and --queue-size must be positive; --warmup must be nonnegative")
    if not args.input.is_file():
        parser.error(f"Video not found: {args.input}")
    model = args.model or ROOT / ("src/pose/yolov8n-pose.onnx" if args.mode == "pose" else "src/perception/yolov8n.onnx")
    try:
        detector = make_detector(args.mode, model.resolve())
        if args.warmup:
            warmup = VideoFileSource(args.input, max_frames=args.warmup)
            try:
                for _ in range(args.warmup):
                    ok, frame = warmup.read()
                    if not ok:
                        raise RuntimeError("Video ended during warmup")
                    detector.inference(frame)
            finally:
                warmup.close()
        source = VideoFileSource(args.input, max_frames=args.frames)
        report = VisionPipeline(source, detector.inference, queue_size=args.queue_size).run()
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Pipeline benchmark failed: {exc}\n")

    report.update({
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "mode": args.mode,
        "input": str(args.input.resolve()),
        "source_reported_fps": source.fps,
        "model": str(model.resolve()),
        "model_size_bytes": model.stat().st_size,
        "providers": detector.session.get_providers(),
        "warmup_frames": args.warmup,
        "queue_capacity": args.queue_size,
        "limitations": [
            "Recorded video is paced by metadata FPS; it is not a physical camera.",
            "Read-to-output begins after decoder read, not at camera exposure.",
            "Drops count frames removed from this software queue, not camera-driver drops.",
        ],
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}: {report['processed_fps']} processed FPS, "
          f"{report['dropped_frame_rate']:.1%} software drops")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
