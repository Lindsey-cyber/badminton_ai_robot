#!/usr/bin/env python3
"""Replay recorded video through bounded pose inference into SQLite events."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from badminton_ai.events import (CandidateEventProcessor, EventDispatcher, EventType,
                                 SessionEventEmitter, ShotCandidateDetector)
from badminton_ai.pipeline import VideoFileSource, VisionPipeline
from badminton_ai.storage import EventStore
from benchmark import file_sha256, make_detector, report_path, runtime_versions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "src/pose/yolov8n-pose.onnx")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--queue-size", type=int, default=2)
    parser.add_argument("--threshold-px-s", type=float, required=True)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--db", type=Path, default=ROOT / "outputs/events.sqlite3")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/training_replay.json")
    args = parser.parse_args()
    if not args.input.is_file() or args.frames < 1 or args.queue_size < 1:
        parser.error("--input must be an existing video; frames and queue-size must be positive")

    session_id = args.session_id or str(uuid4())
    try:
        pose = make_detector("pose", args.model.resolve())
        source = VideoFileSource(args.input, max_frames=args.frames)
        try:
            candidate_detector = ShotCandidateDetector(source.fps, args.threshold_px_s)
            with EventStore(args.db) as store:
                if store.session_exists(session_id):
                    raise ValueError("Session ID already exists; use a new ID for each replay")
                store.create_session(session_id)
                dispatcher = EventDispatcher()
                dispatcher.subscribe(EventType.SHOT_CANDIDATE, store.append)
                emitter = SessionEventEmitter(session_id, dispatcher)
                processor = CandidateEventProcessor(candidate_detector, emitter)
                pipeline = VisionPipeline(
                    source, lambda image: pose.inference(image)[0],
                    on_result=lambda frame, detections: processor.process(frame.sequence, detections),
                    queue_size=args.queue_size,
                )
                metrics = pipeline.run()
                candidate_count = store.next_sequence(session_id)
        finally:
            source.close()
    except (OSError, FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Training replay failed: {exc}\n")

    report = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "runtime_versions": runtime_versions(),
        "input": report_path(args.input),
        "input_sha256": file_sha256(args.input),
        "model": report_path(args.model),
        "model_sha256": file_sha256(args.model),
        "model_size_bytes": args.model.stat().st_size,
        "db": str(args.db),
        "source_reported_fps": source.fps,
        "threshold_px_s": args.threshold_px_s,
        "candidate_count": candidate_count,
        "metrics": metrics,
        "limitations": [
            "Candidates are wrist-speed peaks, not confirmed racket-shuttle contacts.",
            "One largest-person selection per frame can switch player identity.",
            "Software drops exclude camera-driver drops; replay is not a live camera.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {candidate_count} unverified candidates to {args.db}; "
          f"{metrics['processed_frames']}/{metrics['captured_frames']} frames processed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
