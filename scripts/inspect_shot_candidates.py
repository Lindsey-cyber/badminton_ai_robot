#!/usr/bin/env python3
"""Inspect pose wrist-speed peaks in a recording; these are not verified shots."""

import argparse
import json
from pathlib import Path
import sys
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from badminton_ai.events import (CandidateEventProcessor, EventDispatcher, EventType,
                                 SessionEventEmitter, ShotCandidateDetector)
from badminton_ai.storage import EventStore
from benchmark import file_sha256, make_detector, report_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "src/pose/yolov8n-pose.onnx")
    parser.add_argument("--threshold-px-s", type=float, required=True,
                        help="Exploratory wrist-speed threshold in image pixels/second")
    parser.add_argument("--min-gap-s", type=float, default=0.25)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--db", type=Path, help="Optional SQLite event log")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/shot_candidates.json")
    args = parser.parse_args()
    if not args.input.is_file() or args.max_frames < 1:
        parser.error("--input must be an existing video and --max-frames must be positive")

    try:
        import cv2
        pose = make_detector("pose", args.model.resolve())
        capture = cv2.VideoCapture(str(args.input))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Cannot open video: {args.input}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        candidate_detector = ShotCandidateDetector(fps, args.threshold_px_s, args.min_gap_s)
        dispatcher = EventDispatcher()
        events = []
        dispatcher.subscribe(EventType.SHOT_CANDIDATE, lambda event: events.append({
            "event_id": event.event_id,
            "session_id": event.session_id,
            "sequence": event.sequence,
            "timestamp_utc": event.timestamp_utc.isoformat(),
            "type": event.type.value,
            "payload": dict(event.payload),
        }))
        session_id = args.session_id or str(uuid4())
        store = None
        processed = 0
        try:
            if args.db is not None:
                store = EventStore(args.db)
                store.create_session(session_id)
                dispatcher.subscribe(EventType.SHOT_CANDIDATE, store.append)
            emitter = SessionEventEmitter(session_id, dispatcher,
                                          store.next_sequence(session_id) if store else 0)
            processor = CandidateEventProcessor(candidate_detector, emitter)
            for frame_index in range(args.max_frames):
                ok, image = capture.read()
                if not ok:
                    break
                detections, _ = pose.inference(image)
                processor.process(frame_index, detections)
                processed += 1
        finally:
            capture.release()
            if store is not None:
                store.close()
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Candidate inspection failed: {exc}\n")

    report = {
        "session_id": emitter.session_id,
        "input": report_path(args.input),
        "input_sha256": file_sha256(args.input),
        "model_sha256": file_sha256(args.model),
        "processed_frames": processed,
        "source_reported_fps": fps,
        "candidate_count": len(events),
        "threshold_px_s": args.threshold_px_s,
        "events": events,
        "limitations": [
            "A pose wrist-speed peak is not a verified racket-shuttle contact.",
            "The largest bounding box is selected independently per frame; player identity can switch.",
            "No shot precision or recall can be calculated without labeled rallies.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(events)} unverified candidates from {processed} frames to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
