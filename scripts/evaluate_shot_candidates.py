#!/usr/bin/env python3
"""Score candidate contact frames against a fully human-reviewed video clip."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from badminton_ai.event_evaluation import evaluate_candidates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True,
                        help="JSON from inspect_shot_candidates.py (sequential replay)")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--annotations", type=Path,
                        help="Manually completed contact-frame JSON")
    action.add_argument("--template", action="store_true",
                        help="Create a prefilled annotation template for full-video review")
    parser.add_argument("--tolerance-s", type=float, default=0.15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
        if args.template:
            count = predictions["processed_frames"]
            sha = predictions["input_sha256"]
            fps = predictions["source_reported_fps"]
            if type(count) is not int or count <= 0 or not isinstance(sha, str) or len(sha) != 64:
                raise ValueError("Predictions lack a valid reviewed frame count or video hash")
            result = {
                "annotation_schema_version": 1,
                "input_sha256": sha,
                "source_reported_fps": fps,
                "review_status": "pending",
                "reviewer": "",
                "reviewed_frames": [0, count - 1],
                "shot_frame_indices": [],
                "candidate_frames_for_review": [event["payload"]["frame_index"]
                                                for event in predictions["events"]],
            }
        else:
            annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
            result = evaluate_candidates(predictions, annotations, args.tolerance_s)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(1, f"Event evaluation failed: {exc}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.template:
        print(f"Saved incomplete review template to {args.output}; inspect every video frame")
    else:
        print(f"Saved {args.output}: {result['true_positives']} matches, "
              f"{result['false_positives']} extra candidates, {result['false_negatives']} missed contacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
