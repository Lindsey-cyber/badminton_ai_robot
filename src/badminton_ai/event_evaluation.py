"""One-to-one scoring for human-reviewed racket contact frames.

The scorer cannot verify that a human actually reviewed every frame. It only
accepts an explicit complete-review record tied to the prediction video's hash.
"""

import math
import statistics
from typing import Any


def _frame_list(value: Any, name: str) -> list[int]:
    if not isinstance(value, list) or any(type(frame) is not int or frame < 0 for frame in value):
        raise ValueError(f"{name} must be a list of nonnegative integer frame indices")
    if any(a >= b for a, b in zip(value, value[1:])):
        raise ValueError(f"{name} must be strictly increasing without duplicates")
    return value


def evaluate_candidates(predictions: dict[str, Any], annotations: dict[str, Any],
                        tolerance_s: float) -> dict[str, Any]:
    """Maximize one-to-one matches, then minimize their total timing error."""
    if not math.isfinite(tolerance_s) or tolerance_s < 0:
        raise ValueError("tolerance_s must be nonnegative and finite")
    sha = predictions.get("input_sha256")
    if not isinstance(sha, str) or len(sha) != 64 or annotations.get("input_sha256") != sha:
        raise ValueError("Prediction and annotation video SHA-256 must match")
    fps = predictions.get("source_reported_fps")
    if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Prediction FPS must be positive and finite")
    if annotations.get("source_reported_fps") != fps:
        raise ValueError("Annotation FPS differs from the prediction recording")
    count = predictions.get("processed_frames")
    if type(count) is not int or count <= 0:
        raise ValueError("Predictions must report a positive processed_frames count")
    if (annotations.get("annotation_schema_version") != 1
            or annotations.get("review_status") != "complete"
            or not isinstance(annotations.get("reviewer"), str)
            or not annotations["reviewer"].strip()
            or annotations.get("reviewed_frames") != [0, count - 1]):
        raise ValueError("A named reviewer must mark the entire processed clip complete")
    truth = _frame_list(annotations.get("shot_frame_indices"), "shot_frame_indices")
    if truth and truth[-1] >= count:
        raise ValueError("Ground-truth shot is outside the reviewed clip")
    events = predictions.get("events")
    if not isinstance(events, list) or any(not isinstance(e, dict) for e in events):
        raise ValueError("Predictions must contain an events list")
    if any(e.get("type") != "ShotCandidate" or not isinstance(e.get("payload"), dict)
           for e in events):
        raise ValueError("Only ShotCandidate events can be scored")
    detected = _frame_list([e["payload"].get("frame_index") for e in events],
                           "candidate frame indices")
    if detected and detected[-1] >= count:
        raise ValueError("Candidate is outside the reviewed clip")

    # Ordered matching avoids reusing one prediction for two nearby contacts.
    # Each cell holds (match count, negative total frame error, decision).
    n, m = len(detected), len(truth)
    dp = [[(0, 0, "end") for _ in range(m + 1)] for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            options = [(*dp[i + 1][j][:2], "skip_prediction"),
                       (*dp[i][j + 1][:2], "skip_truth")]
            error = abs(detected[i] - truth[j])
            if error / fps <= tolerance_s + 1e-12:
                count_after, error_after, _ = dp[i + 1][j + 1]
                options.append((count_after + 1, error_after - error, "match"))
            dp[i][j] = max(options, key=lambda option: option[:2])
    matches = []
    i = j = 0
    while i < n and j < m:
        decision = dp[i][j][2]
        if decision == "match":
            matches.append({"candidate_frame": detected[i], "shot_frame": truth[j],
                            "signed_error_ms": round((detected[i] - truth[j]) / fps * 1000, 3)})
            i += 1
            j += 1
        elif decision == "skip_prediction":
            i += 1
        else:
            j += 1
    tp = len(matches)
    precision = tp / n if n else None
    recall = tp / m if m else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else (0.0 if n and m else None)
    errors = sorted(abs(match["signed_error_ms"]) for match in matches)
    return {
        "input_sha256": sha,
        "source_reported_fps": fps,
        "reviewer": annotations["reviewer"],
        "reviewed_frames": annotations["reviewed_frames"],
        "tolerance_s": tolerance_s,
        "true_positives": tp,
        "false_positives": n - tp,
        "false_negatives": m - tp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "absolute_timing_error_p50_ms": statistics.median(errors) if errors else None,
        "absolute_timing_error_p95_ms": errors[math.ceil(0.95 * len(errors)) - 1] if errors else None,
        "matches": matches,
    }
