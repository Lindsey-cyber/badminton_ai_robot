"""Compare compatible benchmark runs without mixing timing boundaries."""

import math
from typing import Any


MATCH_FIELDS = ("platform", "python", "runtime_versions", "mode", "class_id",
                "model_sha256", "input", "input_sha256", "source_reported_fps",
                "providers", "warmup_frames", "queue_capacity")


def compare_reports(baseline: dict[str, Any], current: dict[str, Any],
                    max_fps_drop: float = 0.15, max_p95_rise: float = 0.20,
                    max_rss_rise: float = 0.20) -> dict[str, Any]:
    if not all(math.isfinite(x) and x >= 0 for x in
               (max_fps_drop, max_p95_rise, max_rss_rise)):
        raise ValueError("Regression tolerances must be finite and nonnegative")
    differences = [field for field in MATCH_FIELDS if baseline.get(field) != current.get(field)]
    if differences:
        raise ValueError(f"Incomparable benchmark metadata: {', '.join(differences)}")
    latency_keys = [key for key in ("read_to_output_ms", "read_start_to_result_ms")
                    if key in baseline and key in current]
    if len(latency_keys) != 1:
        raise ValueError("Benchmarks have different latency boundaries")
    latency_key = latency_keys[0]
    if ("captured_frames" in baseline) != ("captured_frames" in current):
        raise ValueError("Cannot compare sequential and queued benchmarks")
    count_key = "captured_frames" if "captured_frames" in baseline else "processed_frames"
    if baseline[count_key] != current[count_key]:
        raise ValueError(f"Input frame count differs: {count_key}")

    values = {
        "processed_fps": (baseline["processed_fps"], current["processed_fps"], max_fps_drop),
        f"{latency_key}.p95": (baseline[latency_key]["p95"],
                                 current[latency_key]["p95"], max_p95_rise),
        "process_rss_mb.p95": (baseline["process_rss_mb"]["p95"],
                               current["process_rss_mb"]["p95"], max_rss_rise),
    }
    results = {}
    for name, (old, new, tolerance) in values.items():
        if not (math.isfinite(old) and old > 0 and math.isfinite(new) and new >= 0):
            raise ValueError(f"Invalid measured value: {name}")
        change = (new - old) / old
        regressed = change < -tolerance if name == "processed_fps" else change > tolerance
        results[name] = {"baseline": old, "current": new,
                         "change_fraction": round(change, 4), "regressed": regressed}
    return {"passed": not any(item["regressed"] for item in results.values()),
            "latency_boundary": latency_key, "metrics": results}
