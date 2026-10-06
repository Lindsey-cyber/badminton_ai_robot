# Resume evidence ledger

Only measurements actually produced by this repository belong here. This file
separates software throughput from model accuracy and target-hardware claims.

## Supported software measurements

- A recorded 200-frame, 24-FPS video replay on an **x86_64 cloud container**
  processed 153 frames, discarded 47 old software-queue frames (23.5%), and
  reported 18.064 processed FPS. Read-completion-to-output p50/p95 latency was
  116.583/146.958 ms; process RSS p95 was 266.420 MB. The command, environment,
  raw report and measurement boundaries are in
  [the benchmark note](cloud_benchmark_2026-10-05.md) and
  [`cloud_pipeline_replay.json`](../outputs/benchmarks/cloud_pipeline_replay.json).
- The corresponding sequential run processed 200 frames at 17.732 FPS with
  read-start-to-result p50/p95 54.768/71.763 ms. Its timing boundary differs;
  these two latencies must not be presented as a speedup comparison.
- The local suite passed 33 tests after the windowed metrics and raw-YOLO box
  suppression changes (2026-10-05). This is a software test count, not model accuracy.

## Not yet supported

No YOLO fine-tuning, shuttlecock precision/recall/mAP, source-to-ONNX parity,
Pi FPS, live camera latency, shot-event precision/recall, or real robot reliability
figures have been measured. The [public data audit](public_data_audit.md)
documents provisional source counts and unresolved media rights; those counts
must not become model-performance or legally cleared dataset claims.

Resume wording should identify **cloud recorded-video replay**, its input and
timing boundary. Add ML numbers only after a licensed, checked dataset, an
untouched source-group test split, actual training/evaluation runs, ONNX parity
checks and linked benchmark reports exist.
