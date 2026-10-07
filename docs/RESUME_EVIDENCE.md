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
- The selected one-class ONNX model ran through the **existing bounded
  pipeline** on the same 200-frame, 24-FPS x86 recorded-video replay. It
  captured at 24.119 FPS, processed 153 frames at 18.048 FPS and dropped 47 old software-queue
  frames (23.5%); inference p50/p95 was 54.900/74.917 ms and
  read-completion-to-output p50/p95 was 116.121/145.151 ms. Process RSS
  p95 was 267.855 MB; process CPU was 789.65% on a one-core-100 scale. See
  the [raw report](../outputs/benchmarks/v1_selected_cloud_pipeline.json).
  This is not a Pi or physical-camera measurement; do not compare its
  latency directly with the sequential run's different timing boundary.
- The local suite passed 35 tests after the validation error-analysis and ONNX
  comparison tools were added. This is a software test count, not model accuracy.

## Closed-loop software evidence

- The optional simulation-only policy connects pose candidates to accepted MOVE
  and LAUNCH commands, SimulatedRobot state transitions and the existing
  ordered SQLite event stream. The existing WebSocket reads that same log.
  A synthetic-pose integration test crosses the bounded pipeline, event
  processor, policy, simulator and storage. This validates software wiring,
  not real shot recognition or motor safety.
- The camera benchmark entry point now uses the existing bounded pipeline;
  its source adapter and CLI report were tested with a fake camera. No Pi
  timing or real camera reliability number is available. In this workspace,
  26 focused software tests passed; the complete suite and real-model replay
  could not be rerun because the retained visual dependencies failed to load
  and package installation was blocked.

### Resume bullet drafts (scope must stay explicit)

- Built a bounded producer/consumer ONNX video pipeline that kept the latest
  frames under load; on a 200-frame, 24-FPS x86 recorded replay it processed
  153 frames at 18.048 FPS with 145.151 ms p95 read-to-output latency and
  23.5% software queue drops.
- Connected pose-based shot **candidates** to a rule-based, simulation-only
  robot controller and ordered SQLite events; command ACKs and state changes
  are exposed through the local event API and WebSocket replay.
- Fine-tuned and evaluated a one-class YOLOv8n shuttlecock detector on
  source-grouped public data, exported ONNX and checked 100-image
  postprocessed parity before benchmarking in the existing pipeline.
  Mention provisional label and footage-rights limitations when discussing
  the ML experiment; do not pitch its accuracy as deployment-ready.

## Provisional ML experiments, not resume accuracy claims

- A source-grouped public shuttlecock dataset and COCO-pretrained baseline were
  measured; a two-epoch YOLOv8n V1 and one-factor V2 scale ablation were
  genuinely trained and validated. Their hashes and results are in the
  [experiment record](shuttle_experiments.md). Original V1 validation
  mAP50-95 was 0.00658; V2 was 0.00454. Recovered V1 weights separately
  measured 0.005409 on validation and 0.01150 on the one held-out video.
  These low, scene-dependent values should not appear as an accuracy success
  claim. The reproduced V1 score differs from the original V1 score.
- The recovered V1 PyTorch-to-ONNX raw output comparison passed on 20 real
  validation images: worst absolute difference 0.003204 against a 0.05
  tolerance. An expanded [100-image report](../outputs/experiments/v1_onnx_parity_postprocess.json)
  measured 47 detections for each version, no count/class mismatches and
  maximum paired coordinate difference 0.000061 pixels after the same
  detector postprocessing. This sampled parity is not model accuracy.
- The underlying broadcast footage rights remain unverified, and tiny
  fixed-size boxes need broader human review. Do not publish media or weights.

## Not yet supported

No Pi FPS, physical camera latency, shot-event precision/recall, or real robot
reliability figure has been measured. Label quality and source-media rights
prevent marketing the provisional held-out ML scores as product accuracy.
The sequential shot-candidate report and full-clip human annotation template
can now be scored with `scripts/evaluate_shot_candidates.py`, but no reviewed
contact frames have been supplied. Its tests cover matching and incomplete
annotation rejection; they do not establish real-video event performance.

Resume wording should identify **cloud recorded-video replay**, its input and
timing boundary. Add ML numbers only after a licensed, checked dataset, an
untouched source-group test split, actual training/evaluation runs, ONNX parity
checks and linked benchmark reports exist.
