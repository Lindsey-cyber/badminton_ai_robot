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
  processed 153 frames at 18.048 FPS and dropped 47 old software-queue
  frames (23.5%); inference p50/p95 was 54.900/74.917 ms and
  read-completion-to-output p50/p95 was 116.121/145.151 ms. Process RSS
  p95 was 267.855 MB. See the [raw report](../outputs/benchmarks/v1_selected_cloud_pipeline.json).
  This is not a Pi or physical-camera measurement; do not compare its
  latency directly with the sequential run's different timing boundary.
- The local suite passed 35 tests after the validation error-analysis and ONNX
  comparison tools were added. This is a software test count, not model accuracy.

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
  tolerance. This is pre-NMS numerical parity, not accuracy or exact
  postprocessed-detection parity; see the [raw report](../outputs/experiments/v1_onnx_parity.json).
- The underlying broadcast footage rights remain unverified, and tiny
  fixed-size boxes need broader human review. Do not publish media or weights.

## Not yet supported

No Pi FPS, physical camera latency, shot-event precision/recall, or real robot
reliability figure has been measured. Label quality and source-media rights
prevent marketing the provisional held-out ML scores as product accuracy.

Resume wording should identify **cloud recorded-video replay**, its input and
timing boundary. Add ML numbers only after a licensed, checked dataset, an
untouched source-group test split, actual training/evaluation runs, ONNX parity
checks and linked benchmark reports exist.
