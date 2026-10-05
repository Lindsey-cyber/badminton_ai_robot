# Real-Time Edge AI Badminton Training System (in progress)

The repository currently contains **offline vision demos and data collection tools**, not an integrated real-time training system. See [the repository audit](docs/repository_audit.md) for an evidence-based inventory and the staged engineering plan. No shuttlecock fine-tuning or Raspberry Pi result has been verified.

## Run the existing demos

Use Python 3.10+ and install `pip install -r requirements.txt` in a virtual environment. From the repository root:

```bash
python scripts/demo_pose_inference.py --input assets/demo_inputs/badminton_sample.mp4 --max-frames 100
python scripts/demo_badminton_detection.py --input assets/demo_inputs/badminton_sample.mp4
python scripts/stereo_distance_demo.py --mode simulate
```

The ball demo may download a generic YOLOv8n model. It searches the COCO `sports ball` class; its detections are not validated shuttlecock detections. The pose demo uses the checked-in `src/pose/yolov8n-pose.onnx` model. `scripts/generate_training_report.py --demo` uses generated data and its numbers must not be treated as measured performance.

The ball demo defaults to COCO class 32. Class 0 in that model is **person**. If you supply a separately trained one-class shuttlecock model, pass `--model path/to/model.onnx --class-id 0` after verifying the model's output class mapping. Tracking and apparent speed calculations live in `src/badminton_ai/tracking.py`. A court-width pixel scale cannot recover the 3D speed of a flying shuttle; its converted speed is an approximation, not a measured physical speed.

## Reproducible sequential baseline

Run this **before** changing the inference implementation, ideally on the Raspberry Pi using the same input, model, power mode and environment for later comparisons:

```bash
python scripts/benchmark.py --input assets/demo_inputs/badminton_sample.mp4 --mode pose --frames 200 --warmup 10 --output outputs/benchmarks/pi_pose_baseline.json
python scripts/benchmark.py --camera 0 --mode pose --frames 300 --warmup 10 --output outputs/benchmarks/pi_camera_baseline.json
python scripts/benchmark.py --input assets/demo_inputs/badminton_sample.mp4 --mode ball --model src/perception/yolov8n.onnx --frames 200 --warmup 10 --output outputs/benchmarks/pi_ball_baseline.json
```

The ball command needs a model already present at the specified path. The benchmark never silently downloads one. Reports include source reported FPS, measured processing throughput, p50/p95 ONNX `session.run` time, p50/p95 inference including preprocessing and postprocessing, p50/p95 read-start-to-result time, process CPU and RSS. Warmup and model loading are excluded. This sequential baseline has no bounded queue or driver frame counter, so dropped-frame rate and true exposure-to-result latency are `null`, **not zero**. File throughput is not camera FPS. Reports are environment-specific and no Raspberry Pi numbers are claimed here.

## Bounded pipeline replay

After the sequential baseline, run the same recording at its metadata FPS through one capture worker and a bounded latest-frame queue:

```bash
python scripts/benchmark_pipeline.py --input assets/demo_inputs/badminton_sample.mp4 --mode pose --frames 200 --warmup 10 --queue-size 2 --output outputs/benchmarks/pipeline_replay_pose.json
```

The report adds capture/processing FPS, software queue depth and dropped-frame rate, p50/p95 inference and read-to-output latency, CPU and RSS. On overflow the queue discards the oldest frame to keep inference fresh. One capture worker lets capture continue during inference; the calling thread runs inference and output. Software drops exclude camera-driver drops. Read-to-output begins **after** decoding and is not exposure-to-output latency. A capture or inference failure aborts without producing a successful report.

An [x86 cloud recorded-video benchmark](docs/cloud_benchmark_2026-10-05.md) and its raw JSON reports are checked in. The paced 24 FPS replay processed 153 of 200 frames at 18.064 FPS and discarded 47 old frames in its software queue. The sequential run processed 200 frames at 17.732 FPS. These figures are specific to that cloud run; Pi FPS, camera-driver drops and exposure latency remain unmeasured.

Run the software tests without a camera or ONNX runtime:

```bash
python -m unittest discover -s tests -v
```

## Exploratory shot candidates

The existing report script guesses shot frames from arm-speed peaks and extrapolates shuttle speed from wrist speed. There is no labeled rally set to validate those guesses. The new in-process event module deliberately reports **ShotCandidate**, not confirmed `ShotDetected`:

```bash
python scripts/inspect_shot_candidates.py --input assets/demo_inputs/badminton_sample.mp4 --threshold-px-s 300 --max-frames 200 --output outputs/shot_candidates.json
```

The threshold is an exploratory image-pixel value, not a calibrated physical speed. The command writes session IDs, unique event IDs, ordered sequence numbers, UTC processing timestamps and video-relative frame times. It selects the largest detected person per frame, so player identity can switch; a wrist-speed peak can occur without racket contact. A smoke run completed 200 checked-in video frames and emitted five **unverified** candidates with this threshold. No shot precision or recall is claimed. `src/badminton_ai/events.py` keeps dispatch synchronous in one process; a queue or network broker would not solve a current requirement.

Add `--db outputs/events.sqlite3 --session-id my-session` to persist those events in SQLite. The small event log stores sessions and ordered events, accepts an exact replay of the same event ID once, and rejects conflicting event IDs or session sequences. It can resume a session at the next stored sequence. Reprocessing a video intentionally creates new event IDs; use a new session ID for a new analysis run. This is a local log of **candidates**, not a verified Shot table or a backend API. A 200-frame software smoke run persisted five candidates and read them back in sequence.

## Development order

Use recorded video to develop and test inference, tracking, events, storage and a robot simulator before integrating hardware. Preserve the existing ONNX demo while extracting independent responsibilities into a small package. The bounded queue is covered by software failure and overflow tests, but its Pi performance remains unmeasured. The historical ~30 FPS Pi screenshot is a clue, not a current benchmark. The [audit](docs/repository_audit.md) records hardware and training-data gates.
