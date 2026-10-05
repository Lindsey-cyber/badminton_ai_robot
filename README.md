# Real-Time Edge AI Badminton Training System (in progress)

The repository currently contains **offline vision demos and data collection tools**, not an integrated real-time training system. See [the repository audit](docs/repository_audit.md) for an evidence-based inventory and the staged engineering plan. No shuttlecock fine-tuning or Raspberry Pi result has been verified.

## Run the existing demos

Use Python 3.10+ and install `pip install -r requirements.txt` in a virtual environment. From the repository root:

```bash
python scripts/demo_pose_inference.py --input assets/demo_inputs/badminton_sample.mp4 --max-frames 100
python scripts/demo_badminton_detection.py --input assets/demo_inputs/badminton_sample.mp4
python scripts/stereo_distance_demo.py --mode simulate
```

The ball demo may download a generic YOLOv8n model. It searches the COCO `sports ball` class; its detections are not validated shuttlecock detections. The pose demo uses the checked-in `src/pose/yolov8n-pose.onnx` model. Historical HTML reports in `outputs/reports/` contain heuristic and synthetic quantities; they are not measured performance or model evaluation.

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

Compare a future benchmark with a baseline **from the same device, input, model and measurement mode**:

```bash
python scripts/compare_benchmarks.py outputs/benchmarks/pi_pipeline_baseline.json outputs/benchmarks/pi_pipeline_after.json --max-fps-drop 0.15 --max-p95-rise 0.20 --max-rss-rise 0.20
```

The command exits 1 for a threshold regression and 2 when runs cannot be compared. It checks platform, Python/runtime versions, model hash, input identity, source FPS, queue size, warmup, frame count and latency boundary before comparing processed FPS, p95 latency and p95 RSS. Thresholds are starting tolerances, not statistically established Pi limits. Rerun on a controlled device to investigate a failure; shared GitHub runners are used for functional tests, not a strict FPS gate. Changing the model or timing boundary requires a new baseline and a separate accuracy/latency trade-off evaluation.

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

For a single bounded replay from recording through pose inference, candidate detection and SQLite, run:

```bash
python scripts/run_training.py --input assets/demo_inputs/badminton_sample.mp4 --frames 200 --queue-size 2 --threshold-px-s 300 --db outputs/events.sqlite3 --output outputs/training_replay.json
```

The report includes captured/processed FPS, p50/p95 read-to-output and inference time, dropped software frames, queue depth, CPU, RSS and the new session ID. Use a new session ID per replay. Event writes happen in the same output callback as the vision pipeline, so its read-to-output latency includes candidate processing and SQLite commits. If the API is down, replay still records events locally; the API can read them after restart. No robot action is issued from an unverified wrist peak.

One [raw integrated cloud replay](outputs/benchmarks/cloud_training_replay.json) processed 164 of 200 recorded frames at 19.512 FPS with 36 software drops and read-to-output p50/p95 of 104.220/140.959 ms. It persisted five **unverified** candidates. A separate local run on the same video processed 146 frames and yielded two candidates; CPU load and timing affect which frames survive, so candidate counts are not deterministic under backpressure. These cloud measurements are not Pi results or shot accuracy, and the corresponding local SQLite databases are not published.

## Robot simulator

`src/badminton_ai/robot.py` provides a deterministic software robot for tests and development. Commands have caller-supplied IDs and an immediate acceptance ACK; that ACK does **not** mean motion or launch completed. `advance(seconds)` advances simulated motion at a configured speed, consumes launch time, and exposes the current state, position, active command and pending queue depth. The pending queue is bounded; a normal stop clears active and pending commands. Emergency stop latches until an explicit simulation-only reset and rejects normal commands. Repeating an identical command ID returns its original ACK without executing it again; reuse with different contents is an error.

```python
from badminton_ai.robot import SimulatedRobot

robot = SimulatedRobot(move_speed_m_s=2.0)
robot.move_to(2.0, 0.0, command_id="move-1")
robot.launch(command_id="launch-1")
robot.advance(1.0)
print(robot.get_state())
```

Use `PYTHONPATH=src python your_script.py` for the example from the repository root. The coordinates and timing are **simulation parameters**, not measured court motion or a safe motor controller. Command history is retained in memory for the lifetime of one simulator instance to protect duplicate IDs. No transport, hardware ACK timeout, retry policy or Raspberry Pi adapter is defined yet: there is no hardware protocol to test those behaviors against. Hardware integration remains a separate stage.

## Local event API

Install the optional dependencies and start a local server against the same SQLite path as the candidate inspection command:

```bash
python -m pip install -r requirements-api.txt
python scripts/run_api.py --db outputs/events.sqlite3
```

`GET /health`, `GET /sessions`, `GET /sessions/{session_id}/events?after_sequence=-1&limit=100`, `POST /sessions`, and `POST /events` expose the stored candidate log. The POST event body includes `event_id`, `session_id`, nonnegative `sequence`, timezone-aware `timestamp_utc`, `type: "ShotCandidate"`, and `payload`. An identical retry returns `{"inserted": false}`; an ID or order conflict returns HTTP 409. `ws://127.0.0.1:8000/sessions/{session_id}/stream?after_sequence=-1` first replays stored events, then checks SQLite every 250 ms for new ones. Reconnect with the last received sequence to resume without a gap. A video analysis process can continue writing SQLite while the API or a client is disconnected. Start the API only on a trusted local interface: authentication and remote deployment have not been implemented.

The API currently streams **candidate events**, not confirmed shots, live images, FPS or robot telemetry. Polling a durable local log is sufficient for this single-machine setup; there is no measured need for a message broker. The additional API tests require `requirements-api.txt`; the vision and simulator tests do not.

## Software CI

Install `requirements-dev.txt` and run `python -m pytest tests -q`. GitHub Actions runs this on `main` pushes and pull requests with Python 3.12, including a 24-frame integration run through the checked-in video, ONNX pose model and bounded pipeline. It asserts frame accounting and successful inference, not an arbitrary cloud FPS threshold or unverified shot accuracy. Hardware tests and performance regression thresholds require an actual edge baseline and remain separate.

## Shuttlecock training data

No labeled shuttlecock dataset exists in the repository; the generic COCO sports-ball model is not fine-tuned. The [dataset guide](docs/shuttle_dataset.md) defines the one-class train/val/test layout and the validation command that generates an actual Ultralytics `data.yaml` once real labels are supplied. Exact duplicate images and malformed boxes are rejected, but nearby frames from a single video must be grouped by recording before splitting. No accuracy metric has been measured.

`scripts/train_shuttle.py` is the runnable fine-tuning, held-out evaluation and fixed-shape ONNX export entry point once labeled data is available. It writes a report with measured test metrics and model hashes only after a real training run. Use `--class-id 0` with the exported one-class model in the ball demo and benchmarks; the generic COCO model uses class 32. Export compatibility is checked before writing the report. Target-device inference latency still requires a separate benchmark; no trained weights or training result is claimed here.

The old self-installing `setup_and_fix.py` and heuristic HTML report generator were removed. The former could download or overwrite model files as a side effect; the latter substituted assumed quantities, including shuttle speed, and its demo generated synthetic metrics. Install from the explicit requirements files and use the measured JSON tools above. Existing HTML artifacts remain as historical examples with [provenance notes](outputs/reports/README.md).

The synthetic pose-video generator was also retired; its [historical video](outputs/demo_videos/README.md) remains labeled as an animation. The recorded-video path above is the current software demonstration.

## Development order

Use recorded video to develop and test inference, tracking, events, storage and a robot simulator before integrating hardware. Preserve the existing ONNX demo while extracting independent responsibilities into a small package. The bounded queue is covered by software failure and overflow tests, but its Pi performance remains unmeasured. The historical ~30 FPS Pi screenshot is a clue, not a current benchmark. The [audit](docs/repository_audit.md) records hardware and training-data gates.

The multi-camera collector is an **unsynchronized sequential OpenCV reader**. Its timestamp precedes reading each camera, so it cannot establish stereo exposure alignment. `--dry-run` prints settings without creating a session directory. An old `--audio` flag only created an empty folder and was removed; audio capture remains a hardware-stage task if needed.
