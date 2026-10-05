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

## Reproducible sequential baseline

Run this **before** changing the inference implementation, ideally on the Raspberry Pi using the same input, model, power mode and environment for later comparisons:

```bash
python scripts/benchmark.py --input assets/demo_inputs/badminton_sample.mp4 --mode pose --frames 200 --warmup 10 --output outputs/benchmarks/pi_pose_baseline.json
python scripts/benchmark.py --camera 0 --mode pose --frames 300 --warmup 10 --output outputs/benchmarks/pi_camera_baseline.json
python scripts/benchmark.py --input assets/demo_inputs/badminton_sample.mp4 --mode ball --model src/perception/yolov8n.onnx --frames 200 --warmup 10 --output outputs/benchmarks/pi_ball_baseline.json
```

The ball command needs a model already present at the specified path. The benchmark never silently downloads one. Reports include source reported FPS, measured processing throughput, p50/p95 ONNX `session.run` time, p50/p95 inference including preprocessing and postprocessing, p50/p95 read-start-to-result time, process CPU and RSS. Warmup and model loading are excluded. This sequential baseline has no bounded queue or driver frame counter, so dropped-frame rate and true exposure-to-result latency are `null`, **not zero**. File throughput is not camera FPS. Reports are environment-specific and no Raspberry Pi numbers are claimed here.

Run the benchmark's dependency-free tests:

```bash
python -m unittest discover -s tests -v
```

## Next stage

Preserve the existing ONNX demo while extracting the measured capture and inference responsibilities into a small package. Add a bounded latest-frame queue only after a real camera baseline exposes latency accumulation. Training, robot, backend and dashboard work depend on actual data or protocol evidence; the [audit](docs/repository_audit.md) records those gates.
