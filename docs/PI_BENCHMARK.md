# Raspberry Pi validation handoff

No Raspberry Pi measurement is available yet. Run this only when the camera
and Pi are physically present. The same `VisionPipeline` and ONNX inference
path used for recorded replay is measured here; `--camera` never fabricates a
recording or a model result.

## Setup

On the Pi, clone this repository and use a Python version with working ARM
wheels for OpenCV and ONNX Runtime. From its root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -c "import cv2, onnxruntime; print(cv2.__version__, onnxruntime.__version__)"
```

Connect a camera that OpenCV can open as index 0. If it is another index,
replace `--camera 0`. Test the bundled pose model first; the provisional
one-class shuttle model is not published because source-media rights and
annotation review remain unresolved.

## Measure

```bash
python scripts/benchmark_pipeline.py \
  --camera 0 --mode pose --frames 300 --warmup 10 --queue-size 2 \
  --width 640 --height 480 --camera-fps 30 \
  --output outputs/benchmarks/pi_camera_pose.json

python scripts/benchmark_pipeline.py \
  --input assets/demo_inputs/badminton_sample.mp4 --mode pose \
  --frames 200 --warmup 10 --queue-size 2 \
  --output outputs/benchmarks/pi_video_pose.json
```

Requested camera width, height and FPS can be ignored or adjusted by the
driver. The JSON records the actual values reported by OpenCV. Warmup and
model loading are outside timed metrics. The report contains captured and
processed FPS; inference and read-completion-to-output p50/p95; process CPU
on a one-core-100% scale; process RSS p50/p95; bounded-queue depth and
software dropped-frame rate. Camera exposure-to-output latency and driver
drops remain unknown. A camera read error fails the run; it does not produce
a success report or silently reconnect.

For reproducibility, also save the environment next to the two JSON files:

```bash
uname -a > outputs/benchmarks/pi_environment.txt
python -V >> outputs/benchmarks/pi_environment.txt 2>&1
python -m pip freeze >> outputs/benchmarks/pi_environment.txt
```

Send `pi_camera_pose.json`, `pi_video_pose.json`, and `pi_environment.txt`.
Also state Pi model, camera model, power supply and whether it throttled.
Do not compare camera FPS with video FPS as if they shared capture conditions;
use repeat runs on the same input and model for performance changes. The
historical screenshot of approximately 30 FPS is not a reproducible baseline.
