# Recorded-video software benchmark (2026-10-05)

This measurement ran in an ephemeral **x86_64 Linux cloud container**, not on Raspberry Pi. The input is the checked-in `assets/demo_inputs/badminton_sample.mp4` (720 × 1280, metadata 24 FPS), and the checked-in `src/pose/yolov8n-pose.onnx` (13,514,572 bytes; SHA-256 in the JSON reports). Python 3.12.14, NumPy 2.5.3, OpenCV headless 5.0.0.93 and ONNX Runtime 1.30.0 were used. ONNX Runtime reported `AzureExecutionProvider` and `CPUExecutionProvider`; this environment emitted a CPU vendor detection warning, so these numbers must not be treated as representative of a known CPU or the Pi.

Run from the repository root, with dependencies installed:

```bash
python scripts/benchmark.py --input assets/demo_inputs/badminton_sample.mp4 --mode pose --frames 200 --warmup 10 --output outputs/benchmarks/cloud_pose_baseline.json
python scripts/benchmark_pipeline.py --input assets/demo_inputs/badminton_sample.mp4 --mode pose --frames 200 --warmup 10 --queue-size 2 --output outputs/benchmarks/cloud_pipeline_replay.json
```

| Run | Input behavior | Captured | Processed | Software queue drops | Processed FPS | Timing boundary and p50 / p95 | Process CPU | RSS p95 |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| Sequential | Decode as fast as inference permits | N/A | 200 | Unknown | 17.732 | Read start to result: 54.768 / 71.763 ms | 407.36% | 240.284 MB |
| Bounded replay | Pace at video metadata 24 FPS; capacity 2 | 200 | 153 | 47 (23.5%) | 18.064 | Read completion to output: 116.583 / 146.958 ms | 411.70% | 266.420 MB |

The two latency boundaries differ and must not be compared as like-for-like end-to-end latencies. The replay includes queue residence in its read-to-output number. The sequential loop has no software queue and cannot know camera-driver drops. The bounded replay processed the freshest available frames, but **did not improve inference throughput** in this run; its cost included more memory and discarded frames. The source is a file, not a camera, so neither run measures exposure-to-output latency or validates camera disconnect recovery. CPU percentage is process CPU time divided by wall time: 100% means one fully used core.

Raw reports: [`cloud_pose_baseline.json`](../outputs/benchmarks/cloud_pose_baseline.json) and [`cloud_pipeline_replay.json`](../outputs/benchmarks/cloud_pipeline_replay.json). This is one run of each variant, not a statistical hardware comparison. Repeat on the same Pi and with the same model/input before drawing performance conclusions or attempting optimization.
