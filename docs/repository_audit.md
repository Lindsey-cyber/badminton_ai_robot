# Repository audit — 2026-10-05

Scope: `main` at `962603a35d7ce110a6a45ccd9cc2ec0267dd481e`. This is a source inspection, not hardware validation. The initial commit has no pre-existing tests or root README.

## What exists

| Area | Evidence | Current status |
| --- | --- | --- |
| Layout | `scripts/` contains eight standalone Python files; `src/pose/` contains one ONNX file, plus duplicate ONNX and `.pt` at root | Demo-oriented, no importable application package |
| Pose | `scripts/demo_pose_inference.py` loads `yolov8n-pose.onnx`, letterboxes to 640, runs ONNX Runtime, decodes 17 COCO keypoints | Functional-looking offline/camera demo; no annotated accuracy evaluation |
| Shuttle detection/tracking | `scripts/demo_badminton_detection.py` downloads general YOLOv8n ONNX, selects COCO class 32 `sports ball`, uses a Kalman filter and pixel speed approximation | General-ball heuristic, not a shuttlecock-trained detector or validated flight tracker |
| Model provenance | Checked-in `src/pose/yolov8n-pose.onnx` and root `yolov8n-pose.onnx` share the same Git blob; `yolov8n-pose.pt` is checked in. Pose scripts name Ultralytics release or export from `yolov8n-pose.pt`; ball ONNX is downloaded from an Ultralytics release | Pretrained model provenance is suggested by code; no repository evidence of fine-tuning or export settings for checked-in blob |
| Cameras / Pi | `collect_multicam_data.py` opens 1–4 OpenCV cameras and writes JPEGs and sync metadata; both inference demos can open one camera | No Raspberry Pi-specific adapter, deployment/service, reconnect, bounded queue or independent edge node. Multiple sequential `read()` calls are not hardware synchronization |
| Stereo / trajectory | `stereo_distance_demo.py` has calibration and disparity functions, an example calibration and a random-noise simulation; `live` prints TODO. Detection demo smooths points; report script uses motion/pose heuristics | No validated live stereo shuttle trajectory or landing measurement |
| Data | Three JSON **templates** in `data/annotations/`; three small demo MP4 inputs; generated outputs and HTML reports | No actual labelled image set, train/val/test split, dataset YAML, training logs or fine-tuned weights |
| Dependencies | `requirements.txt` has broad minimum versions; training, pytest and torch are commented out | No pinned or lockfile environment; scripts rely on OpenCV/ONNX Runtime |
| Tests / CI | No `tests/` or workflow in initial tree | No automated gate before this audit |
| Existing benchmark | `outputs/benchmarks/pose_benchmark.json` says 241 video frames, median 5.216 ms and p95 6.845 ms ONNX call; `avg_fps` 177.839 equals 1000 / mean 5.623 ms | Historical macOS-path artifact with no CPU, RSS, capture latency or dropped-frame count; it does **not** establish current Pi or end-to-end FPS |

The HTML report generator's `--demo` branch seeds synthetic shot and ball-speed values and sets `avg_infer_ms` to 48.3. Its reports and the generated pose video must be labelled synthetic, never used as model evaluation or throughput evidence. `video_batch_infer.py` previously measured an empty loop when models failed to load, and suppressed inference exceptions; phase 1 now fails those cases rather than publishing misleading values.

Later software cleanup removed the legacy report generator rather than extending its unvalidated physical estimates, and removed the redundant self-installing `setup_and_fix.py`. Historical report artifacts are retained with an explicit provenance notice in `outputs/reports/README.md`.
The synthetic pose-video generator was likewise retired after the real recorded-video pipeline was integrated; historical video artifacts are labeled in `outputs/demo_videos/README.md`.

## Keep and improve

- Keep the ONNX pose pre/postprocessing and video demo as the baseline integration path. Verify its keypoint output against annotated data before using pose metrics for coaching.
- Keep the collector as an initial capture utility and the stereo calibration/math routines for later measured tests. Do not claim synchronized stereo from sequential OpenCV reads.
- Keep the Kalman filter and report heuristics as prototypes; separate pixel estimates from physical measurements and test with labeled video.
- Keep the three sample videos as reproducible inputs, with provenance and permission to redistribute checked before using them publicly.

## Three largest engineering gaps

1. **Unverified performance and failure semantics.** Reported inference-only FPS is presented alongside camera/demo FPS, but end-to-end delay, dropped frames, CPU and memory were not measured. Sequential capture can hide camera driver drops. Establish the baseline first.
2. **No validated shuttlecock model or event ground truth.** COCO `sports ball` is not a badminton label; annotation files are examples, while shot and speed estimates rely on unvalidated heuristics. Real labels and held-out evaluation are prerequisites for training claims.
3. **No integrated reliable runtime.** Independent scripts have no bounded capture/inference handoff, recoverable camera loop, robot protocol, persistence or tested shutdown. Build one small process after profiling, without adding services or transports prematurely.

## Phase 1 change and measurement limits

- Add `scripts/benchmark.py`, which calls the existing demo inference class unchanged and records throughput, p50/p95 latency, CPU and RSS with input/model/provider metadata. It distinguishes ONNX time, full inference, and read-start-to-result. It deliberately records unknown driver drops and exposure-to-result latency as `null`.
- Add `tests/test_benchmark.py` for percentile behavior and the video EOF/drop distinction; add this audit and a root README.
- Correct `scripts/video_batch_infer.py` so missing models and inference errors do not yield apparently valid inference FPS.

Run `python -m unittest discover -s tests -v` and `python scripts/benchmark.py --help` locally. A real baseline needs OpenCV and ONNX Runtime, the checked-in model and input video. On the Raspberry Pi, run the three commands in README and send the generated JSON files plus `python -V`, `uname -a`, `pip freeze` and camera model/resolution/power mode. Camera driver drop count and true exposure latency still require hardware timestamps or a driver counter; they cannot be inferred from this sequential loop. Do not report a `null` metric as zero.

## Decisions for later phases

Refactor only measured responsibilities out of the demos, preserving their CLIs. Use one process and a bounded latest-frame queue if camera measurement shows stale frames; a separate messaging service has no demonstrated need. A training pipeline needs actual labeled frames and held-out sessions. Robot GPIO/serial details cannot be implemented or validated without the controller protocol and an emergency-stop test on hardware. Backend and dashboard should follow verified event semantics rather than preempt them.
