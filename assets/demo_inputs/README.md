# Demo inputs

The checked-in sample videos are for exercising the current scripts. They do not include ground-truth shuttlecock or shot labels, so they cannot establish detection accuracy.

To try a different recording, pass its path to the existing demos:

```bash
python scripts/demo_pose_inference.py --input assets/demo_inputs/badminton_sample.mp4
python scripts/demo_badminton_detection.py --input assets/demo_inputs/badminton_sample.mp4 --model /path/to/real/yolov8n.onnx
python scripts/video_batch_infer.py --input-dir assets/demo_inputs/ --mode pose
```

The historical `outputs/demo_videos/pose_demo.mp4` is a generated animation with synthetic frames and overlays, not a model demonstration. The generator was retired in favor of the checked-in recorded-video pipeline. `python scripts/stereo_distance_demo.py --mode simulate` is also a simulation, not a calibrated camera measurement.

Recordings intended for training need explicit distribution rights and session-level train/validation/test splits. See the root README and audit before interpreting results.
