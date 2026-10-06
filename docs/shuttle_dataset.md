# Shuttlecock detection dataset

There is **no labeled shuttlecock image dataset** in this repository. The three JSON files in `data/annotations/` are templates, and the demo videos are not ground truth. The generic YOLOv8n model has not been fine-tuned here. See the [public data audit](public_data_audit.md) for external sources, the local source-video regrouping, label-quality findings and unresolved broadcast/footage rights. Do not train for a published result on an uncleared source.

Organize newly labeled images and one-class YOLO boxes as follows, using different recorded sessions for each split:

```text
dataset-root/
  images/{train,val,test}/frame_name.jpg
  labels/{train,val,test}/frame_name.txt
```

Each label row is `0 center_x center_y width height` with coordinates normalized to `[0, 1]`. A negative image gets an **empty** matching `.txt` file. Keep original video/session IDs with your dataset so that adjacent frames from one rally never appear in both training and evaluation splits. Include near and far shuttles, occlusions, motion blur, different lighting and court backgrounds. The final test split must stay untouched during model selection.

Run `python scripts/prepare_shuttle_dataset.py --dataset /path/to/dataset-root`. The command verifies decodable images, matched labels, finite in-bounds one-class boxes, positive examples in every split and exact image duplicates across splits. It writes `data.yaml` into that dataset root with an absolute `path`, plus `train`, `val`, `test`, and `names: {0: shuttlecock}` entries. It **cannot** detect near-duplicate frames, confirm that boxes mark actual shuttlecocks, or judge whether the dataset covers different conditions. Do not interpret a successful validation as model accuracy.

## Train, test and export once labels exist

On a machine with a suitable GPU, install `requirements-training.txt`. Run:

```bash
python scripts/train_shuttle.py --dataset /path/to/dataset-root --device 0 --epochs 50 --batch 8
python scripts/benchmark.py --input assets/demo_inputs/badminton_sample.mp4 --mode ball --model /path/to/best.onnx --class-id 0 --frames 200 --warmup 10 --output outputs/benchmarks/shuttle_on_this_machine.json
```

The training command starts from the public pretrained `yolov8n.pt` weights (Ultralytics may download them), uses train and validation splits for model selection, evaluates the selected `best.pt` on the held-out test split, and exports a fixed 640-pixel raw one-class ONNX model. It verifies the `[1, 3, 640, 640]` input and `[1, 5, N]` output expected by the existing detector. The report beside the training weights records test precision, recall, mAP50, mAP50-95, export size, model hashes, split counts and package versions. Training and ONNX export have **not** run on real shuttlecock labels yet. An export shape check proves format compatibility, not prediction accuracy.

The initial augmentation is deliberately modest: 10% scale, horizontal flips, no mosaic or vertical flips. These are starting settings for small, fast targets, not a measured optimum. Keep the test split untouched while comparing training settings. Benchmark the ONNX file on the actual edge machine with the same frames, warmup and image size before selecting a deployed model. The training report leaves inference latency `null` until a separate real benchmark is recorded. The demo accepts class 0 via `--class-id 0`; its default class 32 remains the generic COCO sports-ball model.
