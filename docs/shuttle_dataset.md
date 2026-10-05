# Shuttlecock detection dataset

There is **no labeled shuttlecock image dataset** in this repository. The three JSON files in `data/annotations/` are templates, and the demo videos are not ground truth. The generic YOLOv8n model has not been fine-tuned here.

Organize newly labeled images and one-class YOLO boxes as follows, using different recorded sessions for each split:

```text
dataset-root/
  images/{train,val,test}/frame_name.jpg
  labels/{train,val,test}/frame_name.txt
```

Each label row is `0 center_x center_y width height` with coordinates normalized to `[0, 1]`. A negative image gets an **empty** matching `.txt` file. Keep original video/session IDs with your dataset so that adjacent frames from one rally never appear in both training and evaluation splits. Include near and far shuttles, occlusions, motion blur, different lighting and court backgrounds. The final test split must stay untouched during model selection.

Run `python scripts/prepare_shuttle_dataset.py --dataset /path/to/dataset-root`. The command verifies decodable images, matched labels, finite in-bounds one-class boxes, positive examples in every split and exact image duplicates across splits. It writes `data.yaml` into that dataset root with an absolute `path`, plus `train`, `val`, `test`, and `names: {0: shuttlecock}` entries. It **cannot** detect near-duplicate frames, confirm that boxes mark actual shuttlecocks, or judge whether the dataset covers different conditions. Do not interpret a successful validation as model accuracy.
