# Shuttlecock detection experiments

These are **provisional local research results** on the [audited external
dataset](public_data_audit.md). The footage has professional broadcast overlays,
and its underlying media rights remain unverified. Do not redistribute images,
training weights or market this as a rights-cleared model. Filename-derived
groups separate videos 2 (train), 1 (validation) and 3 (sealed test). Dataset
SHA-256: `89f6f74fe229baef12d5ac532de4ff4add047bfbcd7373b37d2bdf891bf61071`.

## Pretrained baseline, validation group 1

The official COCO YOLOv8n weights have 80 classes; `sports ball` is class 32,
while our one-class labels are class 0. Evaluating them directly would score
the wrong class. `scripts/evaluate_pretrained_shuttle.py` makes an **evaluation-only**
label copy from 0 → 32, with local image hard links, then evaluates only the
validation group. Its first attempt with a directory symlink resolved back to
the original class-0 labels and was interrupted; no result from that attempt
was used.

```bash
python scripts/evaluate_pretrained_shuttle.py \
  --dataset /path/to/grouped-shuttle \
  --weights /path/to/yolov8n.pt \
  --workspace /path/to/empty/baseline-coco32-val --split val --batch 8
```

On the current x86 cloud CPU with Python 3.12.14, torch 2.14.1+cpu and
Ultralytics 8.4.174, 1,215 validation images yielded:

| Model | Precision | Recall | mAP50 | mAP50-95 | Model size |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pretrained COCO class 32 | 0.015685 | 0.043621 | 0.000851 | 0.000166 | 6,549,796 bytes |

The raw [baseline report](../outputs/experiments/pretrained_coco32_val.json)
contains exact weights/dataset hashes, versions, split mapping and validator
speed (49.78 ms mean inference/image); that speed is **not** an ONNX bounded
pipeline benchmark or a Raspberry Pi latency. The test group remains untouched.
The tiny fixed-size boxes and one validation scene limit what these metrics mean.

## Fine-tuning and selection

V1 and V2 must be compared on the **validation** group only. The selected
model may be evaluated on video group 3 once, after settings are frozen. The
existing trainer now reports validation scores rather than evaluating test
after each training attempt. Save each run's `args.yaml`, `results.csv`,
weights hash, source dataset hash and ONNX hash before comparing models.
