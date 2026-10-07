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

The first constrained CPU run uses YOLOv8n COCO initialization, 640px input,
seed 42, batch 8, two epochs, 0.1 warmup epoch, mosaic disabled, scale 0.1,
horizontal flip 0.5, and no vertical flip. This short run is an initial
experiment, not a converged training recipe. Its exact Ultralytics arguments
are saved in the run's `args.yaml`. The grouped dataset remains outside Git,
as do its weights and broadcast frames.

`scripts/analyze_shuttle_errors.py` creates validation false-positive and
false-negative cases at a fixed 0.25 confidence and 0.5 IoU operating point.
Those counts are for deciding a V2 change, not a replacement for Ultralytics
precision/recall/mAP. Inspect actual frames for blur, occlusion and lighting;
the script cannot infer those visual causes. `scripts/evaluate_shuttle_model.py`
records a frozen model's validation or held-out test metrics and hashes.
`scripts/check_onnx_parity.py` compares raw .pt and ONNX predictions on the
same tensors produced by the existing detector preprocessing.

## V1 validation and error analysis

V1 completed two epochs from official `yolov8n.pt` on 5,391 video-2 training
images. The raw [V1 report](../outputs/experiments/v1_val_report.json) contains
the dataset/manifest, starting weights, `best.pt`, `args.yaml` and `results.csv`
hashes and model size. `best.pt` is 6,243,434 bytes and remains outside Git.

| Epoch | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.21418 | 0.13416 | 0.05516 | 0.00637 |
| 2 | 0.16384 | 0.14650 | 0.05382 | 0.00658 |

These are Ultralytics validation metrics for **one video group**; the held-out
test group is still sealed. They do not establish production accuracy. At a
separate fixed confidence of 0.25 and IoU 0.5, error triage matched 31 of
1,215 validation labels, with 170 unmatched predicted boxes. Lowering
confidence to 0.05 matched 276 labels but produced 3,790 unmatched boxes.
These operating-point counts are **not** the mAP table above. Of the 1,184
misses at confidence 0.25, 167 had a nearby prediction center within 10
pixels; tiny offsets can fail IoU on boxes only about 6.5 × 11.5 pixels in
the validation group. Magnified sampled frames show some visible tiny bright
targets and several ambiguous regions. Complete human ground-truth correction
and footage rights clearance remain outstanding.

V2 changes one factor: disable random scale augmentation (`scale=0.1` →
`scale=0.0`) to avoid shrinking already narrow targets during training. It
uses the same pretrained initialization, source split, seed, batch, input size,
and two epochs. Compare V1/V2 on validation only before choosing one for a
single held-out test evaluation.

## V2 result and selection

V2 finished two epochs on the same source-group split and initial COCO weights.
Its raw [validation report](../outputs/experiments/v2_val_report.json) records
the model, dataset, configuration and export hashes. The validation experiment
did **not** support turning off scale augmentation:

| Model, selected epoch | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| V1, epoch 2 | 0.16384 | 0.14650 | 0.05382 | 0.00658 |
| V2, epoch 2 | 0.19567 | 0.09300 | 0.03894 | 0.00454 |

The separate V2 `best.pt` validation measured P 0.19652, R 0.09383,
mAP50 0.03889 and mAP50-95 0.004539. The small difference from the epoch
row is a separate validator invocation; both favor V1 on the predeclared
mAP50-95 criterion. The original V1 weights were lost when the temporary
execution environment reset. A same-parameter V1 reproduction was interrupted
by another execution outage before its second epoch completed. No selected
model's held-out test, ONNX parity or bounded-pipeline benchmark is claimed.
This is a negative ablation result, not production readiness.
