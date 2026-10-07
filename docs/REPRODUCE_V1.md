# Reproduce the selected V1 model on stable compute

The temporary cloud workspace disconnected twice during CPU training and erased
uncommitted weights. V1 beat the completed V2 scale ablation on validation
mAP50-95 (0.00658 versus 0.00454), but the original V1 weights are gone.
This recipe recreates V1; the source-group test set remains untouched until
the new validation result is checked.

Use a persistent Linux machine with Python 3.12, enough disk for the ~1.6 GB
source mirror, and preferably a CUDA GPU. The public mirror's underlying
broadcast footage rights are unverified. Keep frames and weights private.

```bash
git clone https://github.com/Lindsey-cyber/badminton_ai_robot.git
cd badminton_ai_robot
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-training.txt onnxruntime

git clone --depth 1 https://github.com/muhammadyasin79/Badminton_Analytics_Project.git ../public_data_source
git -C ../public_data_source rev-parse HEAD
python scripts/prepare_public_shuttle_dataset.py \
  --source ../public_data_source/datasets/Shuttlecock.v1i.yolov11 \
  --output ../public_shuttle_grouped \
  --mirror-revision 262c65ad59a71da97ee1a2f5fb16b547c95ab49d
python -c "import json; print(json.load(open('../public_shuttle_grouped/manifest.json'))['dataset_sha256'])"
python -c "import torch; print(torch.cuda.is_available())"
```

Expected source revision: `262c65ad59a71da97ee1a2f5fb16b547c95ab49d`.
Expected dataset hash:
`89f6f74fe229baef12d5ac532de4ff4add047bfbcd7373b37d2bdf891bf61071`.
Stop if either differs. If CUDA is available, use `--device 0`; otherwise
use `--device cpu`. Install a PyTorch build appropriate for the machine if
CUDA is installed but the check prints `False`.

```bash
python scripts/train_shuttle.py \
  --dataset ../public_shuttle_grouped \
  --weights yolov8n.pt --epochs 2 --warmup-epochs 0.1 \
  --imgsz 640 --batch 8 --workers 0 --device 0 \
  --runs-dir "$PWD/outputs/training" --name v1_reproduced \
  --seed 42 --mosaic 0.0 --scale 0.1 --fliplr 0.5
```

The script validates only the video-1 group, exports fixed-shape raw ONNX,
checks its layout, and writes
`outputs/training/v1_reproduced/training_report.json`. Keep
`weights/best.pt` and `weights/best.onnx` private; do not commit media or
model files. The original V1 validation row was P 0.16384, R 0.14650,
mAP50 0.05382, mAP50-95 0.00658. GPU arithmetic can differ slightly; report
actual values rather than copying these.

Send back the dataset hash and these files from
`outputs/training/v1_reproduced/`: `training_report.json`, `args.yaml`,
`results.csv`, `weights/best.pt`, and `weights/best.onnx`. The model
files are about 6.2 MB and 12.2 MB. We can then run the existing
`scripts/check_onnx_parity.py`, evaluate the held-out video-3 group once,
and benchmark the selected ONNX model in `scripts/benchmark_pipeline.py`.
No Raspberry Pi, stereo camera or robot is needed for this handoff.
