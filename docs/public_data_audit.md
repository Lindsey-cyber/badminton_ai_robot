# Public shuttlecock data audit (2026-10-06)

The repository has **not** fine-tuned a shuttlecock model. Public data was inspected
outside the repository; no third-party images, labels or model weights are committed.

| Source | Annotation and acquisition | Rights / decision |
| --- | --- | --- |
| [Mathieu Cartron, Roboflow v1](https://universe.roboflow.com/mathieu-cartron/shuttlecock-cqzy3/dataset/1) | 8,053 YOLO-box images, declared CC BY 4.0. The [attributed GitHub mirror](https://github.com/muhammadyasin79/Badminton_Analytics_Project) includes the full export. | Publisher's dataset license is stated, but inspected images visibly contain professional broadcast graphics. The underlying footage rights have **not** been independently cleared. Use only for local data-quality investigation until rights are resolved; do not publish images, weights or accuracy claims based on this source as cleared work. |
| [badyfriends, Roboflow v3](https://universe.roboflow.com/badyfriends/badminton-shuttlecock-dv7zr/dataset/3) | One-class YOLO-box export, 2,067 generated images in v3; official download requires a Roboflow account. | Publisher labels it Public Domain; original image provenance is unspecified. Generated variants must be grouped with their originals to avoid leakage. Not acquired. |
| [ETH Zürich ShuttlecockFinder](https://sites.google.com/leggedrobotics.com/shuttlecockfinder/startseite) | Authors report 20,510 semi-automatically labeled frames across 11 self-captured backgrounds; [download folder](https://drive.google.com/drive/folders/1neBfC7-Fp-l53muxZiIbIET6lYH5xv9s?usp=sharing). | Best fit for diverse real footage, but the project does not state a separate image/dataset license. Seek written permission before model training or redistribution. The GitHub repository's AGPL license applies to code and does not by itself grant rights to the separate images. |
| [Student self-recorded badminton games](https://github.com/astrochialinko/Badminton-Shuttlecock-Tracking) | Four games, 2,098 frames with shuttle center coordinates, [download folder](https://drive.google.com/drive/folders/1mNntqLRaQkIhUmLZPC19Yc7HXesgBrpQ?usp=share_link). | No shuttle bounding boxes; a fixed-size box around a center is **not** verified ground truth. Separate video permission is unclear. |
| [RacketVision](https://huggingface.co/datasets/linfeng302/RacketVision) | Match videos and framewise shuttle centers; no shuttle extent boxes. | Broadcast footage and center-only labels make it unsuitable as a cleared YOLO-box training set here. |

## What was actually inspected

The Mathieu v1 export was cloned from the attributed GitHub mirror at commit
`262c65ad59a71da97ee1a2f5fb16b547c95ab49d`. Its `README.dataset.txt` declares
CC BY 4.0 and its Roboflow export README specifies 8,053 images, 640 × 640 stretch,
and no augmentation. The downloaded media remained outside this repository.

The publisher's train/valid/test folders mix frames from filename-derived source groups
`video_label_1`, `video_label_2` and `video_label_3` in **every** split. Those
published splits can leak neighboring frames. `scripts/prepare_public_shuttle_dataset.py`
reassembles whole groups: video 2 → train (5,391 images, 5,322 boxes), video 1 →
validation (1,215 images/boxes), video 3 → held-out test (1,447 images/boxes).
The 69 empty labels in training are retained as negatives. The groups are inferred
from filenames and visually sampled; their original recording/session boundaries
were not supplied separately. Local image decoding,
YOLO box bounds, matched labels and cross-split exact image hashes passed validation.
The manifest for this local preparation has SHA-256
`89f6f74fe229baef12d5ac532de4ff4add047bfbcd7373b37d2bdf891bf61071`.

Nearly all labeled boxes in each video have identical width and height: 5,313 of
5,322 boxes for video 2 have width `0.00703125`, height `0.01171875` in normalized
640-pixel coordinates. Twelve sample annotations were visually inspected, but a
human has **not** checked all 8,053 labels or confirmed the apparent extent of
blurred shuttles. Only three source videos are available here, with one per split;
results would have high scene variance and cannot establish general edge accuracy.

To reproduce the local grouping after separately obtaining the same licensed export:

```bash
python scripts/prepare_public_shuttle_dataset.py \
  --source /path/to/Shuttlecock.v1i.yolov11 \
  --output /path/to/grouped-shuttle \
  --mirror-revision 262c65ad59a71da97ee1a2f5fb16b547c95ab49d
```

The generated `manifest.json` records the source, attribution, revision, split
groups, image/box counts and dataset hash; `data.yaml` feeds the existing trainer.
This is a **data structure check**, not legal clearance or verified box ground truth.
Do not treat the published random split's scores as a held-out test result.
