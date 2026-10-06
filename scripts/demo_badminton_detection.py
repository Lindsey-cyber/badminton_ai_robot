#!/usr/bin/env python3
"""
demo_badminton_detection.py
===========================
Generic sports-ball detection, trajectory tracking and apparent speed demo.

Features:
  1. Detect COCO sports balls with a generic YOLOv8n ONNX model.
  2. Smooth image-plane tracks with a Kalman filter.
  3. Estimate apparent image-plane speed with optional approximate court scale.
  4. Draw track history and a predicted direction.
  5. Write an annotated video.

Usage:
    python demo_badminton_detection.py --input assets/demo_inputs/badminton_sample.mp4 --model yolov8n.onnx
    python demo_badminton_detection.py --camera 0 --model yolov8n.onnx

Limitations:
  - The generic pretrained model's COCO class 32 is sports ball, not shuttlecock.
  - No fine-tuned shuttlecock detector or validated accuracy is present.
  - A single camera does not measure a flying shuttle's 3D speed.
"""

import argparse
import json
import logging
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False
    print("[ERROR] Install onnxruntime: pip install onnxruntime")
    sys.exit(1)

from PIL import Image, ImageDraw

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from badminton_ai.tracking import BallKalmanFilter, SpeedEstimator
MODEL_DIR = PROJECT_ROOT / "src" / "perception"
DEFAULT_MODEL_PATH = MODEL_DIR / "yolov8n.onnx"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "demo_videos"
BENCHMARK_DIR = PROJECT_ROOT / "outputs" / "benchmarks"

# The generic COCO model uses class 32 for sports ball. Its class 0 is person.
# A future shuttlecock-specific model may use class 0, but that must be
# selected explicitly after validating its class mapping.
BALL_CLASS_IDS = {32: "sports ball"}
TARGET_CLASS_IDS = [32]

# Trajectory colors, from older to newer
TRAJECTORY_COLORS = [
    (0, 255, 255),   # yellow-green
    (0, 200, 200),
    (0, 150, 200),
    (0, 100, 200),
    (0, 50, 255),    # blue
]

MAX_TRAJECTORY_LEN = 30  # Keep at most 30 tracked positions.


# =============================================================================
# YOLOv8 detector
# =============================================================================

class YOLOv8Detector:
    """Generic YOLOv8 object detector (ONNX)."""

    INPUT_SIZE = 640

    def __init__(self, model_path: str, conf_threshold: float = 0.20,
                 target_classes: list = None, iou_threshold: float = 0.45):
        if not 0 <= conf_threshold <= 1 or not 0 < iou_threshold < 1:
            raise ValueError("Confidence must be in [0, 1] and NMS IoU in (0, 1)")
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.target_classes = target_classes  # None detects all classes.

        logger.info(f"Loading detector: {model_path}")
        providers = ort.get_available_providers()
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        logger.info("Detector loaded")

    def preprocess(self, image_bgr: np.ndarray):
        h, w = image_bgr.shape[:2]
        scale = self.INPUT_SIZE / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        pad_h = (self.INPUT_SIZE - new_h) // 2
        pad_w = (self.INPUT_SIZE - new_w) // 2

        if CV2_AVAILABLE:
            resized = cv2.resize(image_bgr, (new_w, new_h))
            img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = np.array(Image.fromarray(image_bgr).resize((new_w, new_h)))

        img_padded = np.full((self.INPUT_SIZE, self.INPUT_SIZE, 3), 114, dtype=np.uint8)
        img_padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = img_rgb

        img_tensor = img_padded.astype(np.float32) / 255.0
        img_tensor = img_tensor.transpose(2, 0, 1)[np.newaxis, ...]

        return img_tensor, scale, pad_h, pad_w, h, w

    def postprocess(self, outputs, scale, pad_h, pad_w, orig_h, orig_w):
        # YOLOv8 output: [1, 84, num_anchors] (4 box values + 80 classes),
        # or a custom number of classes.
        preds = outputs[0][0].T  # [num_anchors, 84]

        # Select from requested classes before choosing the winning class.
        class_scores = preds[:, 4:]
        if self.target_classes is not None:
            allowed = np.asarray(self.target_classes, dtype=int)
            if (not len(allowed) or np.any(allowed < 0) or
                    np.any(allowed >= class_scores.shape[1])):
                raise ValueError("Target class ID is outside the model's output classes")
            class_ids = allowed[np.argmax(class_scores[:, allowed], axis=1)]
        else:
            class_ids = np.argmax(class_scores, axis=1)
        max_scores = class_scores[np.arange(len(class_ids)), class_ids]
        obj_conf = max_scores  # YOLOv8 has no separate objectness score.

        # Filter low-confidence predictions.
        mask = obj_conf > self.conf_threshold
        preds_filtered = preds[mask]
        class_ids_filtered = class_ids[mask]
        scores_filtered = obj_conf[mask]

        if len(preds_filtered) == 0:
            return []

        # Decode boxes.
        cx = preds_filtered[:, 0]
        cy = preds_filtered[:, 1]
        bw = preds_filtered[:, 2]
        bh = preds_filtered[:, 3]

        x1 = (cx - bw / 2 - pad_w) / scale
        y1 = (cy - bh / 2 - pad_h) / scale
        x2 = (cx + bw / 2 - pad_w) / scale
        y2 = (cy + bh / 2 - pad_h) / scale

        results = []
        for i in range(len(scores_filtered)):
            class_id = int(class_ids_filtered[i])
            results.append({
                "bbox": [float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])],
                "score": float(scores_filtered[i]),
                "class_id": class_id,
                "class_name": BALL_CLASS_IDS.get(class_id, f"class_{class_id}"),
                "center": (float((x1[i] + x2[i]) / 2), float((y1[i] + y2[i]) / 2)),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        kept = []
        for detection in results:
            x1, y1, x2, y2 = detection["bbox"]
            detection["bbox"] = [max(0.0, min(orig_w, x1)),
                                 max(0.0, min(orig_h, y1)),
                                 max(0.0, min(orig_w, x2)),
                                 max(0.0, min(orig_h, y2))]
            x1, y1, x2, y2 = detection["bbox"]
            if x2 <= x1 or y2 <= y1:
                continue
            detection["center"] = ((x1 + x2) / 2, (y1 + y2) / 2)
            area = (x2 - x1) * (y2 - y1)
            duplicate = False
            for previous in kept:
                if previous["class_id"] != detection["class_id"]:
                    continue
                px1, py1, px2, py2 = previous["bbox"]
                overlap = max(0.0, min(x2, px2) - max(x1, px1)) * max(0.0, min(y2, py2) - max(y1, py1))
                previous_area = (px2 - px1) * (py2 - py1)
                if overlap / (area + previous_area - overlap) > self.iou_threshold:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(detection)
        return kept

    def inference(self, image_bgr: np.ndarray):
        img_tensor, scale, pad_h, pad_w, h, w = self.preprocess(image_bgr)
        t0 = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_name: img_tensor})
        t1 = time.perf_counter()
        infer_ms = (t1 - t0) * 1000
        results = self.postprocess(outputs, scale, pad_h, pad_w, h, w)
        return results, infer_ms


# =============================================================================
# Visualization
# =============================================================================

def draw_detections(image: np.ndarray, detections: list, trajectory: deque,
                    speed_info: dict, predictions: list = None) -> np.ndarray:
    """Draw detections, track and apparent speed."""
    vis = image.copy()

    # Draw track history.
    traj_list = list(trajectory)
    for i in range(1, len(traj_list)):
        color_idx = min(int(i / max(len(traj_list), 1) * len(TRAJECTORY_COLORS)),
                        len(TRAJECTORY_COLORS) - 1)
        color = TRAJECTORY_COLORS[color_idx]
        thickness = max(1, int(i / max(len(traj_list), 1) * 4))
        if CV2_AVAILABLE:
            cv2.line(vis,
                     (int(traj_list[i-1][0]), int(traj_list[i-1][1])),
                     (int(traj_list[i][0]), int(traj_list[i][1])),
                     color, thickness)

    # Draw predicted track.
    if predictions and CV2_AVAILABLE:
        for i, pred_pos in enumerate(predictions):
            alpha = 0.8 - i * 0.15
            if 0 < alpha and len(traj_list) > 0:
                if i == 0:
                    start = traj_list[-1]
                else:
                    start = predictions[i-1]
                cv2.line(vis,
                         (int(start[0]), int(start[1])),
                         (int(pred_pos[0]), int(pred_pos[1])),
                         (200, 200, 0), 1)

    # Draw detections.
    for det in detections:
        bbox = det["bbox"]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cx, cy = int(det["center"][0]), int(det["center"][1])

        if CV2_AVAILABLE:
            # Bounding box.
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
            # Center point.
            cv2.circle(vis, (cx, cy), 6, (0, 255, 255), -1)
            # Label.
            label = f"{det['class_name']} {det['score']:.2f}"
            cv2.putText(vis, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Draw apparent speed.
    if speed_info and CV2_AVAILABLE:
        if speed_info.get("speed_kmh") is not None:
            speed_text = f"Speed: {speed_info['speed_kmh']:.1f} km/h"
        else:
            speed_text = f"Speed: {speed_info['pixel_speed']:.0f} px/s"

        cv2.putText(vis, speed_text, (10, vis.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(vis, speed_text, (10, vis.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)

    return vis


def add_info_overlay(image: np.ndarray, fps: float, infer_ms: float,
                     num_detections: int, frame_idx: int) -> np.ndarray:
    if CV2_AVAILABLE:
        info_lines = [
            f"Frame: {frame_idx}",
            f"FPS: {fps:.1f}",
            f"Infer: {infer_ms:.1f}ms",
            f"Detections: {num_detections}",
            "Model: YOLOv8n (ONNX)",
            "Task: Ball Detection",
        ]
        y_start = 20
        for line in info_lines:
            cv2.putText(image, line, (10, y_start),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(image, line, (10, y_start),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_start += 20
    return image


# =============================================================================
# Video processing
# =============================================================================

def process_video(
    detector: YOLOv8Detector,
    input_path: str,
    output_path: str,
    max_frames: int = -1,
    pixels_per_meter: float = None,
) -> dict:
    if not CV2_AVAILABLE:
        logger.error("Video processing requires opencv-python")
        return {}

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error(f"Cannot open: {input_path}")
        return {}

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError("Video FPS is unavailable; provide a recording with valid FPS metadata")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video: {w}x{h} @ {fps:.1f}fps, {total} frames")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Initialize tracking.
    kf = BallKalmanFilter()
    speed_est = SpeedEstimator(fps=fps, pixels_per_meter=pixels_per_meter)
    trajectory = deque(maxlen=MAX_TRAJECTORY_LEN)

    # Statistics.
    infer_times = []
    frame_idx = 0
    detected_frames = 0
    speed_samples = []
    fps_window = []
    t_prev = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames > 0 and frame_idx >= max_frames:
            break

        # Detect.
        results, infer_ms = detector.inference(frame)
        infer_times.append(infer_ms)

        # Select the highest-confidence candidate.
        best_det = None
        if results:
            best_det = results[0]
            detected_frames += 1
            cx, cy = best_det["center"]

            # Update the Kalman filter.
            smooth_pos = kf.update((cx, cy))
            trajectory.append(smooth_pos)

            # Estimate apparent speed.
            speed_info = speed_est.update(smooth_pos, frame_idx)
            if speed_info.get("speed_kmh") is not None:
                speed_samples.append(speed_info["speed_kmh"])
        else:
            # Predict without a detection.
            if kf.initialized:
                kf.advance_without_measurement()
                # Do not add an unobserved point to the displayed track.
            speed_info = {"pixel_speed": 0, "speed_kmh": None, "speed_ms": None}

        # Predict the next position.
        future_preds = kf.predict_next(5) if kf.initialized else []

        # Calculate processing FPS.
        t_now = time.perf_counter()
        fps_window.append(1.0 / max(t_now - t_prev, 1e-6))
        if len(fps_window) > 30:
            fps_window.pop(0)
        fps_display = np.mean(fps_window)
        t_prev = t_now

        # Annotate the frame.
        vis = draw_detections(frame, results if best_det else [], trajectory,
                              speed_info if best_det else {}, future_preds)
        vis = add_info_overlay(vis, fps_display, infer_ms, len(results), frame_idx)

        writer.write(vis)
        frame_idx += 1

        if frame_idx % 50 == 0:
            detect_rate = detected_frames / max(frame_idx, 1) * 100
            logger.info(f"Frame {frame_idx}/{total}, detection rate: {detect_rate:.1f}%, "
                        f"inference: {np.mean(infer_times[-30:]):.1f}ms")

    cap.release()
    writer.release()

    stats = {
        "input": input_path,
        "output": output_path,
        "total_frames": frame_idx,
        "detected_frames": detected_frames,
        "detection_rate": detected_frames / max(frame_idx, 1),
        "avg_infer_ms": float(np.mean(infer_times)) if infer_times else 0,
        "avg_fps_inference": 1000.0 / np.mean(infer_times) if infer_times else 0,
        "avg_speed_kmh": float(np.mean(speed_samples)) if speed_samples else None,
        "max_speed_kmh": float(np.max(speed_samples)) if speed_samples else None,
    }

    logger.info("=" * 50)
    logger.info("Generic sports-ball detection summary:")
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 50)
    logger.info("Generic COCO sports-ball predictions are not validated shuttlecock detections")
    logger.info("A labeled shuttlecock dataset is needed before reporting detector accuracy")

    return stats


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generic sports-ball detection and tracking demo")
    parser.add_argument("--input", "-i", type=str, help="Input video path")
    parser.add_argument("--camera", "-c", type=int, default=None)
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--model", "-m", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--iou", type=float, default=0.45,
                        help="Per-class NMS IoU threshold for raw YOLO output")
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--ppm", type=float, default=None,
                        help="Court-plane pixels per meter for approximate speed")
    parser.add_argument("--class-id", type=int, default=32,
                        help="Target class ID in this model (default: COCO sports ball 32)")
    parser.add_argument("--all-classes", action="store_true",
                        help="Detect every class rather than only COCO sports ball")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_file():
        parser.error(f"Model not found: {model_path}. Export a real YOLOv8n ONNX model first, "
                     "or pass --model /path/to/exported.onnx")

    # Target classes.
    target_classes = None if args.all_classes else [args.class_id]
    detector = YOLOv8Detector(str(model_path), conf_threshold=args.conf,
                               target_classes=target_classes, iou_threshold=args.iou)

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.camera is not None:
        # Camera demo without persistent statistics.
        if not CV2_AVAILABLE:
            logger.error("opencv-python is required")
            return

        cap = cv2.VideoCapture(args.camera)
        kf = BallKalmanFilter()
        trajectory = deque(maxlen=MAX_TRAJECTORY_LEN)
        fps = 30
        speed_est = SpeedEstimator(fps=fps, pixels_per_meter=args.ppm)
        frame_idx = 0

        logger.info("Press q to quit")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results, infer_ms = detector.inference(frame)
            if results:
                cx, cy = results[0]["center"]
                pos = kf.update((cx, cy))
                trajectory.append(pos)
                speed_info = speed_est.update(pos, frame_idx)
            else:
                speed_info = {}

            vis = draw_detections(frame, results, trajectory, speed_info)
            vis = add_info_overlay(vis, 0, infer_ms, len(results), frame_idx)
            cv2.imshow("Ball Detection", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_idx += 1

        cap.release()
        cv2.destroyAllWindows()

    elif args.input:
        output_path = args.output or str(output_dir / "ball_detection_output.mp4")
        stats = process_video(detector, args.input, output_path,
                               max_frames=args.max_frames, pixels_per_meter=args.ppm)
        # Save the benchmark.
        bench_path = str(BENCHMARK_DIR / "ball_detection_benchmark.json")
        BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
        with open(bench_path, "w") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        logger.info(f"Benchmark saved: {bench_path}")
    else:
        logger.info("Provide --input <video path> or --camera <device ID>")
        logger.info("Example: python demo_badminton_detection.py --input video.mp4 --model yolov8n.onnx")
        logger.info("")
        logger.info("TrackNetV3 (a separate shuttlecock tracker):")
        logger.info("  git clone https://github.com/qaz812345/TrackNetV3")


if __name__ == "__main__":
    main()
