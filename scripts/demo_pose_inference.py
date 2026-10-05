#!/usr/bin/env python3
"""
demo_pose_inference.py
======================
Human pose estimation demo using the existing YOLOv8n-pose ONNX model.
Supports video files and cameras, draws 17 keypoints and skeleton edges,
and writes annotated output.

Usage:
    # Recorded video
    python demo_pose_inference.py --input assets/demo_inputs/badminton_sample.mp4

    # Live camera
    python demo_pose_inference.py --camera 0

    # Process only the first N frames
    python demo_pose_inference.py --input video.mp4 --max-frames 100

Dependencies:
    pip install opencv-python-headless onnxruntime numpy
    # Install opencv-python instead if GUI display is needed.
"""

import argparse
import os
import sys
import time
import logging
import urllib.request
from pathlib import Path

import numpy as np

# OpenCV is optional for the legacy imageio/Pillow fallback.
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("[WARNING] OpenCV is unavailable; trying the Pillow fallback")

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False
    print("[ERROR] Install onnxruntime: pip install onnxruntime")
    sys.exit(1)

from PIL import Image, ImageDraw, ImageFont

# =============================================================================
# Configuration
# =============================================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# COCO 17 keypoint names
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# COCO skeleton edges (keypoint index pairs)
SKELETON = [
    (0, 1), (0, 2),           # nose to eyes
    (1, 3), (2, 4),           # eyes to ears
    (0, 5), (0, 6),           # nose to shoulders
    (5, 6),                   # shoulders
    (5, 7), (7, 9),           # left arm
    (6, 8), (8, 10),          # right arm
    (5, 11), (6, 12),         # torso
    (11, 12),                 # hips
    (11, 13), (13, 15),       # left leg
    (12, 14), (14, 16),       # right leg
]

# Colors in OpenCV BGR order
COLORS_BGR = {
    "skeleton": (0, 255, 0),     # green skeleton
    "keypoint": (0, 0, 255),     # red keypoint
    "bbox": (255, 165, 0),       # orange bounding box
    "text": (255, 255, 255),     # white text
    "bg": (0, 0, 0),             # black background
}

# Model path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODEL_DIR = PROJECT_ROOT / "src" / "pose"
DEFAULT_MODEL_PATH = MODEL_DIR / "yolov8n-pose.onnx"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "demo_videos"
BENCHMARK_DIR = PROJECT_ROOT / "outputs" / "benchmarks"

# =============================================================================
# Legacy model download fallback
# =============================================================================

MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.onnx"

def download_model(model_path: Path) -> bool:
    """Find or download YOLOv8n-pose.onnx, rejecting empty files."""
    MIN_VALID_SIZE = 1024 * 10  # A file smaller than 10 KiB cannot be this model.

    # Check an existing model first.
    if model_path.exists() and model_path.stat().st_size > MIN_VALID_SIZE:
        logger.info(f"Model exists ({model_path.stat().st_size / 1024 / 1024:.1f}MB): {model_path}")
        return True
    elif model_path.exists():
        logger.warning(f"Invalid model ({model_path.stat().st_size} bytes); retrying download")
        model_path.unlink()

    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Option 1: Ultralytics download and ONNX export.
    try:
        logger.info("Trying Ultralytics download and ONNX export")
        from ultralytics import YOLO
        m = YOLO("yolov8n-pose.pt")          # Ultralytics caches the pretrained .pt file.
        export_path = m.export(format="onnx", simplify=True, imgsz=640)
        import shutil
        shutil.copy(str(export_path), str(model_path))
        logger.info(f"Model exported: {model_path}")
        return True
    except ImportError:
        logger.info("Ultralytics unavailable; skipping ONNX export option")
    except Exception as e:
        logger.warning(f"ONNX export failed: {e}")

    # Option 2: download a pretrained ONNX file.
    download_urls = [
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.onnx",
        "https://huggingface.co/Ultralytics/assets/resolve/main/yolov8n-pose.onnx",
    ]
    for url in download_urls:
        try:
            logger.info(f"Trying download from {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as response:
                total = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                with open(str(model_path), "wb") as f:
                    while chunk := response.read(65536):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            pct = downloaded / total * 100
                            print(f"\r  Download: {pct:.1f}% ({downloaded}/{total} bytes)", end="", flush=True)
                print()
            if model_path.stat().st_size > MIN_VALID_SIZE:
                logger.info(f"Downloaded ({model_path.stat().st_size / 1024 / 1024:.1f}MB): {model_path}")
                return True
            else:
                logger.warning(f"Downloaded file is too small ({model_path.stat().st_size} bytes)")
                model_path.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Download failed: {e}")

    # Explain manual recovery when both options fail.
    logger.error("All model download methods failed")
    logger.info("")
    logger.info("Try one of these steps:")
    logger.info("")
    logger.info("  # Install Ultralytics so the next run can export automatically")
    logger.info("  pip install ultralytics")
    logger.info("")
    logger.info("  # Or download with curl (macOS/Linux):")
    logger.info(f"  curl -L '{download_urls[0]}' -o '{model_path}'")
    logger.info("")
    logger.info("  # Or export with Python:")
    logger.info("  python -c \"from ultralytics import YOLO; YOLO('yolov8n-pose.pt').export(format='onnx')\"")
    logger.info(f"  # Copy the exported yolov8n-pose.onnx to: {model_path}")
    return False


# =============================================================================
# YOLOv8 pose inference
# =============================================================================

class YOLOv8PoseInference:
    """YOLOv8 pose inference with ONNX Runtime."""

    INPUT_SIZE = 640  # Model input size.

    def __init__(self, model_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Create the inference session.
        logger.info(f"Loading model: {model_path}")
        providers = ort.get_available_providers()
        logger.info(f"Available inference providers: {providers}")

        self.session = ort.InferenceSession(
            model_path,
            providers=providers
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        logger.info(f"Model input: {self.input_name}, shape={self.session.get_inputs()[0].shape}")
        logger.info(f"Model outputs: {self.output_names}")
        logger.info("Model loaded")

    def preprocess(self, image_bgr: np.ndarray):
        """Resize, normalize and transpose the image."""
        h, w = image_bgr.shape[:2]

        # Calculate letterbox padding.
        scale = self.INPUT_SIZE / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        pad_h = (self.INPUT_SIZE - new_h) // 2
        pad_w = (self.INPUT_SIZE - new_w) // 2

        # resize
        if CV2_AVAILABLE:
            resized = cv2.resize(image_bgr, (new_w, new_h))
            img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        else:
            img_pil = Image.fromarray(image_bgr).resize((new_w, new_h))
            img_rgb = np.array(img_pil)

        # letterbox
        img_padded = np.full((self.INPUT_SIZE, self.INPUT_SIZE, 3), 114, dtype=np.uint8)
        img_padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = img_rgb

        # normalize [0,255] -> [0,1], HWC -> CHW, add batch dim
        img_float = img_padded.astype(np.float32) / 255.0
        img_tensor = img_float.transpose(2, 0, 1)[np.newaxis, ...]

        return img_tensor, scale, pad_h, pad_w, h, w

    def postprocess(self, outputs, scale, pad_h, pad_w, orig_h, orig_w):
        """Decode YOLOv8 output and apply NMS."""
        # YOLOv8 pose output shape: [1, 56, num_anchors].
        # 56 = 4(box) + 1(conf) + 17*3(keypoints)
        predictions = outputs[0][0]  # [56, num_anchors]
        predictions = predictions.T   # [num_anchors, 56]

        # Filter low-confidence predictions.
        conf = predictions[:, 4]
        mask = conf > self.conf_threshold
        predictions = predictions[mask]

        if len(predictions) == 0:
            return []

        # Decode boxes (cx, cy, w, h) -> (x1, y1, x2, y2).
        boxes = predictions[:, :4].copy()
        kps = predictions[:, 5:]  # [N, 51] = 17 * (x, y, conf)

        cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        # Transform boxes back to the original image coordinates.
        x1 = (x1 - pad_w) / scale
        y1 = (y1 - pad_h) / scale
        x2 = (x2 - pad_w) / scale
        y2 = (y2 - pad_h) / scale

        # Transform keypoints back to the original image coordinates.
        kps_xy = kps.reshape(-1, 17, 3)  # [N, 17, (x, y, conf)]
        kps_xy[:, :, 0] = (kps_xy[:, :, 0] - pad_w) / scale
        kps_xy[:, :, 1] = (kps_xy[:, :, 1] - pad_h) / scale

        # Basic non-maximum suppression.
        scores = predictions[:, 4]
        results = []
        for i in range(len(scores)):
            results.append({
                "bbox": [float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])],
                "score": float(scores[i]),
                "keypoints": kps_xy[i].tolist()  # [17, 3]
            })

        # Sort by confidence.
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def inference(self, image_bgr: np.ndarray):
        """Run inference and return decoded detections with timing."""
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

def draw_pose_opencv(image_bgr: np.ndarray, results: list, show_keypoint_names: bool = False) -> np.ndarray:
    """Draw pose with OpenCV."""
    vis = image_bgr.copy()

    for person in results:
        bbox = person["bbox"]
        score = person["score"]
        keypoints = person["keypoints"]  # [17, 3]

        # Draw bounding boxes.
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(vis, (x1, y1), (x2, y2), COLORS_BGR["bbox"], 2)
        cv2.putText(vis, f"person {score:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS_BGR["text"], 1)

        # Draw skeleton edges.
        for (i, j) in SKELETON:
            kp_i = keypoints[i]
            kp_j = keypoints[j]
            if kp_i[2] > 0.3 and kp_j[2] > 0.3:  # Confidence filter.
                pt_i = (int(kp_i[0]), int(kp_i[1]))
                pt_j = (int(kp_j[0]), int(kp_j[1]))
                cv2.line(vis, pt_i, pt_j, COLORS_BGR["skeleton"], 2)

        # Draw keypoints.
        for idx, kp in enumerate(keypoints):
            if kp[2] > 0.3:
                pt = (int(kp[0]), int(kp[1]))
                cv2.circle(vis, pt, 4, COLORS_BGR["keypoint"], -1)
                if show_keypoint_names:
                    cv2.putText(vis, KEYPOINT_NAMES[idx], pt,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLORS_BGR["text"], 1)

    return vis


def draw_pose_pillow(image_bgr: np.ndarray, results: list) -> np.ndarray:
    """Draw pose with Pillow when OpenCV is unavailable."""
    # BGR -> RGB
    image_rgb = image_bgr[:, :, ::-1]
    img_pil = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(img_pil)

    for person in results:
        bbox = person["bbox"]
        score = person["score"]
        keypoints = person["keypoints"]

        # Bounding box.
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline=(255, 165, 0), width=2)
        draw.text((x1, max(0, y1 - 15)), f"person {score:.2f}", fill=(255, 255, 255))

        # Skeleton edges.
        for (i, j) in SKELETON:
            kp_i = keypoints[i]
            kp_j = keypoints[j]
            if kp_i[2] > 0.3 and kp_j[2] > 0.3:
                draw.line([kp_i[0], kp_i[1], kp_j[0], kp_j[1]], fill=(0, 255, 0), width=2)

        # Keypoints.
        for kp in keypoints:
            if kp[2] > 0.3:
                x, y = kp[0], kp[1]
                draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(255, 0, 0))

    # Convert back to a BGR NumPy array.
    result_rgb = np.array(img_pil)
    result_bgr = result_rgb[:, :, ::-1]
    return result_bgr


def add_info_overlay(image: np.ndarray, fps: float, infer_ms: float,
                     num_persons: int, frame_idx: int) -> np.ndarray:
    """Draw an information overlay."""
    if CV2_AVAILABLE:
        h, w = image.shape[:2]
        info_lines = [
            f"Frame: {frame_idx}",
            f"FPS: {fps:.1f}",
            f"Infer: {infer_ms:.1f}ms",
            f"Persons: {num_persons}",
            "Model: YOLOv8n-Pose (ONNX)",
        ]
        y_start = 20
        for line in info_lines:
            cv2.putText(image, line, (10, y_start),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)  # Black outline.
            cv2.putText(image, line, (10, y_start),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)  # White text.
            y_start += 22
    return image


# =============================================================================
# Main processing flow
# =============================================================================

def _read_video_frames_imageio(input_path: str, max_frames: int = -1):
    """Read video frames with imageio when OpenCV is unavailable."""
    import imageio.v3 as iio
    try:
        props = iio.improps(input_path, plugin="pyav")
        fps = props.fps if hasattr(props, "fps") and props.fps else 30.0
    except Exception:
        fps = 30.0
    reader = iio.imiter(input_path, plugin="pyav")
    frames = []
    for i, frame_rgb in enumerate(reader):
        if max_frames > 0 and i >= max_frames:
            break
        # imageio returns RGB; keep RGB for the Pillow fallback.
        frames.append(np.array(frame_rgb))
        if i % 50 == 0:
            logger.info(f"  Read frame: {i}")
    return frames, fps


def _write_video_imageio(frames_rgb: list, output_path: str, fps: float):
    """Write video with imageio and ffmpeg."""
    import imageio.v3 as iio
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with iio.imopen(output_path, "w", plugin="pyav") as f:
        f.init_video_stream("libx264", fps=fps)
        for frame in frames_rgb:
            f.write_frame(frame)
    logger.info(f"Video written with imageio: {output_path}")


def process_video(
    model: YOLOv8PoseInference,
    input_path: str,
    output_path: str,
    max_frames: int = -1,
    show_window: bool = False,
) -> dict:
    """Process video with OpenCV or fall back to imageio."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Resolve relative paths against the script directory, then the project root.
    resolved_path = input_path
    if not Path(input_path).is_absolute() and not Path(input_path).exists():
        candidate = PROJECT_ROOT / input_path
        if candidate.exists():
            resolved_path = str(candidate)
            logger.info(f"Resolved input path: {resolved_path}")
        else:
            logger.warning(f"Input missing at script and project-root paths: {input_path}")

    # Select a video decoder.
    use_cv2 = False
    if CV2_AVAILABLE:
        cap = cv2.VideoCapture(resolved_path)
        if cap.isOpened():
            use_cv2 = True
        else:
            cap.release()
            logger.warning(f"OpenCV could not open {resolved_path}; trying imageio")
            logger.warning("Possible causes include codec support or an invalid path")

    # OpenCV decoding path.
    if use_cv2:
        fps_orig = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Video: {w}x{h} @ {fps_orig:.1f}fps, {total_frames} frames")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps_orig, (w, h))

        frame_idx = 0
        infer_times = []
        total_persons_detected = 0
        fps_window = []
        t_prev = time.perf_counter()
        logger.info("Starting inference with OpenCV decoding")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames > 0 and frame_idx >= max_frames:
                break

            results, infer_ms = model.inference(frame)
            infer_times.append(infer_ms)
            total_persons_detected += len(results)

            t_now = time.perf_counter()
            fps_window.append(1.0 / max(t_now - t_prev, 1e-6))
            if len(fps_window) > 30:
                fps_window.pop(0)
            fps_display = np.mean(fps_window)
            t_prev = t_now

            vis_frame = draw_pose_opencv(frame, results)
            vis_frame = add_info_overlay(vis_frame, fps_display, infer_ms, len(results), frame_idx)
            writer.write(vis_frame)

            if show_window:
                cv2.imshow("YOLOv8 Pose Demo", vis_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            if frame_idx % 30 == 0:
                logger.info(f"Processed {frame_idx}/{total_frames} frames, "
                            f"recent mean inference: {np.mean(infer_times[-30:]):.1f}ms")

        cap.release()
        writer.release()
        if show_window:
            cv2.destroyAllWindows()

    # imageio fallback when OpenCV cannot decode the video.
    else:
        logger.info("Processing video with imageio")
        if CV2_AVAILABLE:
            logger.info("OpenCV could not decode this file; trying the imageio backend")
        try:
            frames_rgb, fps_orig = _read_video_frames_imageio(resolved_path, max_frames)
        except Exception as e:
            logger.error(f"imageio video read failed: {e}")
            logger.info("Install imageio[pyav] or opencv-python")
            return {}

        if not frames_rgb:
            logger.error("No frames were read")
            return {}

        logger.info(f"Read {len(frames_rgb)} frames @ {fps_orig:.1f}fps")
        logger.info("Starting inference with imageio decoding")

        frame_idx = 0
        infer_times = []
        total_persons_detected = 0
        out_frames_rgb = []

        for frame_rgb in frames_rgb:
            # Legacy fallback passes imageio RGB frames to the inference path.
            # Check color ordering before using this path for accuracy evaluation.
            frame_as_bgr = frame_rgb[:, :, ::-1]  # RGB → BGR

            results, infer_ms = model.inference(frame_as_bgr)
            infer_times.append(infer_ms)
            total_persons_detected += len(results)

            # Draw with Pillow.
            vis_bgr = draw_pose_pillow(frame_as_bgr, results)
            # Add a Pillow information overlay.
            vis_rgb_pil = Image.fromarray(vis_bgr[:, :, ::-1])
            draw_info = ImageDraw.Draw(vis_rgb_pil)
            info_text = (f"Frame:{frame_idx}  Infer:{infer_ms:.0f}ms  "
                         f"Persons:{len(results)}  Backend:imageio")
            draw_info.rectangle([0, 0, vis_rgb_pil.width, 22], fill=(0, 0, 0))
            draw_info.text((5, 4), info_text, fill=(255, 255, 255))
            vis_rgb = np.array(vis_rgb_pil)
            out_frames_rgb.append(vis_rgb)

            frame_idx += 1
            if frame_idx % 30 == 0:
                logger.info(f"Inferred {frame_idx}/{len(frames_rgb)} frames, "
                            f"recent mean inference: {np.mean(infer_times[-30:]):.1f}ms")

        # Write the annotated video.
        try:
            _write_video_imageio(out_frames_rgb, output_path, fps_orig)
        except Exception as e:
            logger.error(f"Video write failed: {e}")
            # Save selected frames as images if video writing fails.
            kf_dir = Path(output_path).parent / "keyframes"
            kf_dir.mkdir(exist_ok=True)
            for i, kf_idx in enumerate(range(0, len(out_frames_rgb), max(1, len(out_frames_rgb) // 10))):
                kf_path = kf_dir / f"keyframe_{kf_idx:05d}.jpg"
                Image.fromarray(out_frames_rgb[kf_idx]).save(str(kf_path))
            logger.info(f"Selected frames saved to: {kf_dir}/")

    # Summary statistics.
    stats = {
        "input": input_path,
        "output": output_path,
        "total_frames": frame_idx,
        "backend": "opencv" if use_cv2 else "imageio",
        "avg_infer_ms": float(np.mean(infer_times)) if infer_times else 0,
        "median_infer_ms": float(np.median(infer_times)) if infer_times else 0,
        "p95_infer_ms": float(np.percentile(infer_times, 95)) if infer_times else 0,
        "avg_fps": 1000.0 / np.mean(infer_times) if infer_times else 0,
        "total_persons_detected": total_persons_detected,
        "avg_persons_per_frame": total_persons_detected / max(frame_idx, 1),
    }

    logger.info("=" * 50)
    logger.info("Inference summary:")
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 50)

    return stats


def process_image(
    model: YOLOv8PoseInference,
    input_path: str,
    output_path: str,
) -> dict:
    """Process one image."""
    if CV2_AVAILABLE:
        frame = cv2.imread(input_path)
        if frame is None:
            logger.error(f"Cannot read image: {input_path}")
            return {}
    else:
        img_pil = Image.open(input_path).convert("RGB")
        frame = np.array(img_pil)[:, :, ::-1]  # RGB -> BGR

    results, infer_ms = model.inference(frame)

    if CV2_AVAILABLE:
        vis_frame = draw_pose_opencv(frame, results)
        vis_frame = add_info_overlay(vis_frame, 0, infer_ms, len(results), 0)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, vis_frame)
    else:
        vis_frame = draw_pose_pillow(frame, results)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(vis_frame[:, :, ::-1]).save(output_path)

    logger.info(f"Image inference: {len(results)} people, {infer_ms:.1f}ms")
    logger.info(f"Result saved: {output_path}")

    return {"infer_ms": infer_ms, "num_persons": len(results)}


def process_camera(
    model: YOLOv8PoseInference,
    camera_id: int = 0,
    output_path: str = None,
) -> None:
    """Run live camera inference."""
    if not CV2_AVAILABLE:
        logger.error("Live camera inference requires opencv-python")
        return

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        logger.error(f"Cannot open camera {camera_id}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Camera resolution: {w}x{h}")

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, 25, (w, h))

    fps_window = []
    frame_idx = 0
    t_prev = time.perf_counter()
    logger.info("Press q to quit live inference")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results, infer_ms = model.inference(frame)

        t_now = time.perf_counter()
        fps_window.append(1.0 / max(t_now - t_prev, 1e-6))
        if len(fps_window) > 10:
            fps_window.pop(0)
        fps_display = np.mean(fps_window)
        t_prev = t_now

        vis_frame = draw_pose_opencv(frame, results)
        vis_frame = add_info_overlay(vis_frame, fps_display, infer_ms, len(results), frame_idx)

        cv2.imshow("YOLOv8 Pose - Real-time", vis_frame)

        if writer:
            writer.write(vis_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


def save_benchmark(stats: dict, benchmark_path: str):
    """Save the legacy benchmark result."""
    import json
    Path(benchmark_path).parent.mkdir(parents=True, exist_ok=True)
    with open(benchmark_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info(f"Benchmark saved: {benchmark_path}")


# =============================================================================
# CLI entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="YOLOv8 pose keypoint inference demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--input", "-i", type=str, help="Input video or image path")
    parser.add_argument("--camera", "-c", type=int, default=None, help="Camera device ID, for example 0")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output path")
    parser.add_argument("--model", "-m", type=str, default=str(DEFAULT_MODEL_PATH),
                        help=f"ONNX model path (default: download to {DEFAULT_MODEL_PATH})")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--max-frames", type=int, default=-1, help="Maximum frames (-1 = all)")
    parser.add_argument("--show", action="store_true", help="Display a window (requires a GUI)")
    parser.add_argument("--download-only", action="store_true", help="Download the model without inference")

    args = parser.parse_args()

    # Download the model if missing.
    model_path = Path(args.model)
    if not download_model(model_path):
        logger.error("Model unavailable; check the connection or provide a local file")
        sys.exit(1)

    if args.download_only:
        logger.info("Model download complete (--download-only)")
        return

    # Initialize inference.
    model = YOLOv8PoseInference(str(model_path), conf_threshold=args.conf)

    # Choose an output path.
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.camera is not None:
        # Live camera.
        output_path = args.output or str(output_dir / "pose_camera_output.mp4")
        process_camera(model, camera_id=args.camera, output_path=output_path)

    elif args.input:
        input_path = args.input
        ext = Path(input_path).suffix.lower()

        if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            # Image mode.
            output_path = args.output or str(output_dir / "pose_image_output.jpg")
            stats = process_image(model, input_path, output_path)
        else:
            # Video mode.
            output_path = args.output or str(output_dir / "pose_video_output.mp4")
            stats = process_video(
                model, input_path, output_path,
                max_frames=args.max_frames,
                show_window=args.show
            )
            # Save the legacy benchmark.
            benchmark_path = str(BENCHMARK_DIR / "pose_benchmark.json")
            save_benchmark(stats, benchmark_path)
    else:
        # Generate a synthetic image when no input is provided.
        logger.info("No input provided; generating a synthetic inference example")
        demo_image = generate_demo_image()
        output_path = args.output or str(output_dir / "pose_synthetic_demo.jpg")
        results, infer_ms = model.inference(demo_image)
        if CV2_AVAILABLE:
            vis = draw_pose_opencv(demo_image, results)
            vis = add_info_overlay(vis, 0, infer_ms, len(results), 0)
            output_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_path, vis)
        else:
            vis = draw_pose_pillow(demo_image, results)
            output_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(vis[:, :, ::-1]).save(output_path)
        logger.info(f"Synthetic test: {infer_ms:.1f}ms, {len(results)} people")
        logger.info(f"Output saved: {output_path}")
        logger.info("")
        logger.info("Use --input <video.mp4> to test recorded video")
        logger.info("Use --camera 0 for live camera inference")


def generate_demo_image():
    """Generate a synthetic image when no recording is available."""
    h, w = 480, 640
    img = np.ones((h, w, 3), dtype=np.uint8) * 200
    # Draw a simple human outline for a smoke test.
    if CV2_AVAILABLE:
        # Head.
        cv2.circle(img, (320, 80), 40, (100, 150, 200), -1)
        # Torso.
        cv2.rectangle(img, (285, 120), (355, 280), (80, 130, 180), -1)
        # Left arm.
        cv2.rectangle(img, (230, 120), (285, 260), (80, 130, 180), -1)
        # Right arm.
        cv2.rectangle(img, (355, 120), (410, 260), (80, 130, 180), -1)
        # Left leg.
        cv2.rectangle(img, (285, 280), (320, 440), (80, 130, 180), -1)
        # Right leg.
        cv2.rectangle(img, (320, 280), (355, 440), (80, 130, 180), -1)
        # Add a label.
        cv2.putText(img, "SYNTHETIC TEST IMAGE", (100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, "Use --input for real video", (120, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    return img


if __name__ == "__main__":
    main()
