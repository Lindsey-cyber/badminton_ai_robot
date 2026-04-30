#!/usr/bin/env python3
"""
demo_badminton_detection.py
===========================
羽毛球/球体检测 + 轨迹追踪 + 速度估计 Demo

功能：
  1. YOLOv8n ONNX 检测运动球体（可用预训练模型，class="sports ball"）
  2. 卡尔曼滤波轨迹平滑
  3. 单目像素速度估计 → 物理速度近似换算
  4. 轨迹绘制（历史轨迹 + 预测方向）
  5. 输出带标注的视频

用法：
    python demo_badminton_detection.py --input assets/demo_inputs/badminton_sample.mp4
    python demo_badminton_detection.py --camera 0

注意：
  - 当前使用 YOLOv8n 预训练模型（COCO 类别 class 32 = sports ball）
  - 真实场景中应微调专用羽毛球模型（参见 training_and_finetune_plan.md）
  - 高速羽毛球建议使用 TrackNetV3 替代
"""

import argparse
import json
import logging
import sys
import time
import urllib.request
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
    print("[ERROR] 请安装: pip install onnxruntime")
    sys.exit(1)

from PIL import Image, ImageDraw

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# 配置
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODEL_DIR = PROJECT_ROOT / "src" / "perception"
DEFAULT_MODEL_PATH = MODEL_DIR / "yolov8n.onnx"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "demo_videos"
BENCHMARK_DIR = PROJECT_ROOT / "outputs" / "benchmarks"

# YOLOv8n ONNX 下载地址
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.onnx"

# COCO 类别中与球相关的 class ID
# 32: sports ball
# 若使用专项羽毛球模型，class ID = 0 (shuttlecock)
BALL_CLASS_IDS = {32: "sports ball", 0: "shuttlecock"}
TARGET_CLASS_IDS = [32, 0]  # 优先检测这些类别

# 轨迹颜色（从淡到深，表示历史→现在）
TRAJECTORY_COLORS = [
    (0, 255, 255),   # 黄绿
    (0, 200, 200),
    (0, 150, 200),
    (0, 100, 200),
    (0, 50, 255),    # 蓝色
]

MAX_TRAJECTORY_LEN = 30  # 最多保留 30 帧历史轨迹


# =============================================================================
# 模型下载
# =============================================================================

def download_model(model_path: Path) -> bool:
    if model_path.exists():
        logger.info(f"模型已存在: {model_path}")
        return True

    model_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"正在下载 YOLOv8n 模型: {MODEL_URL}")

    try:
        def reporthook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(downloaded * 100 / total_size, 100)
                print(f"\r下载进度: {pct:.1f}%", end="", flush=True)

        urllib.request.urlretrieve(MODEL_URL, str(model_path), reporthook)
        print()
        logger.info("模型下载完成！")
        return True
    except Exception as e:
        logger.error(f"下载失败: {e}")
        return False


# =============================================================================
# YOLOv8 检测器
# =============================================================================

class YOLOv8Detector:
    """YOLOv8 通用目标检测器（ONNX）"""

    INPUT_SIZE = 640

    def __init__(self, model_path: str, conf_threshold: float = 0.20, target_classes: list = None):
        self.conf_threshold = conf_threshold
        self.target_classes = target_classes  # None = 检测所有类别

        logger.info(f"加载检测模型: {model_path}")
        providers = ort.get_available_providers()
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        logger.info("检测模型加载成功！")

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
        # YOLOv8 输出: [1, 84, num_anchors]  (4 box + 80 classes)
        # 或自定义类别数
        preds = outputs[0][0].T  # [num_anchors, 84]

        # 获取最高类别分数
        class_scores = preds[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        max_scores = class_scores[np.arange(len(class_ids)), class_ids]
        obj_conf = max_scores  # YOLOv8 不再有单独 objectness

        # 过滤
        mask = obj_conf > self.conf_threshold
        if self.target_classes is not None:
            class_mask = np.isin(class_ids, self.target_classes)
            mask = mask & class_mask

        preds_filtered = preds[mask]
        class_ids_filtered = class_ids[mask]
        scores_filtered = obj_conf[mask]

        if len(preds_filtered) == 0:
            return []

        # 解码 box
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
        return results

    def inference(self, image_bgr: np.ndarray):
        img_tensor, scale, pad_h, pad_w, h, w = self.preprocess(image_bgr)
        t0 = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_name: img_tensor})
        t1 = time.perf_counter()
        infer_ms = (t1 - t0) * 1000
        results = self.postprocess(outputs, scale, pad_h, pad_w, h, w)
        return results, infer_ms


# =============================================================================
# 卡尔曼滤波器（球的轨迹平滑）
# =============================================================================

class BallKalmanFilter:
    """2D 卡尔曼滤波，状态: [x, y, vx, vy]"""

    def __init__(self):
        self.initialized = False
        self.state = np.zeros(4)      # [x, y, vx, vy]
        self.P = np.eye(4) * 100      # 协方差
        self.Q = np.eye(4) * 1        # 过程噪声
        self.R = np.eye(2) * 10       # 观测噪声
        self.F = np.array([           # 状态转移矩阵
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)
        self.H = np.array([           # 观测矩阵
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)

    def update(self, measurement: tuple):
        z = np.array(measurement, dtype=float)
        if not self.initialized:
            self.state[:2] = z
            self.initialized = True
            return self.state[:2]

        # 预测
        x_pred = self.F @ self.state
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # 更新
        y = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        self.state = x_pred + K @ y
        self.P = (np.eye(4) - K @ self.H) @ P_pred

        return self.state[:2]

    def predict_next(self, n_steps: int = 5):
        """预测未来 n_steps 步的位置"""
        predicted = []
        state = self.state.copy()
        for _ in range(n_steps):
            state = self.F @ state
            predicted.append(state[:2].copy())
        return predicted


# =============================================================================
# 速度估计器
# =============================================================================

class SpeedEstimator:
    """基于像素位移的单目速度估计"""

    def __init__(self, fps: float, pixels_per_meter: float = None):
        """
        fps: 视频帧率
        pixels_per_meter: 像素/米标定值（None = 仅输出像素速度）
        """
        self.fps = fps
        self.pixels_per_meter = pixels_per_meter
        self.prev_pos = None
        self.speeds = deque(maxlen=10)  # 滑动窗口平均

    def calibrate_from_court(self, court_width_px: float, court_width_m: float = 6.1):
        """从球场宽度标定像素/米比例"""
        self.pixels_per_meter = court_width_px / court_width_m
        logger.info(f"标定完成: {self.pixels_per_meter:.1f} pixels/meter")

    def update(self, position: tuple) -> dict:
        """更新位置，返回速度信息"""
        if self.prev_pos is None:
            self.prev_pos = position
            return {"pixel_speed": 0, "speed_kmh": None, "speed_ms": None}

        dx = position[0] - self.prev_pos[0]
        dy = position[1] - self.prev_pos[1]
        pixel_dist = np.sqrt(dx**2 + dy**2)
        pixel_speed = pixel_dist * self.fps  # 像素/秒

        self.prev_pos = position
        self.speeds.append(pixel_speed)
        avg_pixel_speed = np.mean(self.speeds)

        result = {"pixel_speed": avg_pixel_speed}
        if self.pixels_per_meter:
            speed_ms = avg_pixel_speed / self.pixels_per_meter
            result["speed_ms"] = speed_ms
            result["speed_kmh"] = speed_ms * 3.6
        else:
            result["speed_ms"] = None
            result["speed_kmh"] = None

        return result


# =============================================================================
# 可视化
# =============================================================================

def draw_detections(image: np.ndarray, detections: list, trajectory: deque,
                    speed_info: dict, predictions: list = None) -> np.ndarray:
    """绘制检测结果、轨迹、速度"""
    vis = image.copy()

    # 绘制历史轨迹
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

    # 绘制预测轨迹
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

    # 绘制检测框
    for det in detections:
        bbox = det["bbox"]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cx, cy = int(det["center"][0]), int(det["center"][1])

        if CV2_AVAILABLE:
            # 边界框
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
            # 中心点
            cv2.circle(vis, (cx, cy), 6, (0, 255, 255), -1)
            # 标签
            label = f"{det['class_name']} {det['score']:.2f}"
            cv2.putText(vis, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # 绘制速度信息
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
# 视频处理
# =============================================================================

def process_video(
    detector: YOLOv8Detector,
    input_path: str,
    output_path: str,
    max_frames: int = -1,
    pixels_per_meter: float = None,
) -> dict:
    if not CV2_AVAILABLE:
        logger.error("视频处理需要 opencv-python")
        return {}

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error(f"无法打开: {input_path}")
        return {}

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"视频: {w}x{h} @ {fps:.1f}fps, 共 {total} 帧")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # 初始化
    kf = BallKalmanFilter()
    speed_est = SpeedEstimator(fps=fps, pixels_per_meter=pixels_per_meter)
    trajectory = deque(maxlen=MAX_TRAJECTORY_LEN)

    # 统计
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

        # 检测
        results, infer_ms = detector.inference(frame)
        infer_times.append(infer_ms)

        # 选取最佳候选球（最高置信度）
        best_det = None
        if results:
            best_det = results[0]
            detected_frames += 1
            cx, cy = best_det["center"]

            # 卡尔曼更新
            smooth_pos = kf.update((cx, cy))
            trajectory.append(smooth_pos)

            # 速度估计
            speed_info = speed_est.update(smooth_pos)
            if speed_info.get("speed_kmh") is not None:
                speed_samples.append(speed_info["speed_kmh"])
        else:
            # 仅预测（无检测）
            if kf.initialized:
                pred_pos = kf.F @ kf.state
                kf.state = pred_pos
                # 不添加到轨迹（仅内部预测）
            speed_info = {"pixel_speed": 0, "speed_kmh": None, "speed_ms": None}

        # 预测未来轨迹
        future_preds = kf.predict_next(5) if kf.initialized else []

        # 计算 FPS
        t_now = time.perf_counter()
        fps_window.append(1.0 / max(t_now - t_prev, 1e-6))
        if len(fps_window) > 30:
            fps_window.pop(0)
        fps_display = np.mean(fps_window)
        t_prev = t_now

        # 可视化
        vis = draw_detections(frame, results if best_det else [], trajectory,
                              speed_info if best_det else {}, future_preds)
        vis = add_info_overlay(vis, fps_display, infer_ms, len(results), frame_idx)

        writer.write(vis)
        frame_idx += 1

        if frame_idx % 50 == 0:
            detect_rate = detected_frames / max(frame_idx, 1) * 100
            logger.info(f"帧 {frame_idx}/{total}, 检测率: {detect_rate:.1f}%, "
                        f"推理: {np.mean(infer_times[-30:]):.1f}ms")

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
    logger.info("球体检测结果汇总:")
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 50)
    logger.info("⚠️  注意：当前使用通用 YOLOv8n，对羽毛球检测率有限")
    logger.info("   建议：微调专项羽毛球模型或使用 TrackNetV3")

    return stats


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="羽毛球/球体检测 + 轨迹追踪 Demo")
    parser.add_argument("--input", "-i", type=str, help="输入视频路径")
    parser.add_argument("--camera", "-c", type=int, default=None)
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--model", "-m", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--ppm", type=float, default=None,
                        help="像素/米标定值（pixels per meter），用于速度估计")
    parser.add_argument("--all-classes", action="store_true",
                        help="检测所有类别（而非仅球类）")
    args = parser.parse_args()

    # 下载模型
    model_path = Path(args.model)
    if not download_model(model_path):
        sys.exit(1)

    # 目标类别
    target_classes = None if args.all_classes else TARGET_CLASS_IDS
    detector = YOLOv8Detector(str(model_path), conf_threshold=args.conf,
                               target_classes=target_classes)

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.camera is not None:
        # 实时摄像头（简化版，不记录统计）
        if not CV2_AVAILABLE:
            logger.error("需要 opencv-python")
            return

        cap = cv2.VideoCapture(args.camera)
        kf = BallKalmanFilter()
        trajectory = deque(maxlen=MAX_TRAJECTORY_LEN)
        fps = 30
        speed_est = SpeedEstimator(fps=fps, pixels_per_meter=args.ppm)
        frame_idx = 0

        logger.info("按 q 退出...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results, infer_ms = detector.inference(frame)
            if results:
                cx, cy = results[0]["center"]
                pos = kf.update((cx, cy))
                trajectory.append(pos)
                speed_info = speed_est.update(pos)
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
        # 保存 benchmark
        bench_path = str(BENCHMARK_DIR / "ball_detection_benchmark.json")
        BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
        with open(bench_path, "w") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        logger.info(f"Benchmark 已保存: {bench_path}")
    else:
        logger.info("请指定 --input 视频路径 或 --camera 摄像头 ID")
        logger.info("示例: python demo_badminton_detection.py --input video.mp4")
        logger.info("")
        logger.info("TrackNetV3（专业羽毛球追踪）:")
        logger.info("  git clone https://github.com/qaz812345/TrackNetV3")


if __name__ == "__main__":
    main()
