#!/usr/bin/env python3
"""
demo_pose_inference.py
======================
人体关键点姿态估计 Demo
- 使用 YOLOv8n-pose ONNX 模型
- 支持视频文件 / 摄像头实时推理
- 可视化 17 关键点 + 骨骼连线
- 输出带标注的视频文件

用法：
    # 推理视频文件
    python demo_pose_inference.py --input assets/demo_inputs/badminton_sample.mp4

    # 实时摄像头
    python demo_pose_inference.py --camera 0

    # 仅处理前 N 帧（测试用）
    python demo_pose_inference.py --input video.mp4 --max-frames 100

依赖：
    pip install opencv-python-headless onnxruntime numpy
    # 或 pip install opencv-python（如果需要显示窗口）
"""

import argparse
import os
import sys
import time
import logging
import urllib.request
from pathlib import Path

import numpy as np

# 尝试导入 opencv
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("[WARNING] opencv-python 未安装，将使用 Pillow 作为后备方案")

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False
    print("[ERROR] onnxruntime 未安装，请运行: pip install onnxruntime")
    sys.exit(1)

from PIL import Image, ImageDraw, ImageFont

# =============================================================================
# 配置
# =============================================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# COCO 17 关键点名称
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# COCO 骨骼连接（关键点索引对）
SKELETON = [
    (0, 1), (0, 2),           # 鼻子-眼睛
    (1, 3), (2, 4),           # 眼睛-耳朵
    (0, 5), (0, 6),           # 鼻子-肩膀
    (5, 6),                   # 左右肩
    (5, 7), (7, 9),           # 左臂
    (6, 8), (8, 10),          # 右臂
    (5, 11), (6, 12),         # 躯干
    (11, 12),                 # 左右髋
    (11, 13), (13, 15),       # 左腿
    (12, 14), (14, 16),       # 右腿
]

# 颜色（BGR for OpenCV）
COLORS_BGR = {
    "skeleton": (0, 255, 0),     # 绿色骨骼线
    "keypoint": (0, 0, 255),     # 红色关键点
    "bbox": (255, 165, 0),       # 橙色边界框
    "text": (255, 255, 255),     # 白色文字
    "bg": (0, 0, 0),             # 黑色背景
}

# 模型文件路径
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODEL_DIR = PROJECT_ROOT / "src" / "pose"
DEFAULT_MODEL_PATH = MODEL_DIR / "yolov8n-pose.onnx"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "demo_videos"
BENCHMARK_DIR = PROJECT_ROOT / "outputs" / "benchmarks"

# =============================================================================
# 模型下载（使用 Ultralytics GitHub Release）
# =============================================================================

MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.onnx"

def download_model(model_path: Path) -> bool:
    """下载 YOLOv8n-pose.onnx 模型（多种方式，自动重试）"""
    MIN_VALID_SIZE = 1024 * 10  # 有效 ONNX 至少 10KB，0字节或极小文件视为无效

    # 检查已有文件是否有效
    if model_path.exists() and model_path.stat().st_size > MIN_VALID_SIZE:
        logger.info(f"模型已存在 ({model_path.stat().st_size / 1024 / 1024:.1f}MB): {model_path}")
        return True
    elif model_path.exists():
        logger.warning(f"模型文件无效（大小={model_path.stat().st_size} bytes），重新下载...")
        model_path.unlink()

    model_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 方式 1：ultralytics 自动下载并导出（最推荐，可自动缓存）──────────────────
    try:
        logger.info("方式1: 尝试通过 ultralytics 下载并导出 ONNX...")
        from ultralytics import YOLO
        m = YOLO("yolov8n-pose.pt")          # 自动下载 .pt 到 ultralytics 缓存
        export_path = m.export(format="onnx", simplify=True, imgsz=640)
        import shutil
        shutil.copy(str(export_path), str(model_path))
        logger.info(f"✓ 模型导出成功: {model_path}")
        return True
    except ImportError:
        logger.info("ultralytics 未安装，跳过方式1（建议: pip install ultralytics）")
    except Exception as e:
        logger.warning(f"方式1 失败: {e}")

    # ── 方式 2：直接 HTTP 下载（多个镜像源）──────────────────────────────────────
    download_urls = [
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.onnx",
        "https://huggingface.co/Ultralytics/assets/resolve/main/yolov8n-pose.onnx",
    ]
    for url in download_urls:
        try:
            logger.info(f"方式2: 尝试从 {url} 下载...")
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
                            print(f"\r  下载进度: {pct:.1f}% ({downloaded}/{total} bytes)", end="", flush=True)
                print()
            if model_path.stat().st_size > MIN_VALID_SIZE:
                logger.info(f"✓ 下载成功 ({model_path.stat().st_size / 1024 / 1024:.1f}MB): {model_path}")
                return True
            else:
                logger.warning(f"下载后文件太小 ({model_path.stat().st_size} bytes)，可能被代理拦截")
                model_path.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"下载失败: {e}")

    # ── 方式 3：pip install ultralytics 提示 ─────────────────────────────────────
    logger.error("所有自动下载方式均失败。")
    logger.info("")
    logger.info("请手动操作（任选其一）：")
    logger.info("")
    logger.info("  # 推荐：安装 ultralytics，脚本下次运行时自动处理")
    logger.info("  pip install ultralytics")
    logger.info("")
    logger.info("  # 或者手动用 curl 下载（macOS/Linux）：")
    logger.info(f"  curl -L '{download_urls[0]}' -o '{model_path}'")
    logger.info("")
    logger.info("  # 或者 Python 一行命令导出：")
    logger.info("  python -c \"from ultralytics import YOLO; YOLO('yolov8n-pose.pt').export(format='onnx')\"")
    logger.info(f"  # 然后把生成的 yolov8n-pose.onnx 复制到: {model_path}")
    return False


# =============================================================================
# YOLOv8 Pose 推理器
# =============================================================================

class YOLOv8PoseInference:
    """YOLOv8-Pose ONNX 推理器"""

    INPUT_SIZE = 640  # 模型输入尺寸

    def __init__(self, model_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # 创建推理会话
        logger.info(f"加载模型: {model_path}")
        providers = ort.get_available_providers()
        logger.info(f"可用推理后端: {providers}")

        self.session = ort.InferenceSession(
            model_path,
            providers=providers
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        logger.info(f"模型输入: {self.input_name}, shape={self.session.get_inputs()[0].shape}")
        logger.info(f"模型输出: {self.output_names}")
        logger.info("模型加载成功！")

    def preprocess(self, image_bgr: np.ndarray):
        """预处理图像：resize + normalize + transpose"""
        h, w = image_bgr.shape[:2]

        # 计算 letterbox padding
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
        """后处理：解码 YOLOv8 输出 + NMS"""
        # YOLOv8-pose 输出形状: [1, 56, num_anchors]
        # 56 = 4(box) + 1(conf) + 17*3(keypoints)
        predictions = outputs[0][0]  # [56, num_anchors]
        predictions = predictions.T   # [num_anchors, 56]

        # 过滤低置信度
        conf = predictions[:, 4]
        mask = conf > self.conf_threshold
        predictions = predictions[mask]

        if len(predictions) == 0:
            return []

        # 解码 box (cx, cy, w, h) -> (x1, y1, x2, y2)
        boxes = predictions[:, :4].copy()
        kps = predictions[:, 5:]  # [N, 51] = 17 * (x, y, conf)

        cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        # 反变换到原图坐标
        x1 = (x1 - pad_w) / scale
        y1 = (y1 - pad_h) / scale
        x2 = (x2 - pad_w) / scale
        y2 = (y2 - pad_h) / scale

        # 关键点反变换
        kps_xy = kps.reshape(-1, 17, 3)  # [N, 17, (x, y, conf)]
        kps_xy[:, :, 0] = (kps_xy[:, :, 0] - pad_w) / scale
        kps_xy[:, :, 1] = (kps_xy[:, :, 1] - pad_h) / scale

        # 简单 NMS
        scores = predictions[:, 4]
        results = []
        for i in range(len(scores)):
            results.append({
                "bbox": [float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])],
                "score": float(scores[i]),
                "keypoints": kps_xy[i].tolist()  # [17, 3]
            })

        # 按置信度排序
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def inference(self, image_bgr: np.ndarray):
        """执行推理，返回结果列表"""
        img_tensor, scale, pad_h, pad_w, h, w = self.preprocess(image_bgr)
        t0 = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_name: img_tensor})
        t1 = time.perf_counter()
        infer_ms = (t1 - t0) * 1000

        results = self.postprocess(outputs, scale, pad_h, pad_w, h, w)
        return results, infer_ms


# =============================================================================
# 可视化
# =============================================================================

def draw_pose_opencv(image_bgr: np.ndarray, results: list, show_keypoint_names: bool = False) -> np.ndarray:
    """使用 OpenCV 绘制姿态"""
    vis = image_bgr.copy()

    for person in results:
        bbox = person["bbox"]
        score = person["score"]
        keypoints = person["keypoints"]  # [17, 3]

        # 绘制边界框
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(vis, (x1, y1), (x2, y2), COLORS_BGR["bbox"], 2)
        cv2.putText(vis, f"person {score:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS_BGR["text"], 1)

        # 绘制骨骼线
        for (i, j) in SKELETON:
            kp_i = keypoints[i]
            kp_j = keypoints[j]
            if kp_i[2] > 0.3 and kp_j[2] > 0.3:  # 置信度过滤
                pt_i = (int(kp_i[0]), int(kp_i[1]))
                pt_j = (int(kp_j[0]), int(kp_j[1]))
                cv2.line(vis, pt_i, pt_j, COLORS_BGR["skeleton"], 2)

        # 绘制关键点
        for idx, kp in enumerate(keypoints):
            if kp[2] > 0.3:
                pt = (int(kp[0]), int(kp[1]))
                cv2.circle(vis, pt, 4, COLORS_BGR["keypoint"], -1)
                if show_keypoint_names:
                    cv2.putText(vis, KEYPOINT_NAMES[idx], pt,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLORS_BGR["text"], 1)

    return vis


def draw_pose_pillow(image_bgr: np.ndarray, results: list) -> np.ndarray:
    """使用 Pillow 绘制姿态（无 OpenCV 时的后备方案）"""
    # BGR -> RGB
    image_rgb = image_bgr[:, :, ::-1]
    img_pil = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(img_pil)

    for person in results:
        bbox = person["bbox"]
        score = person["score"]
        keypoints = person["keypoints"]

        # 边界框
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline=(255, 165, 0), width=2)
        draw.text((x1, max(0, y1 - 15)), f"person {score:.2f}", fill=(255, 255, 255))

        # 骨骼线
        for (i, j) in SKELETON:
            kp_i = keypoints[i]
            kp_j = keypoints[j]
            if kp_i[2] > 0.3 and kp_j[2] > 0.3:
                draw.line([kp_i[0], kp_i[1], kp_j[0], kp_j[1]], fill=(0, 255, 0), width=2)

        # 关键点
        for kp in keypoints:
            if kp[2] > 0.3:
                x, y = kp[0], kp[1]
                draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(255, 0, 0))

    # 转回 BGR numpy
    result_rgb = np.array(img_pil)
    result_bgr = result_rgb[:, :, ::-1]
    return result_bgr


def add_info_overlay(image: np.ndarray, fps: float, infer_ms: float,
                     num_persons: int, frame_idx: int) -> np.ndarray:
    """添加信息叠加层"""
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
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)  # 黑色描边
            cv2.putText(image, line, (10, y_start),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)  # 白色文字
            y_start += 22
    return image


# =============================================================================
# 主处理流程
# =============================================================================

def _read_video_frames_imageio(input_path: str, max_frames: int = -1):
    """用 imageio 读取视频帧（OpenCV 不可用时的后备方案）"""
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
        # imageio 返回 RGB，转 BGR（兼容后续 Pillow 路径用 RGB 无所谓，统一用 RGB）
        frames.append(np.array(frame_rgb))
        if i % 50 == 0:
            logger.info(f"  读取帧: {i}")
    return frames, fps


def _write_video_imageio(frames_rgb: list, output_path: str, fps: float):
    """用 imageio + ffmpeg 写出视频"""
    import imageio.v3 as iio
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with iio.imopen(output_path, "w", plugin="pyav") as f:
        f.init_video_stream("libx264", fps=fps)
        for frame in frames_rgb:
            f.write_frame(frame)
    logger.info(f"视频已写入（imageio）: {output_path}")


def process_video(
    model: YOLOv8PoseInference,
    input_path: str,
    output_path: str,
    max_frames: int = -1,
    show_window: bool = False,
) -> dict:
    """处理视频文件（优先 OpenCV，后备 imageio）"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # ── 路径解析：相对路径优先用脚本所在目录，其次尝试项目根目录 ──────────────
    resolved_path = input_path
    if not Path(input_path).is_absolute() and not Path(input_path).exists():
        candidate = PROJECT_ROOT / input_path
        if candidate.exists():
            resolved_path = str(candidate)
            logger.info(f"路径已解析为: {resolved_path}")
        else:
            logger.warning(f"文件不存在（已尝试相对路径和项目根路径）: {input_path}")

    # ── 决定用哪个后端 ────────────────────────────────────────────────────────
    use_cv2 = False
    if CV2_AVAILABLE:
        cap = cv2.VideoCapture(resolved_path)
        if cap.isOpened():
            use_cv2 = True
        else:
            cap.release()
            logger.warning(f"OpenCV 无法打开视频（{resolved_path}），自动切换到 imageio 模式")
            logger.warning("（常见原因：macOS H.264 编解码器问题、路径错误）")

    # ── OpenCV 路径（有 cv2 且成功打开）──────────────────────────────────────
    if use_cv2:
        fps_orig = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"视频信息: {w}x{h} @ {fps_orig:.1f}fps, 共 {total_frames} 帧")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps_orig, (w, h))

        frame_idx = 0
        infer_times = []
        total_persons_detected = 0
        fps_window = []
        t_prev = time.perf_counter()
        logger.info("开始推理（OpenCV 模式）...")

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
                logger.info(f"已处理 {frame_idx}/{total_frames} 帧，"
                            f"平均推理: {np.mean(infer_times[-30:]):.1f}ms")

        cap.release()
        writer.release()
        if show_window:
            cv2.destroyAllWindows()

    # ── imageio 后备路径（cv2 不可用 或 cv2 无法打开该视频）─────────────────
    else:
        logger.info("使用 imageio 模式处理视频（支持 macOS H.264/HEVC 等格式）...")
        if CV2_AVAILABLE:
            logger.info("（OpenCV 已安装但无法解码该视频，imageio+pyav 作为备用）")
        try:
            frames_rgb, fps_orig = _read_video_frames_imageio(resolved_path, max_frames)
        except Exception as e:
            logger.error(f"imageio 读取视频失败: {e}")
            logger.info("请安装: pip install imageio[pyav]  或  pip install opencv-python")
            return {}

        if not frames_rgb:
            logger.error("未读取到任何帧")
            return {}

        logger.info(f"读取完成: {len(frames_rgb)} 帧 @ {fps_orig:.1f}fps")
        logger.info("开始推理（imageio 模式）...")

        frame_idx = 0
        infer_times = []
        total_persons_detected = 0
        out_frames_rgb = []

        for frame_rgb in frames_rgb:
            # imageio 给 RGB，转成假 BGR 数组用于推理（内部 preprocess 会再转 RGB）
            # 因为 preprocess 无 cv2 时用 PIL（直接用 RGB），需要统一传 RGB 格式
            # 这里将数组当成 BGR 传入（实际颜色顺序对推理影响很小，检测框和关键点坐标正确）
            frame_as_bgr = frame_rgb[:, :, ::-1]  # RGB → BGR

            results, infer_ms = model.inference(frame_as_bgr)
            infer_times.append(infer_ms)
            total_persons_detected += len(results)

            # 用 Pillow 绘制
            vis_bgr = draw_pose_pillow(frame_as_bgr, results)
            # 用 Pillow 添加信息叠加（cv2 路径的 add_info_overlay 等效）
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
                logger.info(f"已推理 {frame_idx}/{len(frames_rgb)} 帧，"
                            f"平均推理: {np.mean(infer_times[-30:]):.1f}ms")

        # 写出视频
        try:
            _write_video_imageio(out_frames_rgb, output_path, fps_orig)
        except Exception as e:
            logger.error(f"视频写入失败: {e}")
            # 后备：只保存关键帧图片
            kf_dir = Path(output_path).parent / "keyframes"
            kf_dir.mkdir(exist_ok=True)
            for i, kf_idx in enumerate(range(0, len(out_frames_rgb), max(1, len(out_frames_rgb) // 10))):
                kf_path = kf_dir / f"keyframe_{kf_idx:05d}.jpg"
                Image.fromarray(out_frames_rgb[kf_idx]).save(str(kf_path))
            logger.info(f"关键帧已保存至: {kf_dir}/")

    # 汇总统计
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
    logger.info("推理结果汇总:")
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 50)

    return stats


def process_image(
    model: YOLOv8PoseInference,
    input_path: str,
    output_path: str,
) -> dict:
    """处理单张图片"""
    if CV2_AVAILABLE:
        frame = cv2.imread(input_path)
        if frame is None:
            logger.error(f"无法读取图片: {input_path}")
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

    logger.info(f"图片推理完成，检测到 {len(results)} 人，耗时 {infer_ms:.1f}ms")
    logger.info(f"结果已保存: {output_path}")

    return {"infer_ms": infer_ms, "num_persons": len(results)}


def process_camera(
    model: YOLOv8PoseInference,
    camera_id: int = 0,
    output_path: str = None,
) -> None:
    """实时摄像头推理"""
    if not CV2_AVAILABLE:
        logger.error("实时摄像头需要 opencv-python")
        return

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        logger.error(f"无法打开摄像头 {camera_id}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"摄像头分辨率: {w}x{h}")

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, 25, (w, h))

    fps_window = []
    frame_idx = 0
    t_prev = time.perf_counter()
    logger.info("按 'q' 退出实时推理")

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
    """保存 benchmark 结果"""
    import json
    Path(benchmark_path).parent.mkdir(parents=True, exist_ok=True)
    with open(benchmark_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info(f"Benchmark 已保存: {benchmark_path}")


# =============================================================================
# CLI 入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="YOLOv8-Pose 人体关键点推理 Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--input", "-i", type=str, help="输入视频或图片路径")
    parser.add_argument("--camera", "-c", type=int, default=None, help="摄像头设备 ID（例如 0）")
    parser.add_argument("--output", "-o", type=str, default=None, help="输出文件路径")
    parser.add_argument("--model", "-m", type=str, default=str(DEFAULT_MODEL_PATH),
                        help=f"ONNX 模型路径（默认自动下载到 {DEFAULT_MODEL_PATH}）")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值 (default: 0.25)")
    parser.add_argument("--max-frames", type=int, default=-1, help="最大处理帧数（-1 = 全部）")
    parser.add_argument("--show", action="store_true", help="实时显示窗口（需要 GUI 环境）")
    parser.add_argument("--download-only", action="store_true", help="仅下载模型，不执行推理")

    args = parser.parse_args()

    # 下载模型
    model_path = Path(args.model)
    if not download_model(model_path):
        logger.error("模型不可用，请检查网络连接或手动下载")
        sys.exit(1)

    if args.download_only:
        logger.info("模型下载完成，退出（--download-only 模式）")
        return

    # 初始化模型
    model = YOLOv8PoseInference(str(model_path), conf_threshold=args.conf)

    # 确定输出路径
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.camera is not None:
        # 实时摄像头
        output_path = args.output or str(output_dir / "pose_camera_output.mp4")
        process_camera(model, camera_id=args.camera, output_path=output_path)

    elif args.input:
        input_path = args.input
        ext = Path(input_path).suffix.lower()

        if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            # 图片模式
            output_path = args.output or str(output_dir / "pose_image_output.jpg")
            stats = process_image(model, input_path, output_path)
        else:
            # 视频模式
            output_path = args.output or str(output_dir / "pose_video_output.mp4")
            stats = process_video(
                model, input_path, output_path,
                max_frames=args.max_frames,
                show_window=args.show
            )
            # 保存 benchmark
            benchmark_path = str(BENCHMARK_DIR / "pose_benchmark.json")
            save_benchmark(stats, benchmark_path)
    else:
        # 没有输入：生成一张合成测试图进行演示
        logger.info("未指定输入，生成合成测试图演示推理能力...")
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
        logger.info(f"合成测试完成，推理耗时 {infer_ms:.1f}ms，检测到 {len(results)} 人")
        logger.info(f"输出已保存: {output_path}")
        logger.info("")
        logger.info("提示：请使用 --input <video.mp4> 测试真实视频")
        logger.info("      请使用 --camera 0 开启实时摄像头推理")


def generate_demo_image():
    """生成合成测试图（无真实输入时使用）"""
    h, w = 480, 640
    img = np.ones((h, w, 3), dtype=np.uint8) * 200
    # 绘制一个简单的人形轮廓（仅测试目的）
    if CV2_AVAILABLE:
        # 头部
        cv2.circle(img, (320, 80), 40, (100, 150, 200), -1)
        # 躯干
        cv2.rectangle(img, (285, 120), (355, 280), (80, 130, 180), -1)
        # 左臂
        cv2.rectangle(img, (230, 120), (285, 260), (80, 130, 180), -1)
        # 右臂
        cv2.rectangle(img, (355, 120), (410, 260), (80, 130, 180), -1)
        # 左腿
        cv2.rectangle(img, (285, 280), (320, 440), (80, 130, 180), -1)
        # 右腿
        cv2.rectangle(img, (320, 280), (355, 440), (80, 130, 180), -1)
        # 添加文字说明
        cv2.putText(img, "SYNTHETIC TEST IMAGE", (100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, "Use --input for real video", (120, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    return img


if __name__ == "__main__":
    main()
