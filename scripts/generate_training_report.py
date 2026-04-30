#!/usr/bin/env python3
"""
generate_training_report.py
============================
羽毛球训练分析报告生成器

分析项目：
  · 挥臂速度   — 腕部关键点帧间位移 → m/s / km/h
  · 球速检测   — 帧差运动检测小目标 + 轨迹速度；无法检测时用挥臂×物理系数估算
  · 击球手势   — 左/右腕峰值速度比较 → 主导手；发球/回球按场地位置区分
  · 击球高度   — 峰值帧腕部相对脚踝的实际高度（米）
  · 站位热图   — 踝部投影到标准球场坐标系（13.40m × 6.10m）

输出：单文件自包含 HTML 报告（含 Chart.js 图表 + SVG 球场热图）
      可在浏览器直接 Ctrl+P → 打印为 PDF

标准球场尺寸（双打）：
  长 13.40m × 宽 6.10m，网中心高 1.55m
  短发球线距网 1.98m，双打长发球线距底线 0.76m
  单打边线距双打边线 0.46m

用法：
    # 直接分析视频
    python scripts/generate_training_report.py --input assets/demo_inputs/badminton_sample.mp4

    # 生成演示报告（无需模型/视频）
    python scripts/generate_training_report.py --demo

    # 手动提供球场角点（提高站位精度），顺序：左上 右上 右下 左下（像素）
    python scripts/generate_training_report.py --input video.mp4 \\
        --court-corners "100,50 620,50 620,1230 100,1230"

    # 只处理前 N 帧（快速测试）
    python scripts/generate_training_report.py --input video.mp4 --max-frames 90
"""

import argparse
import json
import sys
import time
import math
import logging
import random
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw

# ── 可选依赖 ──────────────────────────────────────────────────────────────────
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
    print("[WARNING] onnxruntime 未安装，将使用演示模式")

# ── 路径配置 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODEL_PATH  = PROJECT_ROOT / "src" / "pose" / "yolov8n-pose.onnx"
OUTPUT_DIR  = PROJECT_ROOT / "outputs" / "reports"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. 常量
# ═══════════════════════════════════════════════════════════════════════════════

# 标准羽毛球场（米）
COURT_LEN   = 13.40   # 纵向（发球端到底线）
COURT_WID   = 6.10    # 横向（双打宽）
NET_H       = 1.55    # 网中心高度
SINGLES_WID = 5.18    # 单打宽度（双侧各缩0.46m）
SHORT_SVC   = 1.98    # 短发球线距网
DBL_LONG_SVC= 0.76    # 双打长发球线距底线
ASSUMED_HEIGHT_M = 1.70  # 假设球员身高（用于尺度估算）

# COCO 17 关键点索引
KP = dict(
    nose=0, left_eye=1, right_eye=2, left_ear=3, right_ear=4,
    left_shoulder=5, right_shoulder=6, left_elbow=7, right_elbow=8,
    left_wrist=9, right_wrist=10, left_hip=11, right_hip=12,
    left_knee=13, right_knee=14, left_ankle=15, right_ankle=16
)
SKELETON = [
    (0,1),(0,2),(1,3),(2,4),(0,5),(0,6),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16),
]

# ═══════════════════════════════════════════════════════════════════════════════
# 2. 球场标定
# ═══════════════════════════════════════════════════════════════════════════════

class CourtCalibration:
    """将图像像素坐标映射到球场实际坐标（米）"""

    def __init__(self, frame_h, frame_w, corners_px=None):
        """
        corners_px: [(x0,y0),(x1,y1),(x2,y2),(x3,y3)] 球场四角像素坐标
                    顺序：左上、右上、右下、左下（俯视球场）
                    如果为 None，使用简单线性映射（假设摄像头基本正对球场）
        """
        self.frame_h = frame_h
        self.frame_w = frame_w
        self.corners_px = corners_px
        self.homography = None
        self.px_per_m = None  # 全局尺度（估算用）

        if corners_px is not None and CV2_AVAILABLE:
            # 计算单应性矩阵
            src = np.array(corners_px, dtype=np.float32)
            dst = np.array([
                [0, 0], [COURT_WID, 0],
                [COURT_WID, COURT_LEN], [0, COURT_LEN]
            ], dtype=np.float32)
            self.homography, _ = cv2.findHomography(src, dst)

    def set_scale_from_person(self, kp_array):
        """用人体关键点估算像素/米比例（仅需一帧）"""
        nose_conf  = kp_array[KP['nose']][2]
        la_conf    = kp_array[KP['left_ankle']][2]
        ra_conf    = kp_array[KP['right_ankle']][2]

        nose_y = kp_array[KP['nose']][1] if nose_conf > 0.3 else None
        if la_conf > 0.3 and ra_conf > 0.3:
            ankle_y = (kp_array[KP['left_ankle']][1] +
                       kp_array[KP['right_ankle']][1]) / 2
        elif la_conf > 0.3:
            ankle_y = kp_array[KP['left_ankle']][1]
        elif ra_conf > 0.3:
            ankle_y = kp_array[KP['right_ankle']][1]
        else:
            ankle_y = None

        if nose_y is not None and ankle_y is not None:
            pixel_h = abs(ankle_y - nose_y)
            if pixel_h > 10:
                self.px_per_m = pixel_h / ASSUMED_HEIGHT_M

    def pixel_to_court(self, px, py):
        """
        返回 (court_x_m, court_y_m)，原点在球场左上角，
        x 轴 = 宽度方向（0~6.10m），y 轴 = 长度方向（0~13.40m）
        """
        if self.homography is not None:
            pt = np.array([[[float(px), float(py)]]], dtype=np.float32)
            out = cv2.perspectiveTransform(pt, self.homography)
            cx, cy = float(out[0][0][0]), float(out[0][0][1])
        else:
            # 线性映射（简单近似）
            cx = (px / self.frame_w) * COURT_WID
            cy = (py / self.frame_h) * COURT_LEN
        return cx, cy

    def pixel_dist_to_m(self, px_dist):
        """像素距离 → 米（需先调用 set_scale_from_person）"""
        if self.px_per_m and self.px_per_m > 0:
            return px_dist / self.px_per_m
        # 后备：用帧高估算（保守）
        return px_dist / self.frame_h * COURT_LEN

    def wrist_height_m(self, kp_array, hand='right'):
        """计算腕部相对地面高度（米）"""
        w_key = 'right_wrist' if hand == 'right' else 'left_wrist'
        a_key = 'right_ankle' if hand == 'right' else 'left_ankle'
        wrist = kp_array[KP[w_key]]
        ankle = kp_array[KP[a_key]]
        if wrist[2] > 0.25 and ankle[2] > 0.25:
            pixel_diff = ankle[1] - wrist[1]  # y 向下，差值为正
            return max(0.0, self.pixel_dist_to_m(pixel_diff))
        # 备选：用肩膀和臀部估算身体各节段
        hip = kp_array[KP['right_hip' if hand == 'right' else 'left_hip']]
        if wrist[2] > 0.25 and hip[2] > 0.25:
            pixel_diff = hip[1] - wrist[1]
            return max(0.0, self.pixel_dist_to_m(pixel_diff) + 0.5)  # 加上腿长估计
        return None

# ═══════════════════════════════════════════════════════════════════════════════
# 3. YOLOv8n-Pose 推理
# ═══════════════════════════════════════════════════════════════════════════════

class PoseInfer:
    INPUT_SIZE = 640

    def __init__(self, model_path: str):
        providers = ort.get_available_providers()
        logger.info(f"加载姿态模型: {model_path}  后端: {providers}")
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.inp_name = self.session.get_inputs()[0].name

    def infer(self, frame_bgr: np.ndarray):
        """返回 (results, infer_ms)，results=[{bbox,score,keypoints}]"""
        h, w = frame_bgr.shape[:2]
        scale = self.INPUT_SIZE / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        pad_h = (self.INPUT_SIZE - new_h) // 2
        pad_w = (self.INPUT_SIZE - new_w) // 2

        if CV2_AVAILABLE:
            resized = cv2.resize(frame_bgr, (new_w, new_h))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        else:
            rgb = np.array(Image.fromarray(frame_bgr).resize((new_w, new_h)))

        padded = np.full((self.INPUT_SIZE, self.INPUT_SIZE, 3), 114, np.uint8)
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = rgb
        tensor = (padded.astype(np.float32) / 255.0
                  ).transpose(2,0,1)[np.newaxis]

        t0 = time.perf_counter()
        out = self.session.run(None, {self.inp_name: tensor})
        ms = (time.perf_counter() - t0) * 1000

        preds = out[0][0].T  # [N,56]
        mask  = preds[:, 4] > 0.25
        preds = preds[mask]
        results = []
        for p in preds:
            cx,cy,bw,bh = p[:4]
            kps = p[5:].reshape(17,3)
            kps[:,0] = (kps[:,0] - pad_w) / scale
            kps[:,1] = (kps[:,1] - pad_h) / scale
            results.append({
                "bbox": [(cx-bw/2-pad_w)/scale, (cy-bh/2-pad_h)/scale,
                         (cx+bw/2-pad_w)/scale, (cy+bh/2-pad_h)/scale],
                "score": float(p[4]),
                "keypoints": kps,   # np.ndarray [17,3]
            })
        results.sort(key=lambda x: x['score'], reverse=True)
        return results, ms


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 球体运动检测（单目，帧差法）
# ═══════════════════════════════════════════════════════════════════════════════

class BallDetector:
    """用背景差分 + 形态学滤波检测羽毛球（速度估算用）"""
    def __init__(self, min_area=4, max_area=300, thresh=20):
        self.min_area = min_area
        self.max_area = max_area
        self.thresh   = thresh
        self.prev_gray = None
        self.trajectory = []   # [(frame_idx, cx, cy)]

    def update(self, frame, frame_idx):
        """返回候选球心 (cx, cy) 或 None"""
        if CV2_AVAILABLE:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5,5), 0)
        else:
            gray = np.array(Image.fromarray(frame).convert('L'))

        if self.prev_gray is None:
            self.prev_gray = gray
            return None

        diff = np.abs(gray.astype(np.int16) - self.prev_gray.astype(np.int16))
        diff = diff.astype(np.uint8)
        _, binary = (cv2.threshold(diff, self.thresh, 255, cv2.THRESH_BINARY)
                     if CV2_AVAILABLE
                     else (None, (diff > self.thresh).astype(np.uint8) * 255))
        self.prev_gray = gray

        if CV2_AVAILABLE:
            kernel = np.ones((3,3), np.uint8)
            binary = cv2.dilate(binary, kernel, iterations=2)
            binary = cv2.erode(binary, kernel, iterations=1)
            cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        else:
            return None  # 无 cv2 时跳过球体检测

        best = None
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if self.min_area < area < self.max_area:
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    cx = M['m10'] / M['m00']
                    cy = M['m01'] / M['m00']
                    if best is None or area < best[2]:
                        best = (cx, cy, area)

        if best:
            self.trajectory.append((frame_idx, best[0], best[1]))
            return best[0], best[1]
        return None

    def compute_ball_speed(self, fps, calibration: CourtCalibration):
        """从轨迹片段计算球速序列（m/s）"""
        if len(self.trajectory) < 2:
            return []
        speeds = []
        for i in range(1, len(self.trajectory)):
            fi, xi, yi = self.trajectory[i]
            fj, xj, yj = self.trajectory[i-1]
            if fi == fj:
                continue
            px_dist = math.hypot(xi - xj, yi - yj)
            m_dist  = calibration.pixel_dist_to_m(px_dist)
            dt      = (fi - fj) / fps
            if dt > 0:
                speeds.append((fi, m_dist / dt))
        return speeds  # [(frame_idx, speed_m_s)]


# ═══════════════════════════════════════════════════════════════════════════════
# 5. 指标计算
# ═══════════════════════════════════════════════════════════════════════════════

def smooth(arr, window=5):
    """简单移动平均平滑"""
    if len(arr) < window:
        return arr
    out = np.convolve(arr, np.ones(window)/window, mode='same')
    return out

def compute_arm_speeds(kp_seq, fps, calibration: CourtCalibration):
    """
    kp_seq: list of np.ndarray [17,3]（每帧主要人物关键点）
    返回 np.ndarray 每帧腕部速度（km/h），长度 = len(kp_seq)
    """
    speeds = np.zeros(len(kp_seq))
    for i in range(1, len(kp_seq)):
        prev, curr = kp_seq[i-1], kp_seq[i]
        best = 0.0
        for wrist_k in ('left_wrist', 'right_wrist'):
            p = prev[KP[wrist_k]]
            c = curr[KP[wrist_k]]
            if p[2] > 0.25 and c[2] > 0.25:
                px_dist = math.hypot(c[0]-p[0], c[1]-p[1])
                m_dist  = calibration.pixel_dist_to_m(px_dist)
                spd     = m_dist * fps        # m/s
                best    = max(best, spd)
        speeds[i] = best
    return smooth(speeds, window=3)  # 轻度平滑去抖

def detect_shot_frames(arm_speeds_ms, fps,
                       min_speed_ms=1.5, min_gap_s=0.25):
    """
    在挥臂速度序列中找击球峰值帧
    返回 list[int] 峰值帧索引
    """
    min_gap = int(min_gap_s * fps)
    peaks = []
    n = len(arm_speeds_ms)
    for i in range(1, n-1):
        if arm_speeds_ms[i] > min_speed_ms:
            if (arm_speeds_ms[i] >= arm_speeds_ms[i-1] and
                arm_speeds_ms[i] >= arm_speeds_ms[i+1]):
                if not peaks or (i - peaks[-1]) > min_gap:
                    peaks.append(i)
    return peaks

def classify_shot(kp_seq, shot_idx, window=6):
    """
    判断主导手（left / right）
    返回 'left' 或 'right'
    """
    start = max(1, shot_idx - window)
    end   = min(len(kp_seq), shot_idx + window + 1)
    lw_max = rw_max = 0.0
    for i in range(start, end):
        prev, curr = kp_seq[i-1], kp_seq[i]
        for k, wk in (('left_wrist', 'lw_max'), ('right_wrist', 'rw_max')):
            p = prev[KP[k]]; c = curr[KP[k]]
            if p[2] > 0.25 and c[2] > 0.25:
                spd = math.hypot(c[0]-p[0], c[1]-p[1])
                if wk == 'lw_max': lw_max = max(lw_max, spd)
                else:              rw_max = max(rw_max, spd)
    return 'left' if lw_max > rw_max else 'right'

def shot_position_type(kp_array, frame_h):
    """
    根据球员站位（踝部纵向位置）判断是发球方还是接球方
    返回 'serve_side' 或 'receive_side'
    """
    la = kp_array[KP['left_ankle']]
    ra = kp_array[KP['right_ankle']]
    if la[2] > 0.25 and ra[2] > 0.25:
        avg_y = (la[1] + ra[1]) / 2
    elif la[2] > 0.25:
        avg_y = la[1]
    elif ra[2] > 0.25:
        avg_y = ra[1]
    else:
        return 'serve_side'
    return 'serve_side' if avg_y < frame_h / 2 else 'receive_side'

def estimate_ball_speed(arm_speed_ms):
    """
    基于挥臂速度估算球速（m/s）
    参考文献：badminton smash wrist speed ≈ 10-25 m/s → 球速 ≈ 1.8-2.3x
    保守取 2.0 倍（考虑非全力扣杀情况）
    """
    return arm_speed_ms * 2.0

def get_court_pos(kp_array, calibration: CourtCalibration):
    """返回球员在球场的坐标 (x_m, y_m) 或 None"""
    la = kp_array[KP['left_ankle']]
    ra = kp_array[KP['right_ankle']]
    if la[2] > 0.25 and ra[2] > 0.25:
        px = (la[0] + ra[0]) / 2
        py = (la[1] + ra[1]) / 2
    elif la[2] > 0.25:
        px, py = la[0], la[1]
    elif ra[2] > 0.25:
        px, py = ra[0], ra[1]
    else:
        return None
    return calibration.pixel_to_court(px, py)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. 视频处理主流程
# ═══════════════════════════════════════════════════════════════════════════════

def process_video(video_path: str, model_path: str,
                  corners_px=None, max_frames=-1):
    """
    完整视频分析，返回 raw_data dict
    如模型不存在则返回 None
    """
    # ── 1. 读取视频帧 ────────────────────────────────────────────────────────
    video_path = Path(video_path)
    if not video_path.exists():
        # 尝试相对于项目根目录
        alt = PROJECT_ROOT / video_path
        if alt.exists():
            video_path = alt
        else:
            logger.error(f"视频不存在: {video_path}")
            return None

    logger.info(f"读取视频: {video_path}")
    frames_bgr = []
    fps = 30.0

    if CV2_AVAILABLE:
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            idx = 0
            while True:
                ret, f = cap.read()
                if not ret: break
                if max_frames > 0 and idx >= max_frames: break
                frames_bgr.append(f)
                idx += 1
            cap.release()
            logger.info(f"OpenCV 读取: {len(frames_bgr)} 帧 @ {fps:.1f}fps")

    if not frames_bgr:
        try:
            import imageio.v3 as iio
            props = iio.improps(str(video_path), plugin="pyav")
            fps = getattr(props, 'fps', None) or 30.0
            reader = iio.imiter(str(video_path), plugin="pyav")
            for i, frm in enumerate(reader):
                if max_frames > 0 and i >= max_frames: break
                frames_bgr.append(frm[:,:,::-1].copy())  # RGB→BGR
            logger.info(f"imageio 读取: {len(frames_bgr)} 帧 @ {fps:.1f}fps")
        except Exception as e:
            logger.error(f"视频读取失败: {e}")
            return None

    if not frames_bgr:
        logger.error("未读取到任何帧")
        return None

    frame_h, frame_w = frames_bgr[0].shape[:2]
    video_duration_s = len(frames_bgr) / fps

    # ── 2. 初始化组件 ────────────────────────────────────────────────────────
    calib = CourtCalibration(frame_h, frame_w, corners_px)
    ball_det = BallDetector()

    # ── 3. 姿态推理 ──────────────────────────────────────────────────────────
    if not Path(model_path).exists():
        logger.warning(f"模型文件不存在: {model_path}，切换演示模式")
        return None

    if not ORT_AVAILABLE:
        logger.warning("onnxruntime 不可用，切换演示模式")
        return None

    try:
        infer = PoseInfer(model_path)
    except Exception as e:
        logger.warning(f"模型加载失败: {e}，切换演示模式")
        return None

    kp_seq   = []   # 每帧主要人物关键点 [17,3]，若未检测到则 None
    infer_ms = []
    ball_positions = []

    logger.info(f"开始逐帧推理 (共 {len(frames_bgr)} 帧)...")
    for fi, frame in enumerate(frames_bgr):
        results, ms = infer.infer(frame)
        infer_ms.append(ms)

        if results:
            kp_seq.append(results[0]['keypoints'])
            # 首次有效关键点时估算尺度
            if calib.px_per_m is None:
                calib.set_scale_from_person(results[0]['keypoints'])
        else:
            kp_seq.append(kp_seq[-1] if kp_seq else
                          np.zeros((17, 3), np.float32))

        # 球体检测
        ball_pt = ball_det.update(frame, fi)
        if ball_pt:
            ball_positions.append((fi, ball_pt[0], ball_pt[1]))

        if fi % 50 == 0:
            logger.info(f"  {fi}/{len(frames_bgr)} 帧，"
                        f"推理 {ms:.0f}ms，"
                        f"px/m={calib.px_per_m:.1f}" if calib.px_per_m else
                        f"  {fi}/{len(frames_bgr)} 帧，推理 {ms:.0f}ms")

    # ── 4. 计算指标 ──────────────────────────────────────────────────────────
    if calib.px_per_m is None:
        calib.px_per_m = frame_h / COURT_LEN  # 后备：假设人站满画面高

    arm_speeds_ms  = compute_arm_speeds(kp_seq, fps, calib)  # m/s 每帧
    arm_speeds_kmh = arm_speeds_ms * 3.6

    shot_frames = detect_shot_frames(arm_speeds_ms, fps)

    shots = []
    for sf in shot_frames:
        hand   = classify_shot(kp_seq, sf)
        height = calib.wrist_height_m(kp_seq[sf], hand)
        pos    = get_court_pos(kp_seq[sf], calib)
        arm_s  = float(arm_speeds_ms[sf])
        ball_s = estimate_ball_speed(arm_s)
        side   = shot_position_type(kp_seq[sf], frame_h)
        shots.append({
            "frame": sf,
            "time_s": sf / fps,
            "hand": hand,
            "height_m": float(height) if height else 1.5,
            "arm_speed_ms": arm_s,
            "arm_speed_kmh": arm_s * 3.6,
            "ball_speed_ms": ball_s,
            "ball_speed_kmh": ball_s * 3.6,
            "court_pos": list(pos) if pos else [COURT_WID/2, COURT_LEN/2],
            "side": side,
        })

    # 球体检测球速
    ball_speed_series = ball_det.compute_ball_speed(fps, calib)
    measured_ball_kmh = ([s * 3.6 for _, s in ball_speed_series]
                         if ball_speed_series else [])

    # 场地位置序列
    court_positions = []
    for kp in kp_seq:
        pos = get_court_pos(kp, calib)
        if pos:
            court_positions.append(list(pos))

    return {
        "video_name": video_path.name,
        "duration_s": video_duration_s,
        "fps": fps,
        "frame_count": len(frames_bgr),
        "frame_h": frame_h, "frame_w": frame_w,
        "avg_infer_ms": float(np.mean(infer_ms)),
        "arm_speeds_kmh": arm_speeds_kmh.tolist(),   # per-frame
        "shots": shots,
        "court_positions": court_positions,
        "measured_ball_speeds_kmh": measured_ball_kmh,
        "ball_detection_count": len(ball_positions),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 7. 演示数据生成（无视频时使用）
# ═══════════════════════════════════════════════════════════════════════════════

def generate_demo_data():
    """生成仿真训练数据（用于演示）"""
    random.seed(42)
    np.random.seed(42)
    fps = 30.0
    duration = 120.0  # 2分钟模拟训练
    n_frames = int(fps * duration)

    # 模拟挥臂速度（背景 0-2 m/s，击球峰值 8-22 m/s）
    arm_raw = np.random.exponential(0.5, n_frames) * 3.6  # km/h base
    shots = []
    shot_frames = sorted(random.sample(range(30, n_frames-30), 24))
    for sf in shot_frames:
        peak = random.uniform(18, 75)   # km/h
        for delta in range(-8, 9):
            idx = sf + delta
            if 0 <= idx < n_frames:
                arm_raw[idx] = max(arm_raw[idx],
                                   peak * math.exp(-0.5 * (delta/3)**2))
        hand   = random.choice(['right','right','right','left'])
        height = random.uniform(1.6, 2.8)
        arm_ms = peak / 3.6
        ball   = estimate_ball_speed(arm_ms)
        # 随机球场位置（多在底线附近）
        cx = random.uniform(0.8, COURT_WID - 0.8)
        cy = random.choice([
            random.uniform(9.5, 12.5),   # 底线附近
            random.uniform(1.0, 4.0),    # 网前
        ])
        side = 'serve_side' if cy < COURT_LEN/2 else 'receive_side'
        shots.append({
            "frame": sf, "time_s": sf/fps,
            "hand": hand, "height_m": height,
            "arm_speed_ms": arm_ms, "arm_speed_kmh": peak,
            "ball_speed_ms": ball, "ball_speed_kmh": ball*3.6,
            "court_pos": [cx, cy], "side": side,
        })

    # 场地热图：主要在两个底线区域徘徊
    court_pos = []
    for _ in range(300):
        cx = random.gauss(COURT_WID/2, 0.8)
        cy = random.gauss(random.choice([10.5, 2.8]), 1.2)
        cx = max(0.1, min(COURT_WID-0.1, cx))
        cy = max(0.1, min(COURT_LEN-0.1, cy))
        court_pos.append([cx, cy])

    # 模拟球体检测球速（部分帧）
    measured_ball_kmh = [random.uniform(60, 280)
                         for _ in range(random.randint(5, 15))]

    return {
        "video_name": "演示训练录像",
        "duration_s": duration,
        "fps": fps,
        "frame_count": n_frames,
        "frame_h": 1280, "frame_w": 720,
        "avg_infer_ms": 48.3,
        "arm_speeds_kmh": arm_raw.tolist(),
        "shots": shots,
        "court_positions": court_pos,
        "measured_ball_speeds_kmh": measured_ball_kmh,
        "ball_detection_count": len(measured_ball_kmh) * 3,
        "is_demo": True,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 8. 摘要统计
# ═══════════════════════════════════════════════════════════════════════════════

def compute_summary(raw: dict) -> dict:
    shots = raw['shots']
    arm_kmh = raw['arm_speeds_kmh']

    # 挥臂速度
    arm_arr  = np.array(arm_kmh)
    peak_arm = float(np.percentile(arm_arr, 95)) if len(arm_arr) else 0.0
    avg_arm  = float(np.mean([s['arm_speed_kmh'] for s in shots])) if shots else 0.0
    max_arm  = float(max((s['arm_speed_kmh'] for s in shots), default=0.0))

    # 球速
    ball_all = ([s['ball_speed_kmh'] for s in shots] +
                raw.get('measured_ball_speeds_kmh', []))
    avg_ball = float(np.mean(ball_all)) if ball_all else 0.0
    max_ball = float(max(ball_all, default=0.0))
    ball_source = ("运动检测+估算" if raw.get('measured_ball_speeds_kmh')
                   else "基于挥臂速度估算")

    # 击球分析
    left_shots  = [s for s in shots if s['hand'] == 'left']
    right_shots = [s for s in shots if s['hand'] == 'right']
    serve_shots = [s for s in shots if s['side'] == 'serve_side']
    recv_shots  = [s for s in shots if s['side'] == 'receive_side']

    # 击球高度
    heights = [s['height_m'] for s in shots if s['height_m'] > 0]
    avg_height = float(np.mean(heights)) if heights else 0.0
    max_height = float(max(heights, default=0.0))

    # 速度时间序列（降采样到最多200点）
    n = len(arm_kmh)
    step = max(1, n // 200)
    time_axis = [i / raw['fps'] for i in range(0, n, step)]
    arm_series = arm_kmh[::step] if step == 1 else [
        float(np.mean(arm_kmh[i:i+step]))
        for i in range(0, n, step)
    ]

    # 击球高度分布（分桶）
    height_bins = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    height_labels = ['<0.5m','0.5-1m','1-1.5m','1.5-2m',
                     '2-2.5m','2.5-3m','>3m']
    height_counts = [0] * len(height_labels)
    for h in heights:
        for bi, bv in enumerate(height_bins[1:]):
            if h < bv:
                height_counts[bi] += 1
                break
        else:
            height_counts[-1] += 1

    # 速度分布（击球时）
    speed_levels = [
        ('低速 (<30 km/h)', sum(1 for s in shots if s['arm_speed_kmh'] < 30)),
        ('中速 (30-50 km/h)', sum(1 for s in shots if 30 <= s['arm_speed_kmh'] < 50)),
        ('高速 (50-70 km/h)', sum(1 for s in shots if 50 <= s['arm_speed_kmh'] < 70)),
        ('极速 (>70 km/h)', sum(1 for s in shots if s['arm_speed_kmh'] >= 70)),
    ]

    return {
        "total_shots": len(shots),
        "left_shots": len(left_shots),
        "right_shots": len(right_shots),
        "serve_shots": len(serve_shots),
        "recv_shots": len(recv_shots),
        "avg_arm_kmh": round(avg_arm, 1),
        "max_arm_kmh": round(max_arm, 1),
        "avg_ball_kmh": round(avg_ball, 1),
        "max_ball_kmh": round(max_ball, 1),
        "ball_source": ball_source,
        "avg_height_m": round(avg_height, 2),
        "max_height_m": round(max_height, 2),
        "time_axis": time_axis,
        "arm_series": arm_series,
        "height_counts": height_counts,
        "height_labels": height_labels,
        "speed_levels": speed_levels,
        "court_positions": raw.get('court_positions', []),
        "shots_detail": shots,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 9. SVG 球场热图
# ═══════════════════════════════════════════════════════════════════════════════

def generate_court_svg(positions, shot_events, svg_w=210, svg_h=460):
    """
    生成标准羽毛球场 SVG（含站位热图 + 击球标注）
    坐标系：左上角 = (0,0)，x→右，y→下
    球场尺寸：COURT_WID × COURT_LEN
    """
    M = 20  # 边距（像素）
    W = svg_w - 2*M
    H = svg_h - 2*M

    def cx(x_m):  return M + x_m / COURT_WID  * W
    def cy(y_m):  return M + y_m / COURT_LEN * H

    lines = []
    # 背景
    lines.append(f'<rect x="0" y="0" width="{svg_w}" height="{svg_h}" '
                 f'fill="#0f4c2a" rx="4"/>')

    def line(x1m, y1m, x2m, y2m, color="white", w=1.2):
        lines.append(f'<line x1="{cx(x1m):.1f}" y1="{cy(y1m):.1f}" '
                     f'x2="{cx(x2m):.1f}" y2="{cy(y2m):.1f}" '
                     f'stroke="{color}" stroke-width="{w}"/>')

    def rect(x1m, y1m, x2m, y2m, fill="none", stroke="white", sw=1.5):
        lines.append(f'<rect x="{cx(x1m):.1f}" y="{cy(y1m):.1f}" '
                     f'width="{cx(x2m)-cx(x1m):.1f}" '
                     f'height="{cy(y2m)-cy(y1m):.1f}" '
                     f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    # 外框（双打场地）
    rect(0, 0, COURT_WID, COURT_LEN, stroke="white", sw=2)

    # 单打边线（各缩 0.46m）
    SING = (COURT_WID - SINGLES_WID) / 2  # 0.46m
    line(SING, 0, SING, COURT_LEN, color="#ffffffaa", w=0.8)
    line(COURT_WID-SING, 0, COURT_WID-SING, COURT_LEN, color="#ffffffaa", w=0.8)

    # 中线（网线）
    NET_Y = COURT_LEN / 2
    line(0, NET_Y, COURT_WID, NET_Y, color="#ffcc00", w=2.5)

    # 短发球线（距网 1.98m 两侧）
    line(0, NET_Y - SHORT_SVC, COURT_WID, NET_Y - SHORT_SVC, w=0.9)
    line(0, NET_Y + SHORT_SVC, COURT_WID, NET_Y + SHORT_SVC, w=0.9)

    # 双打长发球线（距底线 0.76m）
    line(0, DBL_LONG_SVC, COURT_WID, DBL_LONG_SVC, w=0.9)
    line(0, COURT_LEN - DBL_LONG_SVC, COURT_WID, COURT_LEN - DBL_LONG_SVC, w=0.9)

    # 中间纵线（发球区分割）
    MID_X = COURT_WID / 2
    line(MID_X, NET_Y - SHORT_SVC, MID_X, NET_Y, w=0.9)
    line(MID_X, NET_Y, MID_X, NET_Y + SHORT_SVC, w=0.9)

    # 网（可视化）
    lines.append(f'<rect x="{cx(0):.1f}" y="{cy(NET_Y)-3:.1f}" '
                 f'width="{cx(COURT_WID)-cx(0):.1f}" height="6" '
                 f'fill="#ffcc0088" rx="1"/>')

    # 标注文字
    font = 'font-family="Arial" fill="white" font-size="8"'
    lines.append(f'<text x="{svg_w/2:.0f}" y="{cy(NET_Y):.0f}" '
                 f'text-anchor="middle" {font} fill="#ffcc00" '
                 f'font-weight="bold" dy="14">网 {NET_H}m</text>')

    # 站位热图（半透明彩色圆点）
    MAX_OPACITY = 0.65
    for (xm, ym) in positions:
        xm = max(0, min(COURT_WID, xm))
        ym = max(0, min(COURT_LEN, ym))
        # 颜色：底线=红热，中场=橙，网前=蓝
        rel = ym / COURT_LEN
        if rel > 0.7:   c = "#ff4444"
        elif rel > 0.3: c = "#ff8800"
        else:           c = "#44aaff"
        lines.append(f'<circle cx="{cx(xm):.1f}" cy="{cy(ym):.1f}" '
                     f'r="5" fill="{c}" opacity="{MAX_OPACITY}"/>')

    # 击球事件标注
    for s in shot_events[:30]:  # 最多显示30个
        if not s.get('court_pos'):
            continue
        xm, ym = s['court_pos']
        xm = max(0, min(COURT_WID, xm))
        ym = max(0, min(COURT_LEN, ym))
        color = "#22ee88" if s['hand'] == 'right' else "#ee88ff"
        lines.append(f'<polygon points="{cx(xm):.1f},{cy(ym)-7:.1f} '
                     f'{cx(xm)-4:.1f},{cy(ym)+3:.1f} '
                     f'{cx(xm)+4:.1f},{cy(ym)+3:.1f}" '
                     f'fill="{color}" opacity="0.9"/>')

    # 图例
    LG = cy(COURT_LEN) + M/2 + 8
    items = [
        ("#ff4444", "底线"),("#ff8800", "中场"),("#44aaff", "网前"),
        ("#22ee88", "右手"), ("#ee88ff", "左手"),
    ]
    xoff = M
    for color, label in items:
        lines.append(f'<circle cx="{xoff+4}" cy="{LG}" r="4" fill="{color}"/>')
        lines.append(f'<text x="{xoff+11}" y="{LG+4}" '
                     f'font-family="Arial" font-size="7" fill="#ccc">{label}</text>')
        xoff += 38

    total_h = svg_h + 20
    svg = (f'<svg width="{svg_w}" height="{total_h}" '
           f'xmlns="http://www.w3.org/2000/svg">\n' +
           '\n'.join(lines) + '\n</svg>')
    return svg


# ═══════════════════════════════════════════════════════════════════════════════
# 10. AI 训练建议
# ═══════════════════════════════════════════════════════════════════════════════

def generate_recommendations(summary: dict, raw: dict) -> list:
    recs = []
    left  = summary['left_shots']
    right = summary['right_shots']
    total = summary['total_shots']

    if total == 0:
        return ["本次训练未检测到有效击球事件，建议检查视频质量或摄像角度。"]

    # 手部平衡
    if total > 0:
        dom = 'right' if right >= left else 'left'
        weak = 'left' if dom == 'right' else 'right'
        dom_pct = max(left, right) / total * 100
        if dom_pct > 80:
            recs.append(
                f"⚠️ 击球手势严重偏向{'右' if dom == 'right' else '左'}手（{dom_pct:.0f}%），"
                f"建议专项练习{'左' if weak == 'left' else '右'}手{'反手' if weak == 'left' else '正手'}击球。"
            )
        elif dom_pct > 65:
            recs.append(
                f"✅ 主导手为{'右' if dom == 'right' else '左'}手，"
                f"{'左' if weak == 'left' else '右'}手使用率{100-dom_pct:.0f}%，"
                f"适当增加弱手练习可提升全场覆盖能力。"
            )

    # 挥臂速度
    avg = summary['avg_arm_kmh']
    mx  = summary['max_arm_kmh']
    if avg < 30:
        recs.append(
            f"⚠️ 平均挥臂速度 {avg} km/h 偏低，建议加强肩袖肌群爆发力训练，"
            f"可通过重锤甩臂、弹力带旋转等专项练习提升。"
        )
    elif avg < 55:
        recs.append(
            f"✅ 挥臂速度中等（均值 {avg} km/h，峰值 {mx} km/h），"
            f"重点提升扣杀动作连贯性，加强随挥动作以保护肩部。"
        )
    else:
        recs.append(
            f"🏆 挥臂速度优秀（均值 {avg} km/h，峰值 {mx} km/h），"
            f"建议维持同时注重落点控制精度。"
        )

    # 击球高度
    avg_h = summary['avg_height_m']
    if avg_h < 1.5:
        recs.append(
            f"⚠️ 平均击球高度仅 {avg_h:.2f}m，低于网高（1.55m），"
            f"过低击球点使球以大角度向上飞行，易被对手截杀。"
            f"建议提前移动、争取高点击球（目标 ≥ 2.0m）。"
        )
    elif avg_h < 2.0:
        recs.append(
            f"✅ 击球高度 {avg_h:.2f}m 接近标准，"
            f"可通过脚步训练和预判改善，争取更多高压球机会。"
        )
    else:
        recs.append(
            f"🏆 击球高度 {avg_h:.2f}m，处于进攻优势区间，"
            f"保持高压进攻节奏的同时注意落点变化。"
        )

    # 站位
    court_pos = summary['court_positions']
    if court_pos:
        xs = [p[0] for p in court_pos]
        ys = [p[1] for p in court_pos]
        center_y = np.mean(ys)
        x_std = np.std(xs)
        if center_y > COURT_LEN * 0.65:
            recs.append(
                f"⚠️ 站位偏向后场（平均纵向 {center_y:.1f}m），"
                f"被动防守比例较高。建议加强网前步法，主动创造进攻机会。"
            )
        if x_std < 0.5:
            recs.append(
                f"⚠️ 横向移动范围偏小（标准差 {x_std:.2f}m），"
                f"大量球在中路处理。提升两侧大角度接球能力。"
            )

    # 球速
    avg_b = summary['avg_ball_kmh']
    if avg_b > 0:
        recs.append(
            f"📊 估算球速均值 {avg_b:.0f} km/h，峰值 {summary['max_ball_kmh']:.0f} km/h。"
            f"（注：由挥臂速度×物理系数估算，实测需双目摄像头）"
        )

    if raw.get('is_demo'):
        recs.insert(0, "📌 当前为演示报告（合成数据），接入真实视频后分析结果将更准确。")

    return recs


# ═══════════════════════════════════════════════════════════════════════════════
# 11. HTML 报告模板
# ═══════════════════════════════════════════════════════════════════════════════

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>羽毛球训练分析报告 — {video_name}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:#f0f4f8;--card:#fff;--navy:#0f2d5a;--blue:#1d6fa5;--cyan:#06b6d4;
  --green:#16a34a;--amber:#d97706;--red:#dc2626;--purple:#7c3aed;
  --gray:#64748b;--border:#e2e8f0;--text:#1e293b;--subtext:#475569;
}}
body{{background:var(--bg);font-family:'Segoe UI',system-ui,sans-serif;color:var(--text);font-size:14px;padding:24px}}
.page{{max-width:1120px;margin:0 auto}}

/* Header */
.header{{background:linear-gradient(135deg,var(--navy) 0%,#1d3d6b 60%,#1d6fa5 100%);
  border-radius:16px;padding:32px 40px;margin-bottom:24px;
  display:flex;justify-content:space-between;align-items:flex-start;color:#fff}}
.header-logo{{font-size:28px;font-weight:800;letter-spacing:-0.5px}}
.header-sub{{font-size:13px;opacity:0.75;margin-top:4px}}
.header-meta{{text-align:right;font-size:12px;opacity:0.8;line-height:1.8}}
.badge{{display:inline-block;background:rgba(255,255,255,0.15);
  border-radius:20px;padding:2px 10px;font-size:11px;margin-top:8px}}

/* Section titles */
.section-title{{font-size:13px;font-weight:700;color:var(--gray);
  text-transform:uppercase;letter-spacing:0.8px;margin-bottom:12px;
  padding-left:12px;border-left:3px solid var(--cyan)}}

/* Metric cards */
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px}}
.card{{background:var(--card);border-radius:14px;padding:20px 22px;
  box-shadow:0 1px 4px rgba(0,0,0,.07);border:1px solid var(--border);
  position:relative;overflow:hidden}}
.card::before{{content:'';position:absolute;top:0;left:0;right:0;height:4px;background:var(--accent)}}
.card.arm{{--accent:linear-gradient(90deg,#06b6d4,#1d6fa5)}}
.card.ball{{--accent:linear-gradient(90deg,#f59e0b,#dc2626)}}
.card.shot{{--accent:linear-gradient(90deg,#7c3aed,#ec4899)}}
.card.height{{--accent:linear-gradient(90deg,#16a34a,#06b6d4)}}
.card-icon{{font-size:22px;margin-bottom:8px}}
.card-label{{font-size:11px;color:var(--gray);font-weight:600;text-transform:uppercase;letter-spacing:0.5px}}
.card-value{{font-size:38px;font-weight:800;line-height:1.1;margin:4px 0}}
.card-unit{{font-size:14px;font-weight:500;color:var(--subtext);margin-left:2px}}
.card-sub{{font-size:11px;color:var(--subtext);margin-top:4px}}
.card-detail{{font-size:11px;margin-top:8px;padding-top:8px;border-top:1px solid var(--border);
  display:flex;gap:12px;flex-wrap:wrap}}
.pill{{background:var(--bg);border-radius:20px;padding:3px 10px;font-weight:600}}
.pill.green{{background:#dcfce7;color:var(--green)}}
.pill.blue{{background:#dbeafe;color:var(--blue)}}
.pill.red{{background:#fee2e2;color:var(--red)}}
.pill.purple{{background:#f3e8ff;color:var(--purple)}}

/* Charts grid */
.charts-full{{margin-bottom:24px;background:var(--card);border-radius:14px;
  padding:22px;box-shadow:0 1px 4px rgba(0,0,0,.07);border:1px solid var(--border)}}
.charts-row{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:24px}}
.chart-box{{background:var(--card);border-radius:14px;padding:22px;
  box-shadow:0 1px 4px rgba(0,0,0,.07);border:1px solid var(--border)}}
.chart-box canvas{{max-height:240px}}

/* Court + analysis row */
.analysis-row{{display:grid;grid-template-columns:240px 1fr;gap:16px;margin-bottom:24px}}
.court-box{{background:var(--card);border-radius:14px;padding:20px;
  box-shadow:0 1px 4px rgba(0,0,0,.07);border:1px solid var(--border);text-align:center}}
.court-box svg{{max-width:100%;height:auto}}

/* Shot table */
.shot-table-box{{background:var(--card);border-radius:14px;padding:20px;
  box-shadow:0 1px 4px rgba(0,0,0,.07);border:1px solid var(--border);overflow:hidden}}
.shot-stats{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:14px}}
.stat-chip{{background:var(--bg);border-radius:10px;padding:10px 14px;text-align:center}}
.stat-chip .num{{font-size:24px;font-weight:800}}
.stat-chip .lbl{{font-size:10px;color:var(--gray);font-weight:600;text-transform:uppercase}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
th{{background:#f8fafc;color:var(--gray);font-weight:700;text-align:left;
   padding:8px 10px;border-bottom:2px solid var(--border);font-size:11px;
   text-transform:uppercase;letter-spacing:0.4px}}
td{{padding:7px 10px;border-bottom:1px solid var(--border);vertical-align:middle}}
tr:last-child td{{border-bottom:none}}
tr:hover td{{background:#f8fafc}}
.hand-badge{{display:inline-block;padding:2px 8px;border-radius:20px;font-size:10px;font-weight:700}}
.hand-r{{background:#dbeafe;color:var(--blue)}}
.hand-l{{background:#f3e8ff;color:var(--purple)}}
.side-badge{{display:inline-block;padding:2px 8px;border-radius:20px;font-size:10px}}
.side-s{{background:#dcfce7;color:var(--green)}}
.side-r{{background:#fef3c7;color:var(--amber)}}

/* Recommendations */
.rec-box{{background:var(--card);border-radius:14px;padding:24px;
  box-shadow:0 1px 4px rgba(0,0,0,.07);border:1px solid var(--border);margin-bottom:24px}}
.rec-item{{padding:12px 16px;border-radius:10px;background:var(--bg);
  margin-bottom:10px;line-height:1.6;font-size:13px}}
.rec-item:last-child{{margin-bottom:0}}

/* Footer */
.footer{{text-align:center;color:var(--gray);font-size:11px;padding:12px 0}}

/* Print */
@media print{{
  body{{background:#fff;padding:8px}}
  .card,.chart-box,.court-box,.rec-box,.shot-table-box,.charts-full{{box-shadow:none;border:1px solid #ddd}}
  .charts-row{{page-break-inside:avoid}}
}}
</style>
</head>
<body>
<div class="page">

<!-- Header -->
<div class="header">
  <div>
    <div class="header-logo">🏸 羽毛球 AI 教练系统</div>
    <div class="header-sub">训练分析报告 · Badminton AI Coaching Report</div>
    <div class="badge">YOLOv8n-Pose · ONNX Runtime · 单目视觉分析</div>
  </div>
  <div class="header-meta">
    <div><b>训练视频</b>：{video_name}</div>
    <div><b>时长</b>：{duration_str} &nbsp;|&nbsp; <b>帧率</b>：{fps:.1f} fps</div>
    <div><b>分析帧数</b>：{frame_count} 帧</div>
    <div><b>报告生成</b>：{report_date}</div>
    <div><b>平均推理</b>：{avg_infer_ms:.1f} ms/帧</div>
  </div>
</div>

<!-- Metric Cards -->
<div class="section-title">核心指标</div>
<div class="cards">

  <div class="card arm">
    <div class="card-icon">💪</div>
    <div class="card-label">平均挥臂速度</div>
    <div class="card-value">{avg_arm_kmh}<span class="card-unit">km/h</span></div>
    <div class="card-sub">击球时腕部峰值速度均值</div>
    <div class="card-detail">
      <span class="pill blue">峰值 {max_arm_kmh} km/h</span>
      <span class="pill green">≈ {avg_arm_ms:.1f} m/s</span>
    </div>
  </div>

  <div class="card ball">
    <div class="card-icon">🏸</div>
    <div class="card-label">球速检测</div>
    <div class="card-value">{avg_ball_kmh}<span class="card-unit">km/h</span></div>
    <div class="card-sub">{ball_source}</div>
    <div class="card-detail">
      <span class="pill red">峰值 {max_ball_kmh} km/h</span>
    </div>
  </div>

  <div class="card shot">
    <div class="card-icon">🤚</div>
    <div class="card-label">击球次数 / 手势</div>
    <div class="card-value">{total_shots}<span class="card-unit">次</span></div>
    <div class="card-sub">左/右手击球分布</div>
    <div class="card-detail">
      <span class="pill purple">右 {right_shots} 次</span>
      <span class="pill blue">左 {left_shots} 次</span>
    </div>
  </div>

  <div class="card height">
    <div class="card-icon">📐</div>
    <div class="card-label">平均击球高度</div>
    <div class="card-value">{avg_height_m}<span class="card-unit">m</span></div>
    <div class="card-sub">腕部相对地面实际高度</div>
    <div class="card-detail">
      <span class="pill green">峰值 {max_height_m} m</span>
      <span class="pill blue">网高 1.55 m</span>
    </div>
  </div>

</div>

<!-- Speed Timeline -->
<div class="charts-full">
  <div class="section-title">挥臂速度 · 全程时序（km/h）</div>
  <canvas id="speedChart" height="70"></canvas>
</div>

<!-- Charts Row -->
<div class="charts-row">
  <div class="chart-box">
    <div class="section-title">击球手势分布</div>
    <canvas id="handChart"></canvas>
  </div>
  <div class="chart-box">
    <div class="section-title">击球高度分布（米）</div>
    <canvas id="heightChart"></canvas>
  </div>
</div>

<!-- Court Heatmap + Shot Table -->
<div class="analysis-row">
  <div class="court-box">
    <div class="section-title">站位热图</div>
    {court_svg}
    <div style="font-size:10px;color:var(--gray);margin-top:6px">
      球场 13.40m × 6.10m · 双打<br>
      ▲ 绿=右手 ▲ 紫=左手击球点
    </div>
  </div>
  <div class="shot-table-box">
    <div class="section-title">击球记录明细</div>
    <div class="shot-stats">
      <div class="stat-chip"><div class="num" style="color:var(--blue)">{serve_shots}</div><div class="lbl">发球侧击球</div></div>
      <div class="stat-chip"><div class="num" style="color:var(--amber)">{recv_shots}</div><div class="lbl">接球侧击球</div></div>
      <div class="stat-chip"><div class="num" style="color:var(--green)">{ball_detected}</div><div class="lbl">球体检测帧</div></div>
    </div>
    <div style="max-height:360px;overflow-y:auto">
    <table>
      <thead>
        <tr>
          <th>#</th><th>时间(s)</th><th>主导手</th>
          <th>挥臂(km/h)</th><th>球速(km/h)</th>
          <th>击球高(m)</th><th>站位</th>
        </tr>
      </thead>
      <tbody>{shot_rows}</tbody>
    </table>
    </div>
  </div>
</div>

<!-- Recommendations -->
<div class="rec-box">
  <div class="section-title">AI 训练建议</div>
  {rec_html}
</div>

<!-- Speed Level Bar -->
<div class="charts-full" style="margin-bottom:24px">
  <div class="section-title">击球速度等级分布</div>
  <canvas id="speedLevelChart" height="60"></canvas>
</div>

<div class="footer">
  🏸 羽毛球 AI 教练系统 · 报告由 AI 自动生成 · 仅供训练参考 ·
  球场尺寸参照 BWF 标准（长13.40m × 宽6.10m，网高1.55m）
</div>

</div><!-- /page -->

<script>
// ── 数据注入 ────────────────────────────────────────────────────────────────
const TIME_AXIS = {time_axis_json};
const ARM_SERIES = {arm_series_json};
const HEIGHT_COUNTS = {height_counts_json};
const HEIGHT_LABELS = {height_labels_json};
const SPEED_LEVELS  = {speed_levels_json};
const LEFT_SHOTS    = {left_shots};
const RIGHT_SHOTS   = {right_shots};
const SERVE_SHOTS   = {serve_shots};
const RECV_SHOTS    = {recv_shots};

// ── 调色板 ──────────────────────────────────────────────────────────────────
const C = {{
  blue:'#1d6fa5',cyan:'#06b6d4',green:'#16a34a',
  amber:'#d97706',red:'#dc2626',purple:'#7c3aed',
  gray:'#94a3b8',bg:'#f1f5f9'
}};

// ── 速度时序图 ───────────────────────────────────────────────────────────────
new Chart(document.getElementById('speedChart'),{{
  type:'line',
  data:{{
    labels:TIME_AXIS,
    datasets:[{{
      label:'挥臂速度 (km/h)',
      data:ARM_SERIES,
      borderColor:C.cyan,
      backgroundColor:'rgba(6,182,212,0.08)',
      borderWidth:1.5,
      pointRadius:0,
      tension:0.4,
      fill:true,
    }}]
  }},
  options:{{
    responsive:true,
    plugins:{{legend:{{display:false}},tooltip:{{
      callbacks:{{label:ctx=>` ${{ctx.parsed.y.toFixed(1)}} km/h`}}
    }}}},
    scales:{{
      x:{{title:{{display:true,text:'时间 (秒)'}},
         ticks:{{maxTicksLimit:12}}}},
      y:{{title:{{display:true,text:'km/h'}},min:0,
         grid:{{color:'#f1f5f9'}}}}
    }}
  }}
}});

// ── 手势饼图 ────────────────────────────────────────────────────────────────
new Chart(document.getElementById('handChart'),{{
  type:'doughnut',
  data:{{
    labels:['右手击球','左手击球','发球侧','接球侧'],
    datasets:[
      {{data:[RIGHT_SHOTS,LEFT_SHOTS],
        backgroundColor:[C.blue,C.purple],
        borderWidth:0}},
      {{data:[SERVE_SHOTS,RECV_SHOTS],
        backgroundColor:[C.green,C.amber],
        borderWidth:0}}
    ]
  }},
  options:{{
    responsive:true,cutout:'45%',
    plugins:{{
      legend:{{position:'bottom',labels:{{font:{{size:11}},padding:10}}}},
      tooltip:{{callbacks:{{label:ctx=>
        ` ${{ctx.label}}: ${{ctx.raw}} 次`}}}}
    }}
  }}
}});

// ── 击球高度条形图 ──────────────────────────────────────────────────────────
new Chart(document.getElementById('heightChart'),{{
  type:'bar',
  data:{{
    labels:HEIGHT_LABELS,
    datasets:[{{
      label:'击球次数',
      data:HEIGHT_COUNTS,
      backgroundColor:[
        '#94a3b8','#60a5fa','#34d399','#4ade80',
        '#facc15','#f97316','#ef4444'],
      borderRadius:6,borderWidth:0
    }}]
  }},
  options:{{
    responsive:true,
    plugins:{{legend:{{display:false}}}},
    scales:{{
      x:{{title:{{display:true,text:'击球高度'}}}},
      y:{{title:{{display:true,text:'次数'}},ticks:{{stepSize:1}},
         grid:{{color:'#f1f5f9'}}}}
    }}
  }}
}});

// ── 速度等级图 ──────────────────────────────────────────────────────────────
const SL_LABELS = SPEED_LEVELS.map(x=>x[0]);
const SL_DATA   = SPEED_LEVELS.map(x=>x[1]);
new Chart(document.getElementById('speedLevelChart'),{{
  type:'bar',
  data:{{
    labels:SL_LABELS,
    datasets:[{{
      label:'击球次数',
      data:SL_DATA,
      backgroundColor:[C.gray,C.cyan,C.blue,C.red],
      borderRadius:6,borderWidth:0
    }}]
  }},
  options:{{
    responsive:true,indexAxis:'y',
    plugins:{{legend:{{display:false}}}},
    scales:{{
      x:{{title:{{display:true,text:'次数'}},grid:{{color:'#f1f5f9'}}}},
      y:{{grid:{{display:false}}}}
    }}
  }}
}});
</script>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════════════════════════
# 12. 报告组装
# ═══════════════════════════════════════════════════════════════════════════════

def build_shot_rows(shots):
    rows = []
    for i, s in enumerate(shots[:50]):  # 最多显示50行
        hand_cls  = 'hand-r' if s['hand'] == 'right' else 'hand-l'
        hand_txt  = '右手' if s['hand'] == 'right' else '左手'
        side_cls  = 'side-s' if s['side'] == 'serve_side' else 'side-r'
        side_txt  = '发球侧' if s['side'] == 'serve_side' else '接球侧'
        rows.append(
            f'<tr>'
            f'<td>{i+1}</td>'
            f'<td>{s["time_s"]:.1f}</td>'
            f'<td><span class="hand-badge {hand_cls}">{hand_txt}</span></td>'
            f'<td>{s["arm_speed_kmh"]:.1f}</td>'
            f'<td>{s["ball_speed_kmh"]:.1f}</td>'
            f'<td>{s["height_m"]:.2f}</td>'
            f'<td><span class="side-badge {side_cls}">{side_txt}</span></td>'
            f'</tr>'
        )
    return '\n'.join(rows) if rows else '<tr><td colspan="7" style="text-align:center;color:#999">未检测到击球事件</td></tr>'

def render_html(raw: dict, summary: dict, recs: list) -> str:
    duration_s = raw['duration_s']
    m, s = divmod(int(duration_s), 60)
    duration_str = f"{m}分{s:02d}秒" if m else f"{s}秒"

    court_svg = generate_court_svg(
        summary['court_positions'],
        summary['shots_detail'],
        svg_w=200, svg_h=440
    )

    shot_rows = build_shot_rows(summary['shots_detail'])
    rec_html  = ''.join(f'<div class="rec-item">{r}</div>' for r in recs)

    return HTML_TEMPLATE.format(
        video_name    = raw.get('video_name', '未知'),
        duration_str  = duration_str,
        fps           = raw.get('fps', 30.0),
        frame_count   = raw.get('frame_count', 0),
        report_date   = datetime.now().strftime('%Y-%m-%d %H:%M'),
        avg_infer_ms  = raw.get('avg_infer_ms', 0),

        avg_arm_kmh   = summary['avg_arm_kmh'],
        max_arm_kmh   = summary['max_arm_kmh'],
        avg_arm_ms    = summary['avg_arm_kmh'] / 3.6,
        avg_ball_kmh  = summary['avg_ball_kmh'],
        max_ball_kmh  = summary['max_ball_kmh'],
        ball_source   = summary['ball_source'],
        total_shots   = summary['total_shots'],
        left_shots    = summary['left_shots'],
        right_shots   = summary['right_shots'],
        serve_shots   = summary['serve_shots'],
        recv_shots    = summary['recv_shots'],
        avg_height_m  = summary['avg_height_m'],
        max_height_m  = summary['max_height_m'],
        ball_detected = raw.get('ball_detection_count', 0),

        court_svg     = court_svg,
        shot_rows     = shot_rows,
        rec_html      = rec_html,

        time_axis_json   = json.dumps(summary['time_axis']),
        arm_series_json  = json.dumps([round(v,1) for v in summary['arm_series']]),
        height_counts_json = json.dumps(summary['height_counts']),
        height_labels_json = json.dumps(summary['height_labels']),
        speed_levels_json  = json.dumps(summary['speed_levels']),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 13. 主函数
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="羽毛球训练分析报告生成器")
    p.add_argument('--input', type=str, help="视频文件路径")
    p.add_argument('--demo', action='store_true', help="使用合成演示数据（无需视频/模型）")
    p.add_argument('--model', type=str, default=str(MODEL_PATH),
                   help="YOLOv8n-pose.onnx 路径")
    p.add_argument('--output-dir', type=str, default=str(OUTPUT_DIR),
                   help="报告输出目录")
    p.add_argument('--max-frames', type=int, default=-1,
                   help="最多处理帧数（-1=全部）")
    p.add_argument('--court-corners', type=str, default=None,
                   help="球场四角像素坐标，格式: 'x0,y0 x1,y1 x2,y2 x3,y3'")
    return p.parse_args()

def parse_corners(s):
    """解析 '100,50 620,50 620,1230 100,1230' → [(100,50),...]"""
    pts = []
    for tok in s.strip().split():
        x, y = tok.split(',')
        pts.append((int(x), int(y)))
    assert len(pts) == 4, "需要恰好4个角点"
    return pts

def main():
    args = parse_args()

    corners = None
    if args.court_corners:
        try:
            corners = parse_corners(args.court_corners)
            logger.info(f"手动球场角点: {corners}")
        except Exception as e:
            logger.warning(f"角点解析失败: {e}，使用线性映射")

    # ── 获取原始数据 ──────────────────────────────────────────────────────────
    raw = None

    if not args.demo and args.input:
        logger.info(f"开始处理视频: {args.input}")
        raw = process_video(args.input, args.model,
                            corners_px=corners,
                            max_frames=args.max_frames)
        if raw is None:
            logger.warning("视频处理失败，切换演示模式")

    if raw is None:
        if not args.demo and not args.input:
            print("用法: python generate_training_report.py --input video.mp4")
            print("      python generate_training_report.py --demo")
            sys.exit(1)
        logger.info("使用演示数据生成报告...")
        raw = generate_demo_data()

    # ── 计算摘要 + 建议 ───────────────────────────────────────────────────────
    summary = compute_summary(raw)
    recs    = generate_recommendations(summary, raw)

    # ── 生成 HTML ────────────────────────────────────────────────────────────
    html = render_html(raw, summary, recs)

    # ── 写出文件 ─────────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"training_report_{ts}.html"
    out_path.write_text(html, encoding='utf-8')

    size_kb = out_path.stat().st_size / 1024
    print()
    print("=" * 60)
    print(f"  ✅ 训练报告生成成功！")
    print(f"  📄 文件: {out_path}")
    print(f"  📦 大小: {size_kb:.1f} KB")
    print(f"  🏸 总击球次数: {summary['total_shots']}")
    print(f"  💪 平均挥臂速度: {summary['avg_arm_kmh']} km/h")
    print(f"  🏸 估算球速均值: {summary['avg_ball_kmh']} km/h")
    print(f"  📐 平均击球高度: {summary['avg_height_m']} m")
    print(f"  🤚 右手 {summary['right_shots']} 次 / 左手 {summary['left_shots']} 次")
    print("=" * 60)
    print()
    print("  在浏览器中打开 HTML 文件，Ctrl+P 可直接打印为 PDF。")
    print()

    return out_path

if __name__ == "__main__":
    main()
