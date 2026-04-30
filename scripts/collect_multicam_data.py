#!/usr/bin/env python3
"""
collect_multicam_data.py
========================
多摄像头同步数据采集脚本

功能：
  1. 支持 1~4 路摄像头同步采集
  2. 自动帧同步（时间戳记录）
  3. 触发式录制（手动/键盘/传感器触发）
  4. 自动存储帧、标注模板、元数据
  5. 麦克风音频同步采集

目录结构（采集后）：
  data/raw/session_YYYYMMDD_HHMMSS/
    ├── cam0/               # 左摄像头帧
    │   ├── 000000.jpg
    │   └── ...
    ├── cam1/               # 右摄像头帧
    ├── audio/              # 音频录制
    │   └── recording.wav
    ├── sync_log.json       # 时间戳同步日志
    └── session_meta.json   # 采集会话元数据

用法：
    # 单摄像头采集
    python collect_multicam_data.py --cams 0 --duration 30

    # 双目采集（摄像头 0 和 1）
    python collect_multicam_data.py --cams 0 1 --duration 60

    # 带音频的双目采集
    python collect_multicam_data.py --cams 0 1 --audio --duration 60

    # 触发式采集（按空格开始/停止）
    python collect_multicam_data.py --cams 0 1 --trigger-mode keyboard
"""

import argparse
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.error("需要 OpenCV: pip install opencv-python")

# =============================================================================
# 配置
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJECT_ROOT / "data" / "raw"

DEFAULT_FPS = 30
DEFAULT_RESOLUTION = (1280, 720)
DEFAULT_QUALITY = 90  # JPEG 质量


# =============================================================================
# 多摄像头采集器
# =============================================================================

class MultiCamCollector:
    """多路摄像头同步采集"""

    def __init__(
        self,
        camera_ids: list,
        fps: int = DEFAULT_FPS,
        resolution: tuple = DEFAULT_RESOLUTION,
        output_dir: Path = None,
        record_audio: bool = False,
    ):
        self.camera_ids = camera_ids
        self.fps = fps
        self.resolution = resolution
        self.record_audio = record_audio

        # 会话目录
        session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_dir = output_dir or (DATA_ROOT / session_name)
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # 各摄像头目录
        self.cam_dirs = []
        for i, cam_id in enumerate(camera_ids):
            cam_dir = self.session_dir / f"cam{i}"
            cam_dir.mkdir(exist_ok=True)
            self.cam_dirs.append(cam_dir)

        # 音频目录
        if record_audio:
            (self.session_dir / "audio").mkdir(exist_ok=True)

        self.caps = []
        self.is_recording = False
        self.sync_log = []
        self.frame_counters = [0] * len(camera_ids)

        # 元数据
        self.session_meta = {
            "session_dir": str(self.session_dir),
            "camera_ids": camera_ids,
            "fps": fps,
            "resolution": list(resolution),
            "record_audio": record_audio,
            "start_time": None,
            "end_time": None,
            "total_frames_per_cam": [],
            "notes": "",
        }

    def open_cameras(self) -> bool:
        """打开所有摄像头"""
        if not CV2_AVAILABLE:
            logger.error("需要 OpenCV")
            return False

        for cam_id in self.camera_ids:
            cap = cv2.VideoCapture(cam_id)
            if not cap.isOpened():
                logger.error(f"无法打开摄像头 {cam_id}")
                self.close_cameras()
                return False

            # 设置参数
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲延迟

            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            logger.info(f"摄像头 {cam_id}: {actual_w}x{actual_h} @ {actual_fps:.1f}fps")

            self.caps.append(cap)

        logger.info(f"成功打开 {len(self.caps)} 路摄像头")
        return True

    def close_cameras(self):
        for cap in self.caps:
            cap.release()
        self.caps = []

    def capture_frame_sync(self) -> tuple:
        """
        同步采集所有摄像头帧
        返回：(frames_list, timestamp)
        """
        timestamp = time.perf_counter()
        frames = []
        success = True

        for cap in self.caps:
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                frames.append(None)
                success = False

        return frames, timestamp, success

    def record(self, duration: float = -1, trigger_mode: str = "auto"):
        """
        开始录制

        trigger_mode:
          "auto": 自动录制 duration 秒
          "keyboard": 按空格开始/停止
        """
        if not self.open_cameras():
            return

        self.session_meta["start_time"] = datetime.now().isoformat()
        logger.info(f"会话目录: {self.session_dir}")

        if trigger_mode == "keyboard" and CV2_AVAILABLE:
            logger.info("按 [空格] 开始/停止录制，按 [q] 退出")
            self._record_with_keyboard(duration)
        else:
            logger.info(f"自动录制 {duration:.0f} 秒...")
            self.is_recording = True
            self._record_loop(duration)

        self.session_meta["end_time"] = datetime.now().isoformat()
        self.session_meta["total_frames_per_cam"] = self.frame_counters
        self._save_metadata()
        self.close_cameras()

        logger.info("录制完成！")
        logger.info(f"总帧数（每路摄像头）: {self.frame_counters}")
        logger.info(f"数据保存在: {self.session_dir}")

    def _record_loop(self, duration: float):
        """核心录制循环"""
        t_start = time.perf_counter()
        frame_interval = 1.0 / self.fps
        next_frame_time = t_start

        while True:
            if duration > 0 and (time.perf_counter() - t_start) >= duration:
                break
            if not self.is_recording:
                break

            current_time = time.perf_counter()

            # 等待到下一帧时间（控制帧率）
            if current_time < next_frame_time:
                time.sleep(max(0, next_frame_time - current_time - 0.001))

            frames, timestamp, success = self.capture_frame_sync()

            if success or any(f is not None for f in frames):
                # 保存各摄像头帧
                sync_entry = {
                    "timestamp": timestamp - t_start,
                    "frame_indices": [],
                }

                for i, (frame, cam_dir) in enumerate(zip(frames, self.cam_dirs)):
                    if frame is not None:
                        frame_idx = self.frame_counters[i]
                        filename = f"{frame_idx:06d}.jpg"
                        filepath = cam_dir / filename
                        cv2.imwrite(
                            str(filepath), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, DEFAULT_QUALITY]
                        )
                        self.frame_counters[i] += 1
                        sync_entry["frame_indices"].append(frame_idx)
                    else:
                        sync_entry["frame_indices"].append(-1)  # 丢帧标记

                self.sync_log.append(sync_entry)

                # 控制台进度（每 5 秒打印一次）
                elapsed = timestamp - t_start
                if len(self.sync_log) % (self.fps * 5) == 0:
                    logger.info(f"录制中... {elapsed:.0f}s / {duration:.0f}s, "
                                f"帧数: {self.frame_counters}")

            next_frame_time = t_start + len(self.sync_log) * frame_interval

    def _record_with_keyboard(self, max_duration: float):
        """键盘触发录制"""
        if not CV2_AVAILABLE:
            return

        preview_frame = None
        last_capture = None

        while True:
            # 采集预览帧
            frames, timestamp, _ = self.capture_frame_sync()
            if frames and frames[0] is not None:
                preview_frame = frames[0].copy()
                last_capture = (frames, timestamp)

                # 显示预览
                h, w = preview_frame.shape[:2]
                status = "REC" if self.is_recording else "READY"
                color = (0, 0, 255) if self.is_recording else (0, 200, 0)
                cv2.circle(preview_frame, (30, 30), 15, color, -1)
                cv2.putText(preview_frame, status, (50, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(preview_frame,
                            f"Frames: {self.frame_counters[0]}",
                            (10, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # 如果多摄像头，显示拼接预览
                if len(frames) > 1 and frames[1] is not None:
                    preview2 = cv2.resize(frames[1], (w//2, h//2))
                    preview_small = cv2.resize(preview_frame, (w//2, h//2))
                    combined = np.hstack([preview_small, preview2])
                    cv2.imshow("Multi-Cam Capture - SPACE=Record Q=Quit", combined)
                else:
                    cv2.imshow("Multi-Cam Capture - SPACE=Record Q=Quit", preview_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                self.is_recording = not self.is_recording
                if self.is_recording:
                    logger.info("开始录制...")
                else:
                    logger.info("停止录制")

            elif key == ord("q"):
                self.is_recording = False
                break

            # 录制帧保存
            if self.is_recording and last_capture:
                frames_to_save, ts = last_capture
                sync_entry = {"timestamp": ts, "frame_indices": []}
                for i, (frame, cam_dir) in enumerate(zip(frames_to_save, self.cam_dirs)):
                    if frame is not None:
                        frame_idx = self.frame_counters[i]
                        cv2.imwrite(
                            str(cam_dir / f"{frame_idx:06d}.jpg"),
                            frame,
                            [cv2.IMWRITE_JPEG_QUALITY, DEFAULT_QUALITY]
                        )
                        self.frame_counters[i] += 1
                        sync_entry["frame_indices"].append(frame_idx)
                self.sync_log.append(sync_entry)
                last_capture = None

        cv2.destroyAllWindows()

    def _save_metadata(self):
        """保存会话元数据和同步日志"""
        # 同步日志
        sync_path = self.session_dir / "sync_log.json"
        with open(str(sync_path), "w") as f:
            json.dump(self.sync_log, f, indent=2)

        # 会话元数据
        meta_path = self.session_dir / "session_meta.json"
        with open(str(meta_path), "w") as f:
            json.dump(self.session_meta, f, indent=2, ensure_ascii=False)

        logger.info(f"元数据已保存: {meta_path}")

        # 生成标注模板
        self._generate_annotation_template()

    def _generate_annotation_template(self):
        """为采集的数据生成标注模板目录"""
        anno_dir = self.session_dir / "annotations"
        anno_dir.mkdir(exist_ok=True)

        # 生成说明文件
        readme = {
            "说明": "请为每帧添加标注，填写 annotations/ 目录下的 JSON 文件",
            "工具推荐": "CVAT, LabelImg, RoboFlow",
            "标注格式": "COCO JSON（见 export_json_template.py 中的模板）",
            "关键点定义": {
                "0": "nose", "1": "left_eye", "2": "right_eye",
                "3": "left_ear", "4": "right_ear",
                "5": "left_shoulder", "6": "right_shoulder",
                "7": "left_elbow", "8": "right_elbow",
                "9": "left_wrist", "10": "right_wrist",
                "11": "left_hip", "12": "right_hip",
                "13": "left_knee", "14": "right_knee",
                "15": "left_ankle", "16": "right_ankle",
            },
            "羽毛球标注字段": {
                "x": "球心 x 坐标（像素）",
                "y": "球心 y 坐标（像素）",
                "r": "球半径（像素，近似值）",
                "visibility": "0=不可见 1=被遮挡 2=完全可见",
                "speed_kmh": "球速（如已知）",
            }
        }
        with open(str(anno_dir / "ANNOTATION_GUIDE.json"), "w", encoding="utf-8") as f:
            json.dump(readme, f, indent=2, ensure_ascii=False)

        logger.info(f"标注指南已生成: {anno_dir}/ANNOTATION_GUIDE.json")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="多摄像头同步数据采集工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--cams", type=int, nargs="+", default=[0],
                        help="摄像头 ID 列表（例如: --cams 0 1）")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--width", type=int, default=DEFAULT_RESOLUTION[0])
    parser.add_argument("--height", type=int, default=DEFAULT_RESOLUTION[1])
    parser.add_argument("--duration", type=float, default=30.0, help="录制时长（秒）")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--audio", action="store_true", help="同时录制音频")
    parser.add_argument("--trigger-mode", choices=["auto", "keyboard"],
                        default="auto", help="触发模式")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅显示配置，不实际录制（测试用）")

    args = parser.parse_args()

    if not CV2_AVAILABLE:
        logger.error("需要 OpenCV: pip install opencv-python")
        logger.info("模拟运行模式（无实际采集）...")

        # 输出配置说明
        logger.info("=" * 50)
        logger.info("采集配置:")
        logger.info(f"  摄像头: {args.cams}")
        logger.info(f"  分辨率: {args.width}x{args.height} @ {args.fps}fps")
        logger.info(f"  时长: {args.duration}s")
        logger.info(f"  录音: {args.audio}")
        logger.info(f"  存储: {DATA_ROOT}")
        logger.info("=" * 50)
        return

    output_dir = Path(args.output_dir) if args.output_dir else None
    resolution = (args.width, args.height)

    collector = MultiCamCollector(
        camera_ids=args.cams,
        fps=args.fps,
        resolution=resolution,
        output_dir=output_dir,
        record_audio=args.audio,
    )

    logger.info("=" * 50)
    logger.info("采集配置:")
    logger.info(f"  摄像头: {args.cams}")
    logger.info(f"  分辨率: {args.width}x{args.height} @ {args.fps}fps")
    logger.info(f"  时长: {args.duration}s")
    logger.info(f"  触发方式: {args.trigger_mode}")
    logger.info(f"  会话目录: {collector.session_dir}")
    logger.info("=" * 50)

    if args.dry_run:
        logger.info("--dry-run 模式，不实际录制")
        return

    collector.record(duration=args.duration, trigger_mode=args.trigger_mode)


if __name__ == "__main__":
    main()
