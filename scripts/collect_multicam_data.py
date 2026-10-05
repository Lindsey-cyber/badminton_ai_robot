#!/usr/bin/env python3
"""
collect_multicam_data.py
========================
Sequential multi-camera frame collector (hardware validation pending).

It opens the requested OpenCV cameras and reads them one after another.
The logged timestamp is taken before those reads; this is NOT hardware-synced
stereo capture. Keyboard and duration-based recording save JPEGs and metadata.
Audio capture and sensor triggering are not implemented.

Output layout:
  data/raw/session_YYYYMMDD_HHMMSS/
    ├── cam0/               # first camera frames
    │   ├── 000000.jpg
    │   └── ...
    ├── cam1/               # second camera frames
    ├── sync_log.json       # approximate read-loop timestamps
    └── session_meta.json   # session metadata

Usage:
    # One camera
    python collect_multicam_data.py --cams 0 --duration 30

    # Two sequentially read cameras
    python collect_multicam_data.py --cams 0 1 --duration 60

    # Press space to start/stop and q to exit
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
    logger.error("OpenCV is required: pip install opencv-python")

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJECT_ROOT / "data" / "raw"

DEFAULT_FPS = 30
DEFAULT_RESOLUTION = (1280, 720)
DEFAULT_QUALITY = 90  # JPEG quality.


# =============================================================================
# Multi-camera collector
# =============================================================================

class MultiCamCollector:
    """Read cameras sequentially and save frames; no hardware synchronization."""

    def __init__(
        self,
        camera_ids: list,
        fps: int = DEFAULT_FPS,
        resolution: tuple = DEFAULT_RESOLUTION,
        output_dir: Path = None,
    ):
        self.camera_ids = camera_ids
        self.fps = fps
        self.resolution = resolution

        # Session directory.
        session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_dir = output_dir or (DATA_ROOT / session_name)
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Per-camera directories.
        self.cam_dirs = []
        for i, cam_id in enumerate(camera_ids):
            cam_dir = self.session_dir / f"cam{i}"
            cam_dir.mkdir(exist_ok=True)
            self.cam_dirs.append(cam_dir)

        self.caps = []
        self.is_recording = False
        self.sync_log = []
        self.frame_counters = [0] * len(camera_ids)

        # Session metadata.
        self.session_meta = {
            "session_dir": str(self.session_dir),
            "camera_ids": camera_ids,
            "fps": fps,
            "resolution": list(resolution),
            "start_time": None,
            "end_time": None,
            "total_frames_per_cam": [],
            "notes": "",
        }

    def open_cameras(self) -> bool:
        """Open all requested cameras."""
        if not CV2_AVAILABLE:
            logger.error("OpenCV is required")
            return False

        for cam_id in self.camera_ids:
            cap = cv2.VideoCapture(cam_id)
            if not cap.isOpened():
                logger.error(f"Cannot open camera {cam_id}")
                self.close_cameras()
                return False

            # Request capture parameters; the device may choose different values.
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Best-effort buffer request.

            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            logger.info(f"Camera {cam_id}: {actual_w}x{actual_h} @ {actual_fps:.1f}fps")

            self.caps.append(cap)

        logger.info(f"Opened {len(self.caps)} cameras")
        return True

    def close_cameras(self):
        for cap in self.caps:
            cap.release()
        self.caps = []

    def capture_frames(self) -> tuple:
        """
        Read each camera in order; the timestamp precedes the first read.
        Return (frames, loop_start_timestamp, all_succeeded).
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
        Start recording.

        trigger_mode:
          "auto": record for duration seconds.
          "keyboard": press space to toggle recording.
        """
        if not self.open_cameras():
            return

        self.session_meta["start_time"] = datetime.now().isoformat()
        logger.info(f"Session directory: {self.session_dir}")

        if trigger_mode == "keyboard" and CV2_AVAILABLE:
            logger.info("Press space to start/stop recording; press q to exit")
            self._record_with_keyboard(duration)
        else:
            logger.info(f"Recording for {duration:.0f} seconds")
            self.is_recording = True
            self._record_loop(duration)

        self.session_meta["end_time"] = datetime.now().isoformat()
        self.session_meta["total_frames_per_cam"] = self.frame_counters
        self._save_metadata()
        self.close_cameras()

        logger.info("Recording complete")
        logger.info(f"Frame counts per camera: {self.frame_counters}")
        logger.info(f"Data saved in: {self.session_dir}")

    def _record_loop(self, duration: float):
        """Duration-based recording loop."""
        t_start = time.perf_counter()
        frame_interval = 1.0 / self.fps
        next_frame_time = t_start

        while True:
            if duration > 0 and (time.perf_counter() - t_start) >= duration:
                break
            if not self.is_recording:
                break

            current_time = time.perf_counter()

            # Wait for the next requested frame time.
            if current_time < next_frame_time:
                time.sleep(max(0, next_frame_time - current_time - 0.001))

            frames, timestamp, success = self.capture_frames()

            if success or any(f is not None for f in frames):
                # Save available camera frames.
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
                        sync_entry["frame_indices"].append(-1)  # Failed read marker.

                self.sync_log.append(sync_entry)

                # Print progress approximately every five seconds.
                elapsed = timestamp - t_start
                if len(self.sync_log) % (self.fps * 5) == 0:
                    logger.info(f"Recording: {elapsed:.0f}s / {duration:.0f}s, "
                                f"frame counts: {self.frame_counters}")

            next_frame_time = t_start + len(self.sync_log) * frame_interval

    def _record_with_keyboard(self, max_duration: float):
        """Toggle recording with the keyboard preview."""
        if not CV2_AVAILABLE:
            return

        preview_frame = None
        last_capture = None

        while True:
            # Capture preview frames.
            frames, timestamp, _ = self.capture_frames()
            if frames and frames[0] is not None:
                preview_frame = frames[0].copy()
                last_capture = (frames, timestamp)

                # Show the preview.
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

                # Show both cameras side by side when available.
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
                    logger.info("Recording started")
                else:
                    logger.info("Recording stopped")

            elif key == ord("q"):
                self.is_recording = False
                break

            # Save recorded frames.
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
        """Save session metadata and approximate read-loop timestamps."""
        # Read-loop log.
        sync_path = self.session_dir / "sync_log.json"
        with open(str(sync_path), "w") as f:
            json.dump(self.sync_log, f, indent=2)

        # Session metadata.
        meta_path = self.session_dir / "session_meta.json"
        with open(str(meta_path), "w") as f:
            json.dump(self.session_meta, f, indent=2, ensure_ascii=False)

        logger.info(f"Metadata saved: {meta_path}")

        # Generate an annotation guide.
        self._generate_annotation_template()

    def _generate_annotation_template(self):
        """Create an annotation guide for collected frames."""
        anno_dir = self.session_dir / "annotations"
        anno_dir.mkdir(exist_ok=True)

        # Guide only; the captured frames have not been labeled.
        readme = {
            "description": "Frames are unlabeled. See docs/shuttle_dataset.md for YOLO labels and splits.",
            "suggested_tools": "CVAT, LabelImg, Roboflow",
            "pose_keypoints": {
                "0": "nose", "1": "left_eye", "2": "right_eye",
                "3": "left_ear", "4": "right_ear",
                "5": "left_shoulder", "6": "right_shoulder",
                "7": "left_elbow", "8": "right_elbow",
                "9": "left_wrist", "10": "right_wrist",
                "11": "left_hip", "12": "right_hip",
                "13": "left_knee", "14": "right_knee",
                "15": "left_ankle", "16": "right_ankle",
            },
            "shuttlecock_annotation": {
                "format": "class 0 center_x center_y width height (normalized YOLO box)",
                "visibility": "Record occlusion separately when known; not a training label here",
            }
        }
        with open(str(anno_dir / "ANNOTATION_GUIDE.json"), "w", encoding="utf-8") as f:
            json.dump(readme, f, indent=2, ensure_ascii=False)

        logger.info(f"Annotation guide saved: {anno_dir}/ANNOTATION_GUIDE.json")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Sequential multi-camera frame collector (not synchronized stereo)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--cams", type=int, nargs="+", default=[0],
                        help="Camera IDs, for example --cams 0 1")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--width", type=int, default=DEFAULT_RESOLUTION[0])
    parser.add_argument("--height", type=int, default=DEFAULT_RESOLUTION[1])
    parser.add_argument("--duration", type=float, default=30.0, help="Recording duration in seconds")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--trigger-mode", choices=["auto", "keyboard"],
                        default="auto", help="Recording trigger mode")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print configuration without opening cameras or creating files")

    args = parser.parse_args()

    if not 1 <= len(args.cams) <= 4 or len(set(args.cams)) != len(args.cams):
        parser.error("Provide 1 to 4 distinct camera IDs")
    if args.fps < 1 or args.width < 1 or args.height < 1 or args.duration <= 0:
        parser.error("FPS, resolution and duration must be positive")

    output_dir = Path(args.output_dir) if args.output_dir else None
    resolution = (args.width, args.height)

    logger.info("=" * 50)
    logger.info("Capture configuration:")
    logger.info(f"  Camera IDs: {args.cams}")
    logger.info(f"  Resolution: {args.width}x{args.height} @ {args.fps}fps")
    logger.info(f"  Duration: {args.duration}s")
    logger.info(f"  Trigger: {args.trigger_mode}")
    logger.info(f"  Output root: {output_dir or DATA_ROOT}")
    logger.info("=" * 50)

    if args.dry_run:
        logger.info("Dry run: no camera opened and no directory created")
        return

    if not CV2_AVAILABLE:
        parser.error("OpenCV is required: pip install opencv-python")
    collector = MultiCamCollector(
        camera_ids=args.cams, fps=args.fps, resolution=resolution, output_dir=output_dir,
    )
    collector.record(duration=args.duration, trigger_mode=args.trigger_mode)


if __name__ == "__main__":
    main()
