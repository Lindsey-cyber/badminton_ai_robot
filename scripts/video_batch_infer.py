#!/usr/bin/env python3
"""
video_batch_infer.py
====================
批量视频推理脚本（姿态 + 球体检测）

功能：
  - 对目录内所有视频批量运行推理
  - 生成汇总 benchmark 报告
  - 支持仅姿态 / 仅检测 / 联合推理
  - 输出 JSON 汇总报告

用法：
    python video_batch_infer.py --input-dir assets/demo_inputs/ --mode pose
    python video_batch_infer.py --input-dir assets/demo_inputs/ --mode detect
    python video_batch_infer.py --input-dir assets/demo_inputs/ --mode both
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
BENCHMARK_DIR = OUTPUT_DIR / "benchmarks"

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def find_videos(input_dir: str) -> list:
    """递归查找所有视频文件"""
    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(Path(input_dir).rglob(f"*{ext}"))
    return sorted(videos)


def batch_infer(
    input_dir: str,
    mode: str = "pose",
    max_frames_per_video: int = 100,
    pose_model_path: str = None,
    detect_model_path: str = None,
) -> dict:
    """
    批量推理并汇总结果

    mode: "pose" | "detect" | "both"
    """
    videos = find_videos(input_dir)
    if not videos:
        logger.warning(f"在 {input_dir} 中未找到视频文件")
        logger.info(f"支持的格式: {VIDEO_EXTENSIONS}")
        return {}

    logger.info(f"找到 {len(videos)} 个视频文件")

    # 动态导入推理脚本（避免循环依赖）
    sys.path.insert(0, str(SCRIPT_DIR))

    pose_model = None
    detect_model = None

    if mode in ["pose", "both"] and pose_model_path:
        try:
            from demo_pose_inference import YOLOv8PoseInference, download_model as dl_pose
            from pathlib import Path as P
            if dl_pose(P(pose_model_path)):
                pose_model = YOLOv8PoseInference(pose_model_path)
                logger.info("姿态模型加载成功")
        except Exception as e:
            logger.warning(f"姿态模型加载失败: {e}")

    if mode in ["detect", "both"] and detect_model_path:
        try:
            from demo_badminton_detection import YOLOv8Detector, download_model as dl_det
            from pathlib import Path as P
            if dl_det(P(detect_model_path)):
                detect_model = YOLOv8Detector(detect_model_path)
                logger.info("检测模型加载成功")
        except Exception as e:
            logger.warning(f"检测模型加载失败: {e}")

    if pose_model is None and detect_model is None:
        logger.warning("未加载任何模型，将仅分析视频基本信息")

    # 批量处理
    results = []
    total_start = time.perf_counter()

    for video_path in videos:
        logger.info(f"处理: {video_path.name}")
        video_result = process_single_video(
            video_path, pose_model, detect_model, mode, max_frames_per_video
        )
        results.append(video_result)

    total_elapsed = time.perf_counter() - total_start

    # 汇总统计
    summary = {
        "total_videos": len(videos),
        "mode": mode,
        "max_frames_per_video": max_frames_per_video,
        "total_time_s": total_elapsed,
        "per_video_results": results,
    }

    if results:
        avg_infer_ms = [r.get("avg_infer_ms", 0) for r in results if r.get("avg_infer_ms")]
        if avg_infer_ms:
            summary["overall_avg_infer_ms"] = float(np.mean(avg_infer_ms))
            summary["overall_avg_fps"] = 1000.0 / np.mean(avg_infer_ms)

    # 保存报告
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    report_path = BENCHMARK_DIR / f"batch_benchmark_{mode}.json"
    with open(str(report_path), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"批量推理完成，报告已保存: {report_path}")
    logger.info(f"总耗时: {total_elapsed:.1f}s，共处理 {len(videos)} 个视频")

    return summary


def process_single_video(
    video_path: Path,
    pose_model,
    detect_model,
    mode: str,
    max_frames: int
) -> dict:
    """处理单个视频，返回统计结果"""
    try:
        import cv2
    except ImportError:
        logger.error("需要 opencv-python")
        return {"file": str(video_path), "error": "OpenCV not available"}

    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"file": str(video_path), "error": "Cannot open video"}

        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        infer_times = []
        num_persons_list = []
        num_balls_list = []
        frame_idx = 0

        while frame_idx < min(max_frames, total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.perf_counter()

            persons = []
            balls = []

            if pose_model and mode in ["pose", "both"]:
                try:
                    persons, _ = pose_model.inference(frame)
                except Exception:
                    pass

            if detect_model and mode in ["detect", "both"]:
                try:
                    balls, _ = detect_model.inference(frame)
                except Exception:
                    pass

            t1 = time.perf_counter()
            infer_times.append((t1 - t0) * 1000)
            num_persons_list.append(len(persons))
            num_balls_list.append(len(balls))
            frame_idx += 1

        cap.release()

        result = {
            "file": str(video_path),
            "video_info": {
                "fps": fps,
                "width": w,
                "height": h,
                "total_frames": total_frames,
            },
            "processed_frames": frame_idx,
            "avg_infer_ms": float(np.mean(infer_times)) if infer_times else 0,
            "avg_fps": 1000.0 / np.mean(infer_times) if infer_times else 0,
            "avg_persons_per_frame": float(np.mean(num_persons_list)) if num_persons_list else 0,
            "avg_balls_per_frame": float(np.mean(num_balls_list)) if num_balls_list else 0,
        }
        return result

    except Exception as e:
        return {"file": str(video_path), "error": str(e)}


def generate_summary_report(benchmark_path: str) -> str:
    """从 benchmark JSON 生成可读报告"""
    with open(benchmark_path) as f:
        data = json.load(f)

    lines = [
        "=" * 60,
        "批量推理 Benchmark 报告",
        "=" * 60,
        f"模式: {data.get('mode', 'unknown')}",
        f"视频数量: {data.get('total_videos', 0)}",
        f"总耗时: {data.get('total_time_s', 0):.1f}s",
        f"平均推理延迟: {data.get('overall_avg_infer_ms', 0):.1f}ms",
        f"平均推理 FPS: {data.get('overall_avg_fps', 0):.1f}",
        "",
        "各视频详情:",
    ]

    for vr in data.get("per_video_results", []):
        fname = Path(vr["file"]).name
        if "error" in vr:
            lines.append(f"  ✗ {fname}: {vr['error']}")
        else:
            lines.append(f"  ✓ {fname}: {vr['avg_infer_ms']:.1f}ms/帧, "
                         f"{vr['avg_fps']:.1f}fps, "
                         f"人={vr['avg_persons_per_frame']:.1f}, "
                         f"球={vr['avg_balls_per_frame']:.1f}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="批量视频推理工具")
    parser.add_argument("--input-dir", "-i", type=str,
                        default=str(PROJECT_ROOT / "assets" / "demo_inputs"),
                        help="输入视频目录")
    parser.add_argument("--mode", choices=["pose", "detect", "both"],
                        default="pose", help="推理模式")
    parser.add_argument("--max-frames", type=int, default=100,
                        help="每个视频最多处理帧数（-1=全部）")
    parser.add_argument("--pose-model", type=str,
                        default=str(PROJECT_ROOT / "src" / "pose" / "yolov8n-pose.onnx"))
    parser.add_argument("--detect-model", type=str,
                        default=str(PROJECT_ROOT / "src" / "perception" / "yolov8n.onnx"))
    parser.add_argument("--report", type=str, help="从已有 benchmark JSON 生成可读报告")

    args = parser.parse_args()

    if args.report:
        report = generate_summary_report(args.report)
        print(report)
        return

    summary = batch_infer(
        input_dir=args.input_dir,
        mode=args.mode,
        max_frames_per_video=args.max_frames,
        pose_model_path=args.pose_model,
        detect_model_path=args.detect_model,
    )

    if summary:
        print("\n" + "=" * 60)
        print("汇总:")
        print(f"  视频数: {summary.get('total_videos', 0)}")
        print(f"  平均推理: {summary.get('overall_avg_infer_ms', 'N/A'):.1f}ms")
        print(f"  平均 FPS: {summary.get('overall_avg_fps', 'N/A'):.1f}")
        print("=" * 60)


if __name__ == "__main__":
    main()
