#!/usr/bin/env python3
"""
stereo_distance_demo.py
=======================
双目立体视觉测距 Demo

功能：
  1. 双目相机标定（棋盘格方法）
  2. 图像校正（立体校正）
  3. 视差图计算（StereoBM / SGBM）
  4. 3D 点云重建（Q 矩阵反投影）
  5. 选定目标点的距离测量
  6. 与球体检测联动，输出球的 3D 坐标和距离

用法：
  # 标定模式（需要采集 20+ 对棋盘格图像）
  python stereo_distance_demo.py --mode calibrate --left-dir data/left --right-dir data/right

  # 测距模式（已标定）
  python stereo_distance_demo.py --mode measure --left left.jpg --right right.jpg

  # 实时测距
  python stereo_distance_demo.py --mode live --calib stereo_calib.json

  # 用模拟数据演示（无需真实双目摄像头）
  python stereo_distance_demo.py --mode simulate
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("[WARNING] opencv-python 未安装，仅支持模拟模式")

from PIL import Image, ImageDraw

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
CALIB_DIR = PROJECT_ROOT / "src" / "perception"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"
DEFAULT_CALIB_PATH = CALIB_DIR / "stereo_calib.json"


# =============================================================================
# 标定参数（示例值，实际使用时需替换为真实标定结果）
# =============================================================================

EXAMPLE_CALIB = {
    "description": "示例双目标定参数（非真实值，仅用于演示）",
    "image_size": [1280, 720],
    "baseline_m": 0.12,          # 基线距离：12cm
    "focal_length_px": 800.0,    # 焦距（像素）
    "cx_left": 640.0,            # 主点 x（左）
    "cy_left": 360.0,            # 主点 y（左）
    "cx_right": 640.0,
    "cy_right": 360.0,
    "camera_matrix_left": [
        [800.0, 0, 640.0],
        [0, 800.0, 360.0],
        [0, 0, 1]
    ],
    "camera_matrix_right": [
        [800.0, 0, 640.0],
        [0, 800.0, 360.0],
        [0, 0, 1]
    ],
    "dist_coeffs_left": [0.0, 0.0, 0.0, 0.0, 0.0],
    "dist_coeffs_right": [0.0, 0.0, 0.0, 0.0, 0.0],
    "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # 旋转矩阵
    "T": [-0.12, 0, 0],                        # 平移向量（基线）
    "Q_matrix": [
        [1, 0, 0, -640.0],
        [0, 1, 0, -360.0],
        [0, 0, 0,  800.0],
        [0, 0, 8.333, 0]   # 1/baseline_px = 1/(f*B) -> 约 1/(800*0.12)=~8.333
    ]
}


# =============================================================================
# 理论工具函数
# =============================================================================

def disparity_to_depth(disparity_px: float, focal_length_px: float, baseline_m: float) -> float:
    """
    视差转深度（三角测距公式）
    depth = focal_length * baseline / disparity
    """
    if disparity_px <= 0:
        return float("inf")
    return focal_length_px * baseline_m / disparity_px


def depth_to_real_coords(
    pixel_x: float, pixel_y: float, depth_m: float,
    cx: float, cy: float, focal_length_px: float
) -> tuple:
    """
    像素坐标 + 深度 → 3D 实体坐标（相机坐标系，单位：米）
    X = (px - cx) * depth / f
    Y = (py - cy) * depth / f
    Z = depth
    """
    X = (pixel_x - cx) * depth_m / focal_length_px
    Y = (pixel_y - cy) * depth_m / focal_length_px
    Z = depth_m
    return X, Y, Z


def speed_from_3d_positions(pos1: tuple, pos2: tuple, dt: float) -> float:
    """两个 3D 位置之间的速度（m/s）"""
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    dz = pos2[2] - pos1[2]
    dist = math.sqrt(dx**2 + dy**2 + dz**2)
    return dist / max(dt, 1e-6)


# =============================================================================
# 标定流程
# =============================================================================

def run_calibration(left_dir: str, right_dir: str, board_size=(9, 6),
                    square_size_m: float = 0.025) -> dict:
    """
    双目标定（棋盘格方法）
    board_size: (cols, rows) 内角点数
    square_size_m: 棋盘格方格边长（米）
    """
    if not CV2_AVAILABLE:
        logger.error("标定需要 OpenCV")
        return {}

    logger.info(f"开始双目标定: 棋盘格 {board_size}, 方格大小 {square_size_m*100:.1f}cm")

    # 准备物体坐标
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size_m

    objpoints = []
    imgpoints_left = []
    imgpoints_right = []

    left_images = sorted(Path(left_dir).glob("*.jpg")) + sorted(Path(left_dir).glob("*.png"))
    right_images = sorted(Path(right_dir).glob("*.jpg")) + sorted(Path(right_dir).glob("*.png"))

    if len(left_images) != len(right_images):
        logger.error(f"左右图像数量不匹配: {len(left_images)} vs {len(right_images)}")
        return {}

    logger.info(f"找到 {len(left_images)} 对图像")
    img_size = None
    valid_pairs = 0

    for lf, rf in zip(left_images, right_images):
        img_l = cv2.imread(str(lf), cv2.IMREAD_GRAYSCALE)
        img_r = cv2.imread(str(rf), cv2.IMREAD_GRAYSCALE)
        if img_l is None or img_r is None:
            continue

        if img_size is None:
            img_size = (img_l.shape[1], img_l.shape[0])

        ret_l, corners_l = cv2.findChessboardCorners(img_l, board_size, None)
        ret_r, corners_r = cv2.findChessboardCorners(img_r, board_size, None)

        if ret_l and ret_r:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_l = cv2.cornerSubPix(img_l, corners_l, (11, 11), (-1, -1), criteria)
            corners_r = cv2.cornerSubPix(img_r, corners_r, (11, 11), (-1, -1), criteria)

            objpoints.append(objp)
            imgpoints_left.append(corners_l)
            imgpoints_right.append(corners_r)
            valid_pairs += 1
            logger.info(f"有效图像对: {lf.name} + {rf.name}")

    if valid_pairs < 5:
        logger.error(f"有效标定对不足（需要 ≥5 对，当前 {valid_pairs} 对）")
        return {}

    logger.info(f"共 {valid_pairs} 对有效图像，开始计算标定参数...")

    # 单目标定
    _, mtx_l, dist_l, _, _ = cv2.calibrateCamera(objpoints, imgpoints_left, img_size, None, None)
    _, mtx_r, dist_r, _, _ = cv2.calibrateCamera(objpoints, imgpoints_right, img_size, None, None)

    # 立体标定
    flags = cv2.CALIB_FIX_INTRINSIC
    ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        mtx_l, dist_l, mtx_r, dist_r, img_size,
        flags=flags
    )

    logger.info(f"立体标定完成，重投影误差: {ret:.4f} pixels")

    # 计算校正映射
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        mtx_l, dist_l, mtx_r, dist_r, img_size, R, T
    )

    # 提取基线和焦距
    baseline_m = float(abs(T[0]))
    focal_length_px = float(P1[0, 0])

    calib_data = {
        "image_size": list(img_size),
        "reprojection_error": float(ret),
        "baseline_m": baseline_m,
        "focal_length_px": focal_length_px,
        "cx_left": float(mtx_l[0, 2]),
        "cy_left": float(mtx_l[1, 2]),
        "camera_matrix_left": mtx_l.tolist(),
        "camera_matrix_right": mtx_r.tolist(),
        "dist_coeffs_left": dist_l.tolist(),
        "dist_coeffs_right": dist_r.tolist(),
        "R": R.tolist(),
        "T": T.flatten().tolist(),
        "Q_matrix": Q.tolist(),
        "R1": R1.tolist(),
        "R2": R2.tolist(),
        "P1": P1.tolist(),
        "P2": P2.tolist(),
    }

    # 保存
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    with open(str(DEFAULT_CALIB_PATH), "w") as f:
        json.dump(calib_data, f, indent=2)
    logger.info(f"标定参数已保存: {DEFAULT_CALIB_PATH}")
    logger.info(f"基线距离: {baseline_m*100:.1f}cm, 焦距: {focal_length_px:.1f}px")

    return calib_data


# =============================================================================
# 测距
# =============================================================================

def compute_disparity_map(img_l_gray, img_r_gray, algorithm="sgbm"):
    """计算视差图"""
    if not CV2_AVAILABLE:
        return None

    if algorithm == "bm":
        stereo = cv2.StereoBM_create(numDisparities=96, blockSize=15)
    else:  # sgbm
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=96,
            blockSize=9,
            P1=8 * 3 * 9**2,
            P2=32 * 3 * 9**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

    disparity = stereo.compute(img_l_gray, img_r_gray).astype(np.float32) / 16.0
    return disparity


def measure_point_depth(disparity_map, point_xy: tuple, calib: dict,
                         window_size: int = 5) -> float:
    """
    测量图像中某点的深度
    point_xy: (x, y) 像素坐标
    返回深度（米）
    """
    x, y = int(point_xy[0]), int(point_xy[1])
    h, w = disparity_map.shape

    # 窗口内取中位数（更鲁棒）
    x1 = max(0, x - window_size // 2)
    x2 = min(w, x + window_size // 2 + 1)
    y1 = max(0, y - window_size // 2)
    y2 = min(h, y + window_size // 2 + 1)

    window = disparity_map[y1:y2, x1:x2]
    valid = window[window > 1.0]  # 过滤无效视差

    if len(valid) == 0:
        return float("inf")

    disparity_median = float(np.median(valid))
    depth = disparity_to_depth(
        disparity_median,
        calib["focal_length_px"],
        calib["baseline_m"]
    )
    return depth


# =============================================================================
# 模拟模式（无需真实双目摄像头）
# =============================================================================

def run_simulation():
    """
    模拟双目测距演示
    演示物理原理和计算流程，无需真实摄像头
    """
    logger.info("=" * 60)
    logger.info("双目测距模拟演示")
    logger.info("=" * 60)

    calib = EXAMPLE_CALIB
    f = calib["focal_length_px"]
    B = calib["baseline_m"]
    cx = calib["cx_left"]
    cy = calib["cy_left"]

    logger.info(f"模拟参数:")
    logger.info(f"  焦距: {f} px")
    logger.info(f"  基线: {B*100:.1f} cm")
    logger.info(f"  主点: ({cx}, {cy})")
    logger.info("")

    # 模拟不同距离的羽毛球
    test_cases = [
        {"distance_m": 1.0, "pos_img_x": 660, "pos_img_y": 350},
        {"distance_m": 2.0, "pos_img_x": 645, "pos_img_y": 355},
        {"distance_m": 3.5, "pos_img_x": 643, "pos_img_y": 358},
        {"distance_m": 5.0, "pos_img_x": 641, "pos_img_y": 359},
        {"distance_m": 7.0, "pos_img_x": 640, "pos_img_y": 360},
    ]

    results = []
    logger.info(f"{'距离(真实)':>12} {'视差(px)':>12} {'距离(估计)':>12} {'误差%':>8} {'X(m)':>8} {'Y(m)':>8}")
    logger.info("-" * 65)

    for case in test_cases:
        true_dist = case["distance_m"]
        px, py = case["pos_img_x"], case["pos_img_y"]

        # 正向：真实距离 → 理论视差
        true_disparity = f * B / true_dist

        # 加入小量噪声（模拟真实测量误差）
        noise = np.random.normal(0, 0.5)
        measured_disparity = true_disparity + noise

        # 反向：视差 → 估计距离
        estimated_dist = disparity_to_depth(measured_disparity, f, B)

        # 3D 坐标
        X, Y, Z = depth_to_real_coords(px, py, estimated_dist, cx, cy, f)

        error_pct = abs(estimated_dist - true_dist) / true_dist * 100

        logger.info(f"{true_dist:>12.2f}m {measured_disparity:>12.2f}px "
                    f"{estimated_dist:>12.3f}m {error_pct:>8.1f}% "
                    f"{X:>8.3f} {Y:>8.3f}")

        results.append({
            "true_dist_m": true_dist,
            "estimated_dist_m": estimated_dist,
            "error_pct": error_pct,
            "X": X, "Y": Y, "Z": Z
        })

    logger.info("")
    avg_error = np.mean([r["error_pct"] for r in results])
    logger.info(f"平均误差: {avg_error:.2f}%")
    logger.info("")

    # 模拟连续追踪速度估计
    logger.info("=" * 60)
    logger.info("模拟连续帧速度估计（羽毛球飞行轨迹）")
    logger.info("=" * 60)

    fps = 60
    dt = 1.0 / fps
    # 模拟羽毛球：从 4m 处以 50 m/s 飞向摄像头
    initial_z = 4.0
    vz = -50.0  # m/s（飞向摄像头）
    vx, vy = 2.0, -1.0

    positions = []
    for i in range(10):
        t = i * dt
        z = initial_z + vz * t - 0.5 * 9.8 * t**2 * 0.1  # 轻微重力影响
        x = vx * t
        y = vy * t
        positions.append((x, y, z))

    logger.info(f"{'帧':>5} {'X(m)':>8} {'Y(m)':>8} {'Z(m)':>8} {'速度(m/s)':>12} {'速度(km/h)':>12}")
    logger.info("-" * 60)

    for i in range(1, len(positions)):
        speed = speed_from_3d_positions(positions[i-1], positions[i], dt)
        logger.info(f"{i:>5} {positions[i][0]:>8.3f} {positions[i][1]:>8.3f} "
                    f"{positions[i][2]:>8.3f} {speed:>12.1f} {speed*3.6:>12.1f}")

    # 生成可视化图
    generate_stereo_diagram()

    # 保存模拟结果
    sim_result = {
        "mode": "simulation",
        "calib_params": {"focal_length_px": f, "baseline_m": B},
        "distance_estimation_tests": results,
        "avg_error_pct": avg_error,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = OUTPUT_DIR / "stereo_simulation_result.json"
    with open(str(result_path), "w") as fp:
        json.dump(sim_result, fp, indent=2, ensure_ascii=False)
    logger.info(f"\n模拟结果已保存: {result_path}")

    return sim_result


def generate_stereo_diagram():
    """生成双目测距原理示意图（PNG）"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 左图：三角测距原理
        ax = axes[0]
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-0.5, 5)
        ax.set_aspect("equal")
        ax.set_title("双目测距原理（三角测量）", fontsize=13)

        # 两个摄像头
        ax.plot([-0.06, 0.06], [0, 0], "bs", markersize=15, label="摄像头")
        ax.text(-0.06, -0.3, "左摄像头", ha="center", fontsize=9)
        ax.text(0.06, -0.3, "右摄像头", ha="center", fontsize=9)

        # 基线
        ax.annotate("", xy=(0.06, 0), xytext=(-0.06, 0),
                    arrowprops=dict(arrowstyle="<->", color="blue"))
        ax.text(0, 0.15, f"基线 B=12cm", ha="center", fontsize=9, color="blue")

        # 目标点（羽毛球）
        target_z = 3.0
        ax.plot(0.2, target_z, "o", color="orange", markersize=12, label="羽毛球")
        ax.text(0.35, target_z, "羽毛球\n(目标)", fontsize=9, color="darkorange")

        # 视线
        ax.plot([-0.06, 0.2], [0, target_z], "g--", alpha=0.7, label="左摄像头视线")
        ax.plot([0.06, 0.2], [0, target_z], "r--", alpha=0.7, label="右摄像头视线")

        # 深度标注
        ax.annotate("", xy=(0.2, target_z), xytext=(0.2, 0),
                    arrowprops=dict(arrowstyle="<->", color="purple"))
        ax.text(0.45, target_z/2, f"Z={target_z}m", ha="center",
                fontsize=10, color="purple", fontweight="bold")

        # 公式
        ax.text(-1.3, 4.5, r"$Z = \frac{f \cdot B}{d}$", fontsize=14)
        ax.text(-1.3, 4.0, "f=焦距, B=基线, d=视差", fontsize=9, color="gray")

        ax.set_xlabel("水平位置 (m)")
        ax.set_ylabel("深度 (m)")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

        # 右图：测距精度 vs 距离
        ax2 = axes[1]
        distances = np.linspace(0.5, 10, 100)
        f_val, B_val = 800.0, 0.12

        # 假设视差测量误差 0.5px
        disp = f_val * B_val / distances
        disp_err = 0.5  # 像素误差
        # 深度误差 ≈ depth^2 * disp_err / (f * B)
        depth_err = distances**2 * disp_err / (f_val * B_val)
        depth_err_pct = depth_err / distances * 100

        ax2.plot(distances, depth_err_pct, "b-", linewidth=2, label="误差% (视差±0.5px)")
        ax2.axhline(y=5, color="orange", linestyle="--", label="5% 误差线")
        ax2.axhline(y=10, color="red", linestyle="--", label="10% 误差线")
        ax2.axvline(x=4.0, color="green", linestyle=":", label="典型场地距离 4m")

        # 标注典型场地距离的误差
        idx_4m = np.argmin(np.abs(distances - 4.0))
        err_at_4m = depth_err_pct[idx_4m]
        ax2.annotate(f"@4m: {err_at_4m:.1f}%",
                     xy=(4.0, err_at_4m),
                     xytext=(5.5, err_at_4m + 3),
                     arrowprops=dict(arrowstyle="->"),
                     fontsize=10)

        ax2.set_xlabel("目标距离 (m)")
        ax2.set_ylabel("深度估计误差 (%)")
        ax2.set_title(f"双目测距精度分析\n(f={f_val}px, B={B_val*100:.0f}cm)", fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 30)

        plt.tight_layout()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / "stereo_depth_analysis.png"
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"双目测距分析图已保存: {out_path}")

    except Exception as e:
        logger.warning(f"生成图表失败（需要 matplotlib）: {e}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="双目立体视觉测距 Demo")
    parser.add_argument("--mode", choices=["calibrate", "measure", "live", "simulate"],
                        default="simulate", help="运行模式（默认: simulate）")
    parser.add_argument("--left-dir", type=str, help="左摄像头标定图像目录")
    parser.add_argument("--right-dir", type=str, help="右摄像头标定图像目录")
    parser.add_argument("--left", type=str, help="左摄像头图像（测量模式）")
    parser.add_argument("--right", type=str, help="右摄像头图像（测量模式）")
    parser.add_argument("--calib", type=str, default=str(DEFAULT_CALIB_PATH),
                        help="标定参数文件路径")
    parser.add_argument("--board-cols", type=int, default=9, help="棋盘格列数（内角点）")
    parser.add_argument("--board-rows", type=int, default=6, help="棋盘格行数（内角点）")
    parser.add_argument("--square-size", type=float, default=0.025, help="方格大小（米）")

    args = parser.parse_args()

    if args.mode == "simulate":
        logger.info("运行双目测距模拟演示（无需真实摄像头）...")
        run_simulation()

    elif args.mode == "calibrate":
        if not args.left_dir or not args.right_dir:
            logger.error("标定模式需要 --left-dir 和 --right-dir")
            sys.exit(1)
        board_size = (args.board_cols, args.board_rows)
        run_calibration(args.left_dir, args.right_dir, board_size, args.square_size)

    elif args.mode == "measure":
        if not args.left or not args.right:
            logger.error("测量模式需要 --left 和 --right 图像")
            sys.exit(1)

        # 加载标定参数
        calib_path = Path(args.calib)
        if not calib_path.exists():
            logger.warning(f"未找到标定文件 {calib_path}，使用示例参数（精度可能较低）")
            calib = EXAMPLE_CALIB
        else:
            with open(str(calib_path)) as f:
                calib = json.load(f)

        if not CV2_AVAILABLE:
            logger.error("测量模式需要 OpenCV")
            sys.exit(1)

        img_l = cv2.imread(args.left, cv2.IMREAD_GRAYSCALE)
        img_r = cv2.imread(args.right, cv2.IMREAD_GRAYSCALE)

        logger.info("计算视差图...")
        disparity = compute_disparity_map(img_l, img_r, algorithm="sgbm")

        # 测量图像中心点的深度
        h, w = img_l.shape
        center = (w // 2, h // 2)
        depth = measure_point_depth(disparity, center, calib)
        logger.info(f"图像中心点深度: {depth:.3f}m")

        # 保存视差图可视化
        disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(OUTPUT_DIR / "disparity_map.png"), disp_color)
        logger.info(f"视差图已保存: {OUTPUT_DIR}/disparity_map.png")

    elif args.mode == "live":
        logger.info("实时双目测距（需要双目摄像头）")
        logger.info("TODO: 实现实时双目推理（依赖具体摄像头驱动）")
        logger.info("建议使用 OAK-D Lite + depthai 库实现实时深度")


if __name__ == "__main__":
    main()
