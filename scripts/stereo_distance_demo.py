#!/usr/bin/env python3
"""
stereo_distance_demo.py
=======================
Stereo depth mathematics and offline calibration demo.

Includes checkerboard calibration, rectification math, SGBM disparity,
point reprojection and synthetic geometry. No live stereo source, camera
synchronization or validated shuttlecock 3D tracker is implemented.

Usage:
  # Calibrate with matched checkerboard pairs
  python stereo_distance_demo.py --mode calibrate --left-dir data/left --right-dir data/right

  # Measure offline using an actual calibration file
  python stereo_distance_demo.py --mode measure --left left.jpg --right right.jpg

  # Synthetic geometry only; no camera required
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
    print("[WARNING] OpenCV is unavailable; only synthetic geometry can run")

from PIL import Image, ImageDraw

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
CALIB_DIR = PROJECT_ROOT / "src" / "perception"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"
DEFAULT_CALIB_PATH = CALIB_DIR / "stereo_calib.json"


# =============================================================================
# Example calibration values for simulation only.
# =============================================================================

EXAMPLE_CALIB = {
    "description": "Synthetic stereo parameters, not a real calibration",
    "image_size": [1280, 720],
    "baseline_m": 0.12,          # Example 12 cm baseline.
    "focal_length_px": 800.0,    # Focal length in pixels.
    "cx_left": 640.0,            # Left principal point x.
    "cy_left": 360.0,            # Left principal point y.
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
    "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # Rotation matrix.
    "T": [-0.12, 0, 0],                        # Translation (baseline).
    "Q_matrix": [
        [1, 0, 0, -640.0],
        [0, 1, 0, -360.0],
        [0, 0, 0,  800.0],
        [0, 0, 8.333, 0]   # Approximate inverse baseline term for synthetic values.
    ]
}


# =============================================================================
# Geometric utilities
# =============================================================================

def disparity_to_depth(disparity_px: float, focal_length_px: float, baseline_m: float) -> float:
    """
    Convert disparity to depth by triangulation.
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
    Convert pixel coordinates and depth into camera-frame meters.
    X = (px - cx) * depth / f
    Y = (py - cy) * depth / f
    Z = depth
    """
    X = (pixel_x - cx) * depth_m / focal_length_px
    Y = (pixel_y - cy) * depth_m / focal_length_px
    Z = depth_m
    return X, Y, Z


def speed_from_3d_positions(pos1: tuple, pos2: tuple, dt: float) -> float:
    """Speed between two 3D positions in meters per second."""
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    dz = pos2[2] - pos1[2]
    dist = math.sqrt(dx**2 + dy**2 + dz**2)
    return dist / max(dt, 1e-6)


# =============================================================================
# Calibration
# =============================================================================

def run_calibration(left_dir: str, right_dir: str, board_size=(9, 6),
                    square_size_m: float = 0.025,
                    output_path: Path = DEFAULT_CALIB_PATH) -> dict:
    """
    Stereo calibration from matched checkerboard views.
    board_size is the inner corner count (columns, rows).
    square_size_m is the measured square width in meters.
    """
    if not CV2_AVAILABLE:
        logger.error("Calibration requires OpenCV")
        return {}

    logger.info(f"Starting calibration: checkerboard {board_size}, square {square_size_m*100:.1f}cm")

    # Prepare object coordinates.
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size_m

    objpoints = []
    imgpoints_left = []
    imgpoints_right = []

    left_images = sorted(Path(left_dir).glob("*.jpg")) + sorted(Path(left_dir).glob("*.png"))
    right_images = sorted(Path(right_dir).glob("*.jpg")) + sorted(Path(right_dir).glob("*.png"))

    if len(left_images) != len(right_images):
        logger.error(f"Left/right image count mismatch: {len(left_images)} vs {len(right_images)}")
        return {}

    logger.info(f"Found {len(left_images)} image pairs")
    img_size = None
    valid_pairs = 0

    for lf, rf in zip(left_images, right_images):
        if lf.name != rf.name:
            raise ValueError(f"Unmatched checkerboard pair: {lf.name} vs {rf.name}")
        img_l = cv2.imread(str(lf), cv2.IMREAD_GRAYSCALE)
        img_r = cv2.imread(str(rf), cv2.IMREAD_GRAYSCALE)
        if img_l is None or img_r is None:
            continue
        if img_l.shape != img_r.shape:
            raise ValueError(f"Stereo image sizes differ: {lf.name} vs {rf.name}")

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
            logger.info(f"Valid checkerboard pair: {lf.name} + {rf.name}")

    if valid_pairs < 5:
        logger.error(f"Not enough valid checkerboard pairs (need at least 5, found {valid_pairs})")
        return {}

    logger.info(f"Estimating calibration from {valid_pairs} pairs")

    # Calibrate each camera.
    _, mtx_l, dist_l, _, _ = cv2.calibrateCamera(objpoints, imgpoints_left, img_size, None, None)
    _, mtx_r, dist_r, _, _ = cv2.calibrateCamera(objpoints, imgpoints_right, img_size, None, None)

    # Calibrate the stereo pair.
    flags = cv2.CALIB_FIX_INTRINSIC
    ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        mtx_l, dist_l, mtx_r, dist_r, img_size,
        flags=flags
    )

    logger.info(f"Stereo calibration complete; reprojection error: {ret:.4f} pixels")

    # Compute rectification maps.
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        mtx_l, dist_l, mtx_r, dist_r, img_size, R, T
    )

    # Extract baseline and focal length.
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

    # Save calibration.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(calib_data, f, indent=2)
    logger.info(f"Calibration saved: {output_path}")
    logger.info(f"Baseline: {baseline_m*100:.1f}cm, focal length: {focal_length_px:.1f}px")

    return calib_data


# =============================================================================
# Depth measurement
# =============================================================================

def compute_disparity_map(img_l_gray, img_r_gray, algorithm="sgbm"):
    """Compute a disparity map."""
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


def rectify_pair(img_l_gray, img_r_gray, calib: dict):
    """Rectify matched images using measured intrinsic and stereo matrices."""
    if not CV2_AVAILABLE:
        raise RuntimeError("Rectification requires OpenCV")
    if calib.get("description", "").startswith("Synthetic"):
        raise ValueError("Synthetic calibration cannot measure physical depth")
    required = ("image_size", "camera_matrix_left", "camera_matrix_right",
                "dist_coeffs_left", "dist_coeffs_right", "R1", "R2", "P1", "P2",
                "baseline_m", "focal_length_px")
    if any(key not in calib for key in required):
        raise ValueError("Calibration is missing measured rectification matrices")
    width, height = calib["image_size"]
    if (img_l_gray is None or img_r_gray is None or
            img_l_gray.shape != (height, width) or img_r_gray.shape != (height, width)):
        raise ValueError("Stereo images must match the calibrated resolution")
    if calib["baseline_m"] <= 0 or calib["focal_length_px"] <= 0:
        raise ValueError("Calibration baseline and focal length must be positive")

    rectified = []
    for image, side in ((img_l_gray, "left"), (img_r_gray, "right")):
        map_x, map_y = cv2.initUndistortRectifyMap(
            np.array(calib[f"camera_matrix_{side}"], dtype=np.float64),
            np.array(calib[f"dist_coeffs_{side}"], dtype=np.float64),
            np.array(calib["R1" if side == "left" else "R2"], dtype=np.float64),
            np.array(calib["P1" if side == "left" else "P2"], dtype=np.float64),
            (width, height), cv2.CV_32FC1,
        )
        rectified.append(cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR))
    return tuple(rectified)


def measure_point_depth(disparity_map, point_xy: tuple, calib: dict,
                         window_size: int = 5) -> float:
    """
    Measure depth at a pixel point (x, y); return meters.
    """
    x, y = int(point_xy[0]), int(point_xy[1])
    h, w = disparity_map.shape

    # Use the median in a local window.
    x1 = max(0, x - window_size // 2)
    x2 = min(w, x + window_size // 2 + 1)
    y1 = max(0, y - window_size // 2)
    y2 = min(h, y + window_size // 2 + 1)

    window = disparity_map[y1:y2, x1:x2]
    valid = window[window > 1.0]  # Reject invalid disparity.

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
# Synthetic geometry; no cameras or observed accuracy.
# =============================================================================

def run_simulation():
    """
    Demonstrate triangulation math with synthetic noise and positions.
    """
    logger.info("=" * 60)
    logger.info("Synthetic stereo geometry demonstration")
    logger.info("=" * 60)

    calib = EXAMPLE_CALIB
    f = calib["focal_length_px"]
    B = calib["baseline_m"]
    cx = calib["cx_left"]
    cy = calib["cy_left"]

    logger.info("Synthetic parameters:")
    logger.info(f"  Focal length: {f} px")
    logger.info(f"  Baseline: {B*100:.1f} cm")
    logger.info(f"  Principal point: ({cx}, {cy})")
    logger.info("")

    # Synthetic target positions at different depths.
    test_cases = [
        {"distance_m": 1.0, "pos_img_x": 660, "pos_img_y": 350},
        {"distance_m": 2.0, "pos_img_x": 645, "pos_img_y": 355},
        {"distance_m": 3.5, "pos_img_x": 643, "pos_img_y": 358},
        {"distance_m": 5.0, "pos_img_x": 641, "pos_img_y": 359},
        {"distance_m": 7.0, "pos_img_x": 640, "pos_img_y": 360},
    ]

    results = []
    rng = np.random.default_rng(42)
    logger.info(f"{'True depth':>12} {'Disp (px)':>12} {'Estimate':>12} {'Error %':>8} {'X(m)':>8} {'Y(m)':>8}")
    logger.info("-" * 65)

    for case in test_cases:
        true_dist = case["distance_m"]
        px, py = case["pos_img_x"], case["pos_img_y"]

        # True depth to ideal disparity.
        true_disparity = f * B / true_dist

        # Add arbitrary synthetic disparity noise.
        noise = rng.normal(0, 0.5)
        measured_disparity = true_disparity + noise

        # Noisy disparity to estimated depth.
        estimated_dist = disparity_to_depth(measured_disparity, f, B)

        # Synthetic 3D coordinates.
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
    logger.info(f"Synthetic mean error: {avg_error:.2f}% (not measured accuracy)")
    logger.info("")

    # Synthetic sequential positions and speed.
    logger.info("=" * 60)
    logger.info("Synthetic 3D speed calculation")
    logger.info("=" * 60)

    fps = 60
    dt = 1.0 / fps
    # Synthetic target moves toward the camera from 4 m at 50 m/s.
    initial_z = 4.0
    vz = -50.0  # Meters per second toward the camera.
    vx, vy = 2.0, -1.0

    positions = []
    for i in range(4):  # Stop before synthetic depth becomes negative.
        t = i * dt
        z = initial_z + vz * t - 0.5 * 9.8 * t**2 * 0.1
        x = vx * t
        y = vy * t
        positions.append((x, y, z))

    logger.info(f"{'Frame':>5} {'X(m)':>8} {'Y(m)':>8} {'Z(m)':>8} {'m/s':>12} {'km/h':>12}")
    logger.info("-" * 60)

    for i in range(1, len(positions)):
        speed = speed_from_3d_positions(positions[i-1], positions[i], dt)
        logger.info(f"{i:>5} {positions[i][0]:>8.3f} {positions[i][1]:>8.3f} "
                    f"{positions[i][2]:>8.3f} {speed:>12.1f} {speed*3.6:>12.1f}")

    # Draw an explanatory diagram.
    generate_stereo_diagram()

    # Save synthetic outputs with explicit provenance.
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
    logger.info(f"\nSynthetic result saved: {result_path}")

    return sim_result


def generate_stereo_diagram():
    """Draw a PNG explaining idealized stereo geometry."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: idealized triangulation.
        ax = axes[0]
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-0.5, 5)
        ax.set_aspect("equal")
        ax.set_title("Idealized stereo triangulation", fontsize=13)

        # Two synthetic cameras.
        ax.plot([-0.06, 0.06], [0, 0], "bs", markersize=15, label="Cameras")
        ax.text(-0.06, -0.3, "Left", ha="center", fontsize=9)
        ax.text(0.06, -0.3, "Right", ha="center", fontsize=9)

        # Baseline.
        ax.annotate("", xy=(0.06, 0), xytext=(-0.06, 0),
                    arrowprops=dict(arrowstyle="<->", color="blue"))
        ax.text(0, 0.15, "Baseline B=12cm", ha="center", fontsize=9, color="blue")

        # Synthetic target.
        target_z = 3.0
        ax.plot(0.2, target_z, "o", color="orange", markersize=12, label="Target")
        ax.text(0.35, target_z, "Target", fontsize=9, color="darkorange")

        # Sight lines.
        ax.plot([-0.06, 0.2], [0, target_z], "g--", alpha=0.7, label="Left sight line")
        ax.plot([0.06, 0.2], [0, target_z], "r--", alpha=0.7, label="Right sight line")

        # Depth label.
        ax.annotate("", xy=(0.2, target_z), xytext=(0.2, 0),
                    arrowprops=dict(arrowstyle="<->", color="purple"))
        ax.text(0.45, target_z/2, f"Z={target_z}m", ha="center",
                fontsize=10, color="purple", fontweight="bold")

        # Triangulation formula.
        ax.text(-1.3, 4.5, r"$Z = \frac{f \cdot B}{d}$", fontsize=14)
        ax.text(-1.3, 4.0, "f=focal length, B=baseline, d=disparity", fontsize=9, color="gray")

        ax.set_xlabel("Horizontal position (m)")
        ax.set_ylabel("Depth (m)")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

        # Right: theoretical sensitivity to a fixed disparity error.
        ax2 = axes[1]
        distances = np.linspace(0.5, 10, 100)
        f_val, B_val = 800.0, 0.12

        # Assume a synthetic 0.5-pixel disparity error.
        disp = f_val * B_val / distances
        disp_err = 0.5  # Assumed pixel error.
        # Approximate depth error: depth^2 * disparity_error / (f * B).
        depth_err = distances**2 * disp_err / (f_val * B_val)
        depth_err_pct = depth_err / distances * 100

        ax2.plot(distances, depth_err_pct, "b-", linewidth=2, label="Assumed disparity error: 0.5 px")
        ax2.axhline(y=5, color="orange", linestyle="--", label="5% reference")
        ax2.axhline(y=10, color="red", linestyle="--", label="10% reference")
        ax2.axvline(x=4.0, color="green", linestyle=":", label="Example distance: 4m")

        # Mark the example 4 m distance.
        idx_4m = np.argmin(np.abs(distances - 4.0))
        err_at_4m = depth_err_pct[idx_4m]
        ax2.annotate(f"@4m: {err_at_4m:.1f}%",
                     xy=(4.0, err_at_4m),
                     xytext=(5.5, err_at_4m + 3),
                     arrowprops=dict(arrowstyle="->"),
                     fontsize=10)

        ax2.set_xlabel("Target distance (m)")
        ax2.set_ylabel("Theoretical depth error (%)")
        ax2.set_title(f"Sensitivity to assumed disparity error\n(f={f_val}px, B={B_val*100:.0f}cm)", fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 30)

        plt.tight_layout()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / "stereo_depth_analysis.png"
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Synthetic geometry plot saved: {out_path}")

    except Exception as e:
        logger.warning(f"Could not create the plot (matplotlib required): {e}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Offline stereo calibration and synthetic geometry")
    parser.add_argument("--mode", choices=["calibrate", "measure", "simulate"],
                        default="simulate", help="Run mode (default: simulate)")
    parser.add_argument("--left-dir", type=str, help="Left checkerboard image directory")
    parser.add_argument("--right-dir", type=str, help="Right checkerboard image directory")
    parser.add_argument("--left", type=str, help="Left image in measure mode")
    parser.add_argument("--right", type=str, help="Right image in measure mode")
    parser.add_argument("--calib", type=str, default=str(DEFAULT_CALIB_PATH),
                        help="Measured stereo calibration JSON path")
    parser.add_argument("--board-cols", type=int, default=9, help="Inner checkerboard columns")
    parser.add_argument("--board-rows", type=int, default=6, help="Inner checkerboard rows")
    parser.add_argument("--square-size", type=float, default=0.025, help="Measured square width in meters")

    args = parser.parse_args()

    if args.mode == "simulate":
        logger.info("Running synthetic stereo geometry without cameras")
        run_simulation()

    elif args.mode == "calibrate":
        if not args.left_dir or not args.right_dir:
            logger.error("Calibration requires --left-dir and --right-dir")
            sys.exit(1)
        if args.board_cols < 3 or args.board_rows < 3 or args.square_size <= 0:
            parser.error("Checkerboard inner corners and measured square width must be positive")
        board_size = (args.board_cols, args.board_rows)
        if not run_calibration(args.left_dir, args.right_dir, board_size,
                               args.square_size, Path(args.calib)):
            parser.error("Calibration failed; collect more usable matched checkerboard pairs")

    elif args.mode == "measure":
        if not args.left or not args.right:
            logger.error("Measurement requires --left and --right images")
            sys.exit(1)

        # Load a measured calibration; example values are not acceptable.
        calib_path = Path(args.calib)
        if not calib_path.exists():
            parser.error(f"Measured calibration not found: {calib_path}")
        with calib_path.open(encoding="utf-8") as f:
            calib = json.load(f)

        if not CV2_AVAILABLE:
            logger.error("Measurement requires OpenCV")
            sys.exit(1)

        img_l = cv2.imread(args.left, cv2.IMREAD_GRAYSCALE)
        img_r = cv2.imread(args.right, cv2.IMREAD_GRAYSCALE)
        try:
            img_l, img_r = rectify_pair(img_l, img_r, calib)
        except (ValueError, KeyError) as exc:
            parser.error(str(exc))

        logger.info("Computing disparity from the supplied images")
        disparity = compute_disparity_map(img_l, img_r, algorithm="sgbm")

        # Measure depth at the image center.
        h, w = img_l.shape
        center = (w // 2, h // 2)
        depth = measure_point_depth(disparity, center, calib)
        if not math.isfinite(depth):
            parser.error("No valid disparity near the image center")
        logger.info(f"Image center depth: {depth:.3f}m")

        # Save a disparity visualization.
        disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(OUTPUT_DIR / "disparity_map.png"), disp_color)
        logger.info(f"Disparity map saved: {OUTPUT_DIR}/disparity_map.png")



if __name__ == "__main__":
    main()
