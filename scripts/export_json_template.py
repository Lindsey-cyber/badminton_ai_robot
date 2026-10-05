#!/usr/bin/env python3
"""
export_json_template.py
=======================
Badminton annotation JSON template generator

Generate three annotation templates:
  1. COCO format (human keypoints and shuttlecock boxes)
  2. Shuttlecock trajectory format (multiple frames)
  3. Training session metadata format

Usage:
    python export_json_template.py --type coco --output annotations/coco_template.json
    python export_json_template.py --type trajectory --output annotations/trajectory_template.json
    python export_json_template.py --type session --output annotations/session_template.json
    python export_json_template.py --all --output-dir annotations/
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "annotations"

# =============================================================================
# Template 1: COCO pose and shuttlecock detection
# =============================================================================

COCO_TEMPLATE = {
    "info": {
        "description": "Badminton AI Dataset - Pose + Shuttlecock Detection",
        "url": "",
        "version": "1.0",
        "year": datetime.now().year,
        "contributor": "Badminton Robot Team",
        "date_created": datetime.now().strftime("%Y/%m/%d"),
    },
    "licenses": [],
    "categories": [
        {
            "supercategory": "person",
            "id": 1,
            "name": "player",
            "keypoints": [
                "nose",           # 0
                "left_eye",       # 1
                "right_eye",      # 2
                "left_ear",       # 3
                "right_ear",      # 4
                "left_shoulder",  # 5
                "right_shoulder", # 6
                "left_elbow",     # 7
                "right_elbow",    # 8
                "left_wrist",     # 9
                "right_wrist",    # 10
                "left_hip",       # 11
                "right_hip",      # 12
                "left_knee",      # 13
                "right_knee",     # 14
                "left_ankle",     # 15
                "right_ankle",    # 16
            ],
            "skeleton": [
                [16, 14], [14, 12], [17, 15], [15, 13],
                [12, 13], [6, 12], [7, 13], [6, 7],
                [6, 8], [7, 9], [8, 10], [9, 11],
                [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]
            ],
        },
        {
            "supercategory": "sports",
            "id": 2,
            "name": "shuttlecock",
            "keypoints": [],
            "skeleton": [],
        },
        {
            "supercategory": "court",
            "id": 3,
            "name": "net",
            "keypoints": ["left_post", "right_post", "net_center_top"],
            "skeleton": [[1, 2], [2, 3]],
        }
    ],
    "images": [
        {
            "__comment__": "Metadata for each image",
            "id": 1,
            "file_name": "cam0/000000.jpg",
            "width": 1280,
            "height": 720,
            "date_captured": "2026-03-27T10:00:00",
            "session_id": "session_20260327_100000",
            "camera_id": 0,
            "frame_index": 0,
            "timestamp_s": 0.0,
            "license": 0,
        }
    ],
    "annotations": [
        {
            "__comment__": "Example human keypoint annotation",
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "segmentation": [],
            "area": 80000.0,
            "bbox": [200, 100, 300, 500],  # [x, y, width, height]
            "iscrowd": 0,
            "keypoints": [
                # Format:[x1, y1, v1, x2, y2, v2, ...]
                # v: 0=not labeled, 1=labeled but not visible, 2=labeled and visible
                350, 120, 2,   # 0: nose
                340, 110, 2,   # 1: left_eye
                360, 110, 2,   # 2: right_eye
                330, 115, 2,   # 3: left_ear
                370, 115, 2,   # 4: right_ear
                310, 160, 2,   # 5: left_shoulder
                390, 160, 2,   # 6: right_shoulder
                290, 220, 2,   # 7: left_elbow
                410, 220, 2,   # 8: right_elbow
                270, 280, 2,   # 9: left_wrist
                430, 280, 2,   # 10: right_wrist
                320, 300, 2,   # 11: left_hip
                380, 300, 2,   # 12: right_hip
                310, 380, 2,   # 13: left_knee
                390, 380, 2,   # 14: right_knee
                305, 460, 2,   # 15: left_ankle
                395, 460, 2,   # 16: right_ankle
            ],
            "num_keypoints": 17,
            "attributes": {
                "is_ready": True,             # Whether ready to receive
                "action_label": "forehand_clear",   # Action label
                "action_quality": 3,          # Action quality, 1–5
            }
        },
        {
            "__comment__": "Example shuttlecock annotation",
            "id": 2,
            "image_id": 1,
            "category_id": 2,
            "segmentation": [],
            "area": 100.0,
            "bbox": [640, 300, 12, 12],    # Shuttlecock box [x, y, w, h]
            "iscrowd": 0,
            "keypoints": [],
            "num_keypoints": 0,
            "attributes": {
                "visibility": 2,              # 0=invisible, 1=partially visible, 2=fully visible
                "motion_blur": 1,             # Motion blur level 0-3
                "speed_estimate_kmh": 120.0,  # Estimated shuttle speed, if available
                "shuttle_state": "in_flight", # in_flight / at_rest / unknown
            }
        }
    ]
}

# =============================================================================
# Template 2: trajectory annotations
# =============================================================================

TRAJECTORY_TEMPLATE = {
    "format_version": "1.0",
    "description": "Shuttlecock trajectory annotation (multiple frames)",
    "session_id": "session_20260327_100000",
    "camera_id": 0,
    "fps": 60,
    "image_width": 1280,
    "image_height": 720,
    "trajectory_id": "traj_001",
    "shot_type": "forehand_clear",    # Shot type: forehand_clear/backhand_clear/smash/net_shot/drive
    "server_id": "robot",             # Server: robot / player_A / player_B
    "receiver_id": "player_A",        # Receiver
    "trajectory": [
        {
            "frame_index": 0,
            "timestamp_s": 0.000,
            "ball_center_px": {"x": 640.0, "y": 200.0},
            "ball_radius_px": 8.0,
            "visibility": 2,          # 0=invisible, 1=blurred, 2=clear
            "motion_blur_level": 0,   # 0=none, 1=mild, 2=severe, 3=extreme
            "annotator_confidence": 3, # Annotator confidence 1-3
        },
        {
            "frame_index": 1,
            "timestamp_s": 0.0167,
            "ball_center_px": {"x": 630.0, "y": 195.0},
            "ball_radius_px": 8.5,
            "visibility": 2,
            "motion_blur_level": 1,
            "annotator_confidence": 3,
        },
        {
            "__comment__": "Example blurred or invisible frame",
            "frame_index": 5,
            "timestamp_s": 0.083,
            "ball_center_px": {"x": None, "y": None},  # invisibleis null
            "ball_radius_px": None,
            "visibility": 0,
            "motion_blur_level": 3,
            "annotator_confidence": 1,
            "interpolated": True,     # Interpolated estimate, not a measured annotation
        }
    ],
    "landing_point": {
        "px": {"x": 850.0, "y": 650.0},
        "court_coords_m": {"x": 2.5, "y": 8.0},   # Court coordinates (homography required)
        "zone": "back_court_left",               # Court zone
    },
    "peak_speed_kmh": 145.0,
    "trajectory_length_px": 1200.0,
    "quality_check": {
        "is_complete": True,         # Whether the trajectory is complete (serve to landing)
        "missing_frames_count": 1,   # Missing frame count
        "usable_for_training": True,
        "reviewer": "reviewer_A",
        "review_date": "2026-03-27",
    }
}

# =============================================================================
# Template 3: training session metadata
# =============================================================================

SESSION_META_TEMPLATE = {
    "format_version": "1.0",
    "session_id": "session_20260327_100000",
    "date": "2026-03-27",
    "time_start": "10:00:00",
    "time_end": "11:30:00",
    "location": {
        "venue": "Example badminton venue",
        "city": "Beijing",
        "court_type": "indoor",
        "floor_type": "Wood",
        "lighting": "Overhead LED lighting, approximately 800 lux",
    },
    "equipment": {
        "cameras": [
            {
                "camera_id": 0,
                "role": "left_stereo",          # left_stereo/right_stereo/overview/side
                "position_m": {"x": 0.0, "y": 0.0, "z": 1.5},  # Court coordinate system
                "orientation_deg": {"pan": 0, "tilt": -15, "roll": 0},
                "model": "Logitech C920",
                "resolution": [1280, 720],
                "fps": 60,
                "focal_length_mm": 3.67,
            },
            {
                "camera_id": 1,
                "role": "right_stereo",
                "position_m": {"x": 0.12, "y": 0.0, "z": 1.5},
                "orientation_deg": {"pan": 0, "tilt": -15, "roll": 0},
                "model": "Logitech C920",
                "resolution": [1280, 720],
                "fps": 60,
                "focal_length_mm": 3.67,
            }
        ],
        "microphone": {
            "model": "USB Microphone",
            "channels": 1,
            "sample_rate_hz": 44100,
        },
        "robot": {
            "firmware_version": "1.0.0",
            "ball_type": "RSL TOURNEY shuttlecock",
            "launch_settings_default": {
                "speed_kmh": 80,
                "angle_deg": 45,
                "spin": "none",
            }
        }
    },
    "subjects": [
        {
            "subject_id": "player_A",
            "age": 25,
            "gender": "M",
            "skill_level": "intermediate",   # beginner/intermediate/advanced/professional
            "dominant_hand": "right",
            "height_cm": 175,
            "notes": "",
        }
    ],
    "protocol": {
        "drill_type": "multi_directional",   # Training drill
        "total_shots_planned": 200,
        "actual_shots": 180,
        "rest_intervals_s": [30, 30, 60],
        "speed_levels_kmh": [60, 80, 100],
    },
    "annotations": {
        "total_frames": 10800,
        "annotated_frames": 9500,
        "total_trajectories": 180,
        "annotated_trajectories": 160,
        "annotation_tool": "CVAT v2.0",
        "annotators": ["annotator_A", "annotator_B"],
    },
    "data_split": {
        "train": 0.7,
        "val": 0.15,
        "test": 0.15,
        "split_strategy": "trajectory_based",  # Split by trajectory to prevent adjacent-frame leakage
    },
    "quality_flags": {
        "has_calibration": True,
        "lighting_consistent": True,
        "audio_clean": True,
        "usable": True,
        "notes": "",
    }
}

# =============================================================================
# Generation functions
# =============================================================================

def save_template(template: dict, output_path: Path, indent: int = 2):
    """Save a template as JSON"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output_path), "w", encoding="utf-8") as f:
        json.dump(template, f, indent=indent, ensure_ascii=False)
    print(f"✓ Template saved: {output_path}")


def validate_coco_annotation(annotation_path: str) -> dict:
    """
    Basic validation of a COCO annotation file
    Return validation result
    """
    issues = []
    try:
        with open(annotation_path) as f:
            data = json.load(f)

        required_keys = ["info", "categories", "images", "annotations"]
        for key in required_keys:
            if key not in data:
                issues.append(f"Missing required field: {key}")

        # Check keypoint count
        for anno in data.get("annotations", []):
            if anno.get("category_id") == 1:  # player
                kps = anno.get("keypoints", [])
                if len(kps) != 51:  # 17 * 3
                    issues.append(f"Annotation ID {anno['id']}: Invalid keypoint count "
                                  f"({len(kps)}，expected 51)")

        if not issues:
            return {"valid": True, "message": "Validation passed"}
        else:
            return {"valid": False, "issues": issues}

    except json.JSONDecodeError as e:
        return {"valid": False, "issues": [f"JSON parsing error: {e}"]}
    except FileNotFoundError:
        return {"valid": False, "issues": ["File does not exist"]}


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Annotation template generator")
    parser.add_argument("--type", choices=["coco", "trajectory", "session"],
                        help="Template type")
    parser.add_argument("--all", action="store_true", help="Generate all templates")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help=f"Output directory (for --all; default:  {DEFAULT_OUTPUT_DIR}）")
    parser.add_argument("--validate", type=str, help="Validate a specified COCO annotation file")

    args = parser.parse_args()

    if args.validate:
        result = validate_coco_annotation(args.validate)
        if result["valid"]:
            print(f"✓ {result['message']}")
        else:
            print("✗ Validation failed:")
            for issue in result["issues"]:
                print(f"  - {issue}")
        return

    output_dir = Path(args.output_dir)

    if args.all:
        save_template(COCO_TEMPLATE, output_dir / "coco_pose_shuttle_template.json")
        save_template(TRAJECTORY_TEMPLATE, output_dir / "trajectory_template.json")
        save_template(SESSION_META_TEMPLATE, output_dir / "session_meta_template.json")
        print(f"\nAll templates generated in: {output_dir}")

    elif args.type == "coco":
        out = Path(args.output) if args.output else output_dir / "coco_template.json"
        save_template(COCO_TEMPLATE, out)

    elif args.type == "trajectory":
        out = Path(args.output) if args.output else output_dir / "trajectory_template.json"
        save_template(TRAJECTORY_TEMPLATE, out)

    elif args.type == "session":
        out = Path(args.output) if args.output else output_dir / "session_meta_template.json"
        save_template(SESSION_META_TEMPLATE, out)

    else:
        print("Specify --type or --all")
        print("Example: python export_json_template.py --all")


if __name__ == "__main__":
    main()
