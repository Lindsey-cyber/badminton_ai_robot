#!/usr/bin/env python3
"""
setup_and_fix.py
================
一键修复脚本：解决 macOS / ARM64 环境下的依赖和模型下载问题

运行方式：
    python scripts/setup_and_fix.py

执行内容：
  1. 检查 Python 环境和已安装包
  2. 自动安装缺失依赖
  3. 下载 / 导出 YOLOv8n-pose.onnx 模型（删除无效文件后重新获取）
  4. 验证模型可加载
  5. 在一张测试图上跑一次推理，确认端到端流程正常
"""

import subprocess
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "src" / "pose" / "yolov8n-pose.onnx"
MIN_MODEL_SIZE = 1024 * 10  # 10KB

# ── ANSI 颜色 ────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
RESET  = "\033[0m"

def ok(msg):   print(f"{GREEN}✓ {msg}{RESET}")
def warn(msg): print(f"{YELLOW}⚠ {msg}{RESET}")
def err(msg):  print(f"{RED}✗ {msg}{RESET}")
def info(msg): print(f"{CYAN}→ {msg}{RESET}")


# ── Step 1: 环境检查 ─────────────────────────────────────────────────────────
def check_python():
    print("\n[Step 1] Python 环境检查")
    ok(f"Python {sys.version}")
    ok(f"Python 路径: {sys.executable}")
    import platform
    ok(f"平台: {platform.system()} {platform.machine()}")


# ── Step 2: 安装缺失依赖 ──────────────────────────────────────────────────────
REQUIRED_PACKAGES = {
    "onnxruntime": "onnxruntime",
    "numpy":       "numpy",
    "PIL":         "Pillow",
    "imageio":     "imageio[pyav]",
}
OPTIONAL_PACKAGES = {
    "cv2":         "opencv-python",
    "ultralytics": "ultralytics",
}

def install_package(pip_name: str) -> bool:
    info(f"安装 {pip_name}...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", pip_name, "-q"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        ok(f"安装成功: {pip_name}")
        return True
    else:
        err(f"安装失败: {pip_name}")
        print(result.stderr[-500:])
        return False

def check_and_install_deps():
    print("\n[Step 2] 依赖检查与安装")
    missing_required = []
    for module, pkg in REQUIRED_PACKAGES.items():
        try:
            __import__(module)
            ok(f"{module} 已安装")
        except ImportError:
            warn(f"{module} 未安装")
            missing_required.append(pkg)

    for pkg in missing_required:
        install_package(pkg)

    print("\n  可选依赖（建议安装）：")
    for module, pkg in OPTIONAL_PACKAGES.items():
        try:
            __import__(module)
            ok(f"{module} 已安装（可选）")
        except ImportError:
            warn(f"{module} 未安装（可选但推荐）")
            ans = input(f"  是否自动安装 {pkg}? [y/N] ").strip().lower()
            if ans == "y":
                install_package(pkg)


# ── Step 3: 模型准备 ──────────────────────────────────────────────────────────
def prepare_model():
    print("\n[Step 3] 模型检查与下载")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 删除空文件
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size < MIN_MODEL_SIZE:
        warn(f"已有模型文件无效（{MODEL_PATH.stat().st_size} bytes），删除重新获取...")
        MODEL_PATH.unlink()

    if MODEL_PATH.exists():
        ok(f"模型已存在 ({MODEL_PATH.stat().st_size / 1024 / 1024:.1f}MB): {MODEL_PATH}")
        return True

    # 方式 A：ultralytics
    try:
        info("尝试方式A: ultralytics 下载并导出 ONNX...")
        import ultralytics
        from ultralytics import YOLO
        m = YOLO("yolov8n-pose.pt")
        export_path = m.export(format="onnx", simplify=True, imgsz=640)
        import shutil
        shutil.copy(str(export_path), str(MODEL_PATH))
        ok(f"方式A 成功！模型已保存: {MODEL_PATH}")
        return True
    except ImportError:
        warn("ultralytics 未安装，跳过方式A")
    except Exception as e:
        warn(f"方式A 失败: {e}")

    # 方式 B：curl 下载（macOS 内置）
    urls = [
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.onnx",
        "https://huggingface.co/Ultralytics/assets/resolve/main/yolov8n-pose.onnx",
    ]
    for url in urls:
        info(f"尝试方式B: curl 下载 {url[:60]}...")
        result = subprocess.run(
            ["curl", "-L", "-o", str(MODEL_PATH), "--max-time", "120",
             "--user-agent", "Mozilla/5.0", url],
            capture_output=False  # 显示 curl 进度
        )
        if result.returncode == 0 and MODEL_PATH.exists() and MODEL_PATH.stat().st_size > MIN_MODEL_SIZE:
            ok(f"方式B 成功！({MODEL_PATH.stat().st_size / 1024 / 1024:.1f}MB)")
            return True
        else:
            warn(f"方式B 失败（文件大小={MODEL_PATH.stat().st_size if MODEL_PATH.exists() else 0}）")
            MODEL_PATH.unlink(missing_ok=True)

    # 方式 C：手动指引
    err("自动下载失败。请手动操作：")
    print()
    print("  # 方法1（推荐）：安装 ultralytics 后重新运行此脚本")
    print("  pip install ultralytics")
    print("  python scripts/setup_and_fix.py")
    print()
    print("  # 方法2：手动 curl 下载")
    print(f"  curl -L '{urls[0]}' -o '{MODEL_PATH}'")
    print()
    print("  # 方法3：浏览器下载后放到指定位置")
    print(f"  下载地址: {urls[0]}")
    print(f"  放置路径: {MODEL_PATH}")
    return False


# ── Step 4: 验证模型加载 ───────────────────────────────────────────────────────
def validate_model():
    print("\n[Step 4] 验证模型加载")
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size < MIN_MODEL_SIZE:
        err("模型文件不存在或无效，跳过验证")
        return False

    try:
        import onnxruntime as ort
        session = ort.InferenceSession(str(MODEL_PATH), providers=ort.get_available_providers())
        inp = session.get_inputs()[0]
        out = session.get_outputs()
        ok(f"模型加载成功！")
        ok(f"  输入: {inp.name} shape={inp.shape}")
        ok(f"  输出: {[o.name for o in out]}")
        ok(f"  可用后端: {ort.get_available_providers()}")
        return True
    except Exception as e:
        err(f"模型加载失败: {e}")
        return False


# ── Step 5: 端到端推理验证 ────────────────────────────────────────────────────
def run_quick_inference():
    print("\n[Step 5] 端到端推理验证（合成图）")
    try:
        import numpy as np
        import onnxruntime as ort
        from PIL import Image, ImageDraw

        # 创建合成人形图像（640x640）
        img = Image.new("RGB", (640, 640), (180, 180, 180))
        draw = ImageDraw.Draw(img)
        # 简单人形
        draw.ellipse([280, 50, 360, 130], fill=(200, 160, 120))   # 头
        draw.rectangle([295, 130, 345, 310], fill=(80, 120, 200)) # 躯干
        draw.rectangle([240, 130, 295, 300], fill=(80, 120, 200)) # 左臂
        draw.rectangle([345, 130, 400, 300], fill=(80, 120, 200)) # 右臂
        draw.rectangle([295, 310, 320, 500], fill=(60, 90, 160))  # 左腿
        draw.rectangle([320, 310, 345, 500], fill=(60, 90, 160))  # 右腿
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = img_np.transpose(2, 0, 1)[np.newaxis, ...]  # [1,3,640,640]

        session = ort.InferenceSession(str(MODEL_PATH), providers=ort.get_available_providers())
        input_name = session.get_inputs()[0].name

        import time
        t0 = time.perf_counter()
        outputs = session.run(None, {input_name: img_tensor})
        t1 = time.perf_counter()
        infer_ms = (t1 - t0) * 1000

        ok(f"推理成功！耗时 {infer_ms:.1f}ms ({1000/infer_ms:.1f} FPS 理论值)")
        ok(f"输出形状: {[o.shape for o in outputs]}")

        # 解析检测结果
        preds = outputs[0][0].T  # [N, 56]
        conf = preds[:, 4]
        n_detected = int((conf > 0.1).sum())
        ok(f"检测到高置信度候选框: {n_detected} 个（阈值=0.1）")

        # 保存验证图
        out_dir = PROJECT_ROOT / "outputs" / "figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "setup_validation_result.jpg"
        img.save(str(out_path))
        ok(f"验证图已保存: {out_path}")

        print(f"\n{GREEN}{'='*55}{RESET}")
        print(f"{GREEN}  ✅ 全部验证通过！环境已就绪。{RESET}")
        print(f"{GREEN}{'='*55}{RESET}")
        print(f"\n现在可以运行推理了：")
        print(f"  python scripts/demo_pose_inference.py --input <你的视频.mp4>")
        print(f"  python scripts/demo_pose_inference.py --camera 0")
        print()
        return True

    except Exception as e:
        err(f"推理验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


# ── 主函数 ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"{CYAN}{'='*55}")
    print(f"  羽毛球 AI 教练系统 - 环境修复脚本")
    print(f"{'='*55}{RESET}")

    check_python()
    check_and_install_deps()
    model_ok = prepare_model()
    if model_ok:
        validate_ok = validate_model()
        if validate_ok:
            run_quick_inference()
        else:
            err("模型验证失败，请检查模型文件是否完整")
    else:
        err("模型准备失败，请参考上方手动操作指引")
        sys.exit(1)
