"""Verify that a team member's environment can run the project.

Checks, in order:
  1. Python version (3.11-3.13 supported, 3.12 recommended).
  2. Required packages import, and their versions.
  3. PyTorch device: CUDA availability, GPU name, VRAM, compute capability,
     and a real computation on the GPU.
  4. YOLO smoke test: downloads the official COCO-pretrained YOLO26n weights
     (about 5 MB) into models/pretrained/ and runs one inference.

The YOLO test uses a synthetic image, so it only proves the pipeline runs.
It says nothing about road-sign accuracy: that model has never seen our data.

Usage (from the repository root, with the virtual environment active):
    python scripts/check_env.py
    python scripts/check_env.py --skip-yolo      # offline / no download
    python scripts/check_env.py --device cpu     # force CPU

Exit code is 0 if every required check passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import importlib
import platform
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PRETRAINED_DIR = REPO_ROOT / "models" / "pretrained"

SUPPORTED_PYTHON = ((3, 11), (3, 13))  # inclusive range
RECOMMENDED_PYTHON = (3, 12)

# (import name, pip name)
REQUIRED_PACKAGES: list[tuple[str, str]] = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("ultralytics", "ultralytics"),
    ("cv2", "opencv-python"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("sklearn", "scikit-learn"),
    ("yaml", "pyyaml"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("streamlit", "streamlit"),
    ("pytest", "pytest"),
]

# ASCII only: legacy Windows consoles may not render Unicode symbols.
OK, WARN, FAIL = "[ OK ]", "[WARN]", "[FAIL]"


class Report:
    """Collects check results and prints them as they happen."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def ok(self, msg: str) -> None:
        print(f"{OK} {msg}")

    def warn(self, msg: str) -> None:
        print(f"{WARN} {msg}")
        self.warnings.append(msg)

    def fail(self, msg: str) -> None:
        print(f"{FAIL} {msg}")
        self.failures.append(msg)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def check_python(report: Report) -> None:
    section("Python")
    v = sys.version_info
    print(f"       {sys.version.split()[0]} on {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"       executable: {sys.executable}")
    lo, hi = SUPPORTED_PYTHON
    if not (lo <= (v.major, v.minor) <= hi):
        report.fail(
            f"Python {v.major}.{v.minor} is not supported. Use {RECOMMENDED_PYTHON[0]}.{RECOMMENDED_PYTHON[1]} "
            "(PyTorch 2.15 dropped 3.10)."
        )
    elif (v.major, v.minor) != RECOMMENDED_PYTHON:
        report.warn(
            f"Python {v.major}.{v.minor} works, but the team standard is "
            f"{RECOMMENDED_PYTHON[0]}.{RECOMMENDED_PYTHON[1]}."
        )
    else:
        report.ok(f"Python {v.major}.{v.minor} (team standard)")

    if sys.prefix == sys.base_prefix:
        report.warn("Not running inside a virtual environment (.venv). Activate it first.")
    else:
        report.ok(f"Virtual environment active: {sys.prefix}")


def check_packages(report: Report) -> dict[str, object]:
    section("Packages")
    modules: dict[str, object] = {}
    for import_name, pip_name in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(import_name)
        except Exception as exc:  # ImportError, or a broken native library
            report.fail(f"{pip_name}: cannot import ({type(exc).__name__}: {exc})")
            continue
        version = getattr(module, "__version__", "unknown")
        report.ok(f"{pip_name:<14} {version}")
        modules[import_name] = module
    return modules


def check_torch_device(report: Report, requested: str) -> str:
    """Return the device string to use for the YOLO test ('cpu' or '0')."""
    section("PyTorch device")
    try:
        import torch
    except Exception:
        report.fail("torch not importable; skipping device checks.")
        return "cpu"

    print(f"       torch {torch.__version__}, built for CUDA {torch.version.cuda or 'none (CPU build)'}")

    if not torch.cuda.is_available():
        if torch.version.cuda is None:
            report.warn(
                "CPU-only PyTorch build. Fine for a machine without an NVIDIA GPU. "
                "If this machine HAS one, reinstall torch with the cu132 index URL."
            )
        elif shutil.which("nvidia-smi") is None:
            report.warn("No NVIDIA driver found on this machine: running on CPU.")
        else:
            report.warn(
                "An NVIDIA driver is installed but PyTorch cannot use the GPU. Most likely cause: "
                "the driver is too old for CUDA 13 (needs 580 or newer). Run `nvidia-smi` and update it."
            )
        return "cpu"

    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1024**3
    report.ok(f"GPU: {props.name}, {vram_gb:.1f} GB VRAM, compute capability {props.major}.{props.minor}")

    try:
        a = torch.randn(2048, 2048, device="cuda")
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            b = a @ a
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000 / 10
        if not torch.isfinite(b).all():
            report.fail("GPU matrix multiply produced non-finite values.")
        else:
            report.ok(f"GPU computation works (2048x2048 matmul: {elapsed_ms:.1f} ms)")
    except Exception as exc:
        report.fail(f"GPU computation failed: {type(exc).__name__}: {exc}")
        return "cpu"

    if vram_gb < 5:
        report.warn(
            f"{vram_gb:.1f} GB VRAM: OK for inference and the CNN classifier; detector training "
            "at high resolution will need a small batch size (or use the RTX 4050 / Colab)."
        )
    return "cpu" if requested == "cpu" else "0"


def check_yolo(report: Report, device: str) -> None:
    section("YOLO smoke test")
    try:
        import numpy as np
        from ultralytics import YOLO
    except Exception as exc:
        report.fail(f"Cannot import ultralytics/numpy: {exc}")
        return

    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    weights = PRETRAINED_DIR / "yolo26n.pt"
    try:
        model = YOLO(str(weights))  # downloads the official weights if missing
    except Exception as exc:
        report.fail(f"Could not load/download yolo26n.pt: {type(exc).__name__}: {exc}")
        return
    report.ok(f"Loaded COCO-pretrained YOLO26n ({weights.relative_to(REPO_ROOT)})")

    # Synthetic 1360x800 frame (same size as GTSDB images): grey noise.
    rng = np.random.default_rng(0)
    frame = rng.integers(90, 160, size=(800, 1360, 3), dtype=np.uint8)
    try:
        model.predict(frame, device=device, imgsz=640, verbose=False)  # warm-up
        runs = 5
        start = time.perf_counter()
        for _ in range(runs):
            results = model.predict(frame, device=device, imgsz=640, verbose=False)
        elapsed_ms = (time.perf_counter() - start) * 1000 / runs
    except Exception as exc:
        report.fail(f"Inference failed on device={device}: {type(exc).__name__}: {exc}")
        return

    n_boxes = len(results[0].boxes)
    where = "CPU" if device == "cpu" else "GPU"
    report.ok(
        f"Inference runs on {where}: {elapsed_ms:.1f} ms/frame at imgsz=640 "
        f"(about {1000 / elapsed_ms:.0f} FPS; {n_boxes} boxes on a noise image, as expected)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-yolo", action="store_true", help="skip the YOLO download and inference test")
    parser.add_argument("--device", choices=["auto", "cpu"], default="auto", help="device for the YOLO test")
    args = parser.parse_args()

    report = Report()
    check_python(report)
    check_packages(report)
    device = check_torch_device(report, args.device)
    if not args.skip_yolo:
        check_yolo(report, device)

    section("Summary")
    if report.failures:
        print(f"{FAIL} {len(report.failures)} required check(s) failed:")
        for msg in report.failures:
            print(f"       - {msg}")
        return 1
    print(f"{OK} Environment ready ({len(report.warnings)} warning(s)).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
