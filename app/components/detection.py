"""Glue between the Streamlit pages and the detector + CNN pipeline.

The pipeline itself (src/roadsigns/pipeline.py) is the same code as the one used by
scripts/evaluate_pipeline.py: the app shows exactly what was evaluated.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from components import results as R
from components.settings import get

GTSDB_DIR = R.DATA_RAW / "FullIJCNN2013"
FIRST_TEST_IMAGE = 600  # images 00600-00899 = official GTSDB test set (never used for training)


def missing_weights() -> list[str]:
    """Names of the weight files missing on this computer (empty list = ready)."""
    status = R.weights_status()
    files = {"detector": f"detector_{R.DETECTOR_RUN}.pt", "classifier": R.CLASSIFIER_FILE}
    return [f"models/weights/{files[k]}" for k, path in status.items() if path is None]


@st.cache_resource(show_spinner="Chargement du détecteur et du CNN…")
def load_pipeline(device: str):
    """Load both models once per device and keep them in memory for every page."""
    from roadsigns.pipeline import RoadSignPipeline  # heavy import (torch, ultralytics)

    return RoadSignPipeline.from_default_weights(device=device)


def current_settings() -> dict:
    return {key: get(key) for key in ("det_conf", "cnn_conf", "imgsz", "device")}


@st.cache_resource(show_spinner=False)
def cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def configured_pipeline(settings: dict):
    """The shared pipeline, with the thresholds of this session."""
    pipe = load_pipeline(settings["device"])
    # The models are shared; only the thresholds change between runs.
    pipe.det_conf = float(settings["det_conf"])
    pipe.cnn_conf = float(settings["cnn_conf"])
    pipe.imgsz = int(settings["imgsz"])
    return pipe


def analyse(image_bgr: np.ndarray, settings: dict):
    """Run the pipeline with the session settings. Returns (result, annotated_bgr, device)."""
    pipe = configured_pipeline(settings)
    result = pipe.predict(image_bgr)
    return result, pipe.annotate(image_bgr, result), pipe.device


def decode_image(data: bytes) -> np.ndarray | None:
    """Bytes of a JPG/PNG/PPM/BMP file -> BGR image, or None if it is not a readable image."""
    if not data:
        return None
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    return image if image is not None and image.size else None


def image_key(data: bytes, settings: dict) -> str:
    """Same image + same settings -> same key: the analysis is not run twice."""
    h = hashlib.sha1(data).hexdigest()[:16]
    return f"{h}|{settings['det_conf']}|{settings['cnn_conf']}|{settings['imgsz']}|{settings['device']}"


def example_images() -> list[Path]:
    """GTSDB TEST images on this computer (never seen during training)."""
    if not GTSDB_DIR.is_dir():
        return []
    return sorted(p for p in GTSDB_DIR.glob("*.ppm") if p.stem.isdigit() and int(p.stem) >= FIRST_TEST_IMAGE)


def crop(image_bgr: np.ndarray, box: tuple[int, int, int, int], margin: float = 0.15) -> np.ndarray:
    """The sign with a small margin around it, for the gallery."""
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = box
    mx, my = int((x2 - x1) * margin), int((y2 - y1) * margin)
    x1, y1 = max(0, x1 - mx), max(0, y1 - my)
    x2, y2 = min(w, x2 + mx), min(h, y2 + my)
    return image_bgr[y1:y2, x1:x2]


def to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def to_png(image_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image_bgr)
    return buf.tobytes() if ok else b""


def add_to_history(name: str, result, device: str, kind: str = "image") -> None:
    """Remember the analyses of this session (shown on the dashboard)."""
    recognised = sum(d.recognised for d in result.detections)
    st.session_state.setdefault("history", []).append({
        "heure": datetime.now().strftime("%H:%M:%S"),
        "type": kind,
        "fichier": name,
        "panneaux": len(result.detections),
        "reconnus": recognised,
        "non reconnus (?)": len(result.detections) - recognised,
        "temps (ms)": result.timings_ms.get("total"),
        "calcul": device,
    })


def add_video_to_history(name: str, summary, device: str) -> None:
    """Same table as for images: one line per analysed video (distinct signs, ms per analysed frame)."""
    rows = summary.rows()
    recognised = sum(r["reconnu"] == "oui" for r in rows)
    st.session_state.setdefault("history", []).append({
        "heure": datetime.now().strftime("%H:%M:%S"),
        "type": "vidéo",
        "fichier": name,
        "panneaux": len(rows),
        "reconnus": recognised,
        "non reconnus (?)": len(rows) - recognised,
        "temps (ms)": round(summary.mean_ms_per_analysed_frame, 1),
        "calcul": device,
    })


def classify_closeup(image_bgr: np.ndarray, settings: dict):
    """CNN only on the central square. Returns (top3 [(id, name, p)], ms, device)."""
    import time

    pipe = configured_pipeline(settings)
    t0 = time.perf_counter()
    top = pipe.classify_closeup(image_bgr, top_k=3)
    return top, (time.perf_counter() - t0) * 1000, pipe.device


def draw_central_square(image_bgr: np.ndarray) -> np.ndarray:
    from roadsigns.pipeline import central_square

    out = image_bgr.copy()
    x1, y1, x2, y2 = central_square(out.shape[1], out.shape[0])
    cv2.rectangle(out, (x1, y1), (x2, y2), (80, 200, 60), max(2, out.shape[1] // 300))
    return out


def add_closeup_to_history(name: str, top, threshold: float, ms: float, device: str) -> None:
    recognised = top[0][2] >= threshold
    st.session_state.setdefault("history", []).append({
        "heure": datetime.now().strftime("%H:%M:%S"),
        "type": "webcam (gros plan)",
        "fichier": name,
        "panneaux": 1,
        "reconnus": int(recognised),
        "non reconnus (?)": int(not recognised),
        "temps (ms)": round(ms, 1),
        "calcul": device,
    })
