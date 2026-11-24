"""Read the REAL results produced by our scripts (results/*/metrics.json, figures).

Nothing here computes or invents a number: if a file is missing, the caller shows
"Not evaluated yet" with the command that produces it.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"
WEIGHTS = REPO_ROOT / "models" / "weights"
DATA_RAW = REPO_ROOT / "data" / "raw"

DETECTOR_RUN = "v2_1280"

# Where each result lives, and the command that creates it.
SOURCES = {
    "baseline": (RESULTS / "baseline", "python scripts/baseline_pretrained.py"),
    "detector": (RESULTS / "detector" / DETECTOR_RUN, f"python scripts/train_detector.py --name {DETECTOR_RUN}"),
    "classifier": (RESULTS / "classifier", "python scripts/train_classifier.py"),
    "classifier_on_gtsdb": (RESULTS / "classifier_on_gtsdb", "python scripts/evaluate_classifier_gtsdb.py"),
    "dataset": (RESULTS / "dataset" / "super", "python scripts/prepare_gtsdb.py"),
    "exploration": (RESULTS / "exploration", "notebooks/01_gtsdb_exploration.ipynb (Run All)"),
}


@st.cache_data(show_spinner=False)
def _read_json(path_str: str, mtime: float) -> dict:
    # mtime is part of the cache key: a re-run script invalidates the cache.
    return json.loads(Path(path_str).read_text(encoding="utf-8"))


def load_metrics(key: str, filename: str = "metrics.json") -> dict | None:
    """Return the parsed JSON for a result, or None if it has not been produced yet."""
    folder, _ = SOURCES[key]
    path = folder / filename
    if not path.is_file():
        return None
    try:
        return _read_json(str(path), path.stat().st_mtime)
    except (OSError, json.JSONDecodeError):
        return None


def figure(key: str, filename: str) -> Path | None:
    path = SOURCES[key][0] / filename
    return path if path.is_file() else None


def command_for(key: str) -> str:
    return SOURCES[key][1]


def weights_status() -> dict[str, Path | None]:
    """Which trained models are present on this computer (weights are not in Git)."""
    files = {
        "detector": WEIGHTS / f"detector_{DETECTOR_RUN}.pt",
        "classifier": WEIGHTS / "classifier_best.pt",
    }
    return {name: (p if p.is_file() else None) for name, p in files.items()}


def datasets_status() -> dict[str, bool]:
    return {
        "GTSDB": (DATA_RAW / "FullIJCNN2013" / "gt.txt").is_file(),
        "GTSRB": (DATA_RAW / "GTSRB" / "Final_Training" / "Images").is_dir(),
        "GTSDB (YOLO)": (REPO_ROOT / "data" / "processed" / "gtsdb_yolo_super" / "data.yaml").is_file(),
    }


def pct(value: float | None, digits: int = 1) -> str:
    """0.9533 -> '95,3 %' (French formatting). None -> '—'."""
    if value is None:
        return "—"
    return f"{value * 100:.{digits}f} %".replace(".", ",")
