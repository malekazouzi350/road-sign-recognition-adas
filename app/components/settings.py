"""Application settings, kept in the Streamlit session (one per browser tab)."""

from __future__ import annotations

import streamlit as st

DEFAULTS = {
    "det_conf": 0.25,   # detector confidence threshold
    "imgsz": 1024,      # detector input size (the model was trained at 1024)
    "cnn_conf": 0.60,   # below this CNN confidence, only the super-category is shown
    "device": "auto",   # auto | cpu | cuda
}


def init_settings() -> None:
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)


def get(key: str):
    init_settings()
    return st.session_state[key]
