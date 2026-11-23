"""Road-sign detection & recognition (ADAS): Streamlit application entry point.

Run from the repository root (venv active):
    streamlit run app/app.py

Pages are added only when they actually work. Image detection uses the detector + CNN
pipeline (src/roadsigns/pipeline.py), video detection uses src/roadsigns/video.py;
the webcam page will follow.
"""

from __future__ import annotations

import streamlit as st

from components.settings import init_settings
from components.ui import inject_css

st.set_page_config(page_title="Road Sign ADAS", page_icon="🚦", layout="wide")
inject_css()
init_settings()

with st.sidebar:
    st.html(
        '<div class="rs-brand">'
        '<span class="rs-dot" style="background:#0b7a7c"></span>'
        '<span class="rs-dot" style="background:#2e9d6a"></span>'
        '<span class="rs-dot" style="background:#5cc8c0"></span>'
        "&nbsp;Road Sign ADAS</div>"
        '<div class="rs-brand-sub">Détection et reconnaissance des panneaux · prototype académique</div>'
    )

pages = {
    "Projet": [
        st.Page("pages/dashboard.py", title="Tableau de bord", icon=":material/dashboard:", default=True),
    ],
    "Détection": [
        st.Page("pages/image_detection.py", title="Sur image", icon=":material/image_search:"),
        st.Page("pages/video_detection.py", title="Sur vidéo", icon=":material/movie:"),
    ],
    "Modèles": [
        st.Page("pages/evaluation.py", title="Évaluation", icon=":material/insights:"),
    ],
    "Configuration": [
        st.Page("pages/settings.py", title="Paramètres", icon=":material/tune:"),
        st.Page("pages/about.py", title="À propos", icon=":material/info:"),
    ],
}

st.navigation(pages).run()
