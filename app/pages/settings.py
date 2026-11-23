"""Settings used by the detection pages (kept for the whole session)."""

from __future__ import annotations

import streamlit as st

from components.settings import DEFAULTS, get
from components.ui import page_header

page_header("05", "Paramètres", "Réglages de la détection",
            "Utilisés par la détection sur image et sur vidéo (et plus tard par la webcam).")

st.session_state["det_conf"] = st.slider(
    "Seuil de confiance du détecteur", 0.05, 0.95, float(get("det_conf")), 0.05,
    help="Une boîte n'est affichée que si le détecteur est au moins aussi confiant.")

st.session_state["cnn_conf"] = st.slider(
    "Seuil de confiance du CNN", 0.10, 0.95, float(get("cnn_conf")), 0.05,
    help="En dessous, l'application affiche seulement la super-catégorie (interdiction, danger…). "
         "Mesuré sur GTSDB : confiance moyenne 0,98 si correct, 0,41 si erreur.")

sizes = [640, 1024, 1280]
st.session_state["imgsz"] = st.select_slider(
    "Taille d'image du détecteur (px)", options=sizes, value=int(get("imgsz")),
    help="Le détecteur a été entraîné à 1024. À 640, 36 % des panneaux passeraient sous 16 px.")


@st.cache_resource(show_spinner=False)
def cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


devices = ["auto", "cpu"] + (["cuda"] if cuda_available() else [])
current = get("device") if get("device") in devices else "auto"
st.session_state["device"] = st.radio("Calcul", devices, index=devices.index(current), horizontal=True,
                                      help="« auto » utilise le GPU NVIDIA s'il est disponible.")
if not cuda_available():
    st.caption("Aucun GPU NVIDIA utilisable sur cet ordinateur : les calculs se feront sur le CPU.")

if st.button("Rétablir les valeurs par défaut"):
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.rerun()
