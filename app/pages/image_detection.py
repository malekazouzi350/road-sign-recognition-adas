"""Image detection: one road image -> detected and recognised signs (detector + CNN)."""

from __future__ import annotations

import streamlit as st

from components import detection as D
from components.image_view import show_analysis
from components.ui import page_header

page_header("02", "Détection", "Détection sur image",
            "Le détecteur YOLO trouve les panneaux, puis notre CNN les reconnaît parmi 43 classes.")

# 1. Models present on this computer? -----------------------------------------------------------
missing = D.missing_weights()
if missing:
    st.error("Modèles absents sur cet ordinateur : " + ", ".join(missing), icon=":material/error:")
    st.caption("Les poids ne sont pas dans Git : récupérez-les sur le Google Drive de l'équipe "
               "et placez-les dans models/weights/, puis rechargez la page.")
    st.stop()

settings = D.current_settings()
st.caption(
    f"Réglages : seuil détecteur {settings['det_conf']:.2f} · seuil CNN {settings['cnn_conf']:.2f} · "
    f"{settings['imgsz']} px · calcul {settings['device']}  (modifiables dans « Paramètres »)"
)

# 2. Choose an image ------------------------------------------------------------------------------
examples = D.example_images()
sources = ["Importer une image"] + (["Exemple GTSDB (images de test)"] if examples else [])
source = st.radio("Source", sources, horizontal=True, label_visibility="collapsed")

name, data = None, None
if source == "Importer une image":
    uploaded = st.file_uploader("Image de route (JPG, PNG, PPM, BMP)",
                                type=["jpg", "jpeg", "png", "ppm", "bmp", "webp"])
    if uploaded is not None:
        name, data = uploaded.name, uploaded.getvalue()
else:
    choice = st.selectbox("Image de test GTSDB (jamais vue pendant l'entraînement)",
                          [p.name for p in examples])
    name = choice
    data = next(p for p in examples if p.name == choice).read_bytes()

if data is None:
    st.info("Importez une photo prise depuis une voiture. Le modèle a été entraîné sur des routes "
            "allemandes (GTSDB) : les panneaux d'autres pays peuvent être moins bien reconnus.",
            icon=":material/upload:")
    st.stop()

show_analysis(name, data, settings)

st.caption("Prototype académique d'aide à la conduite : il ne doit pas servir à contrôler un vrai véhicule.")
