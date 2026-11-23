"""Webcam: take a photo with the computer's camera and analyse it with the same pipeline."""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from components import detection as D
from components.image_view import show_analysis
from components.ui import page_header

page_header("04", "Détection", "Détection par webcam",
            "Prenez une photo avec la caméra de l'ordinateur : elle est analysée comme une image importée.")

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
st.info("Pour la démonstration : montrez à la caméra un panneau imprimé ou affiché sur un téléphone, "
        "assez grand et bien éclairé. Le navigateur demande l'autorisation d'utiliser la caméra.",
        icon=":material/photo_camera:")

photo = st.camera_input("Prendre une photo")
if photo is None:
    st.stop()

# Each new photo gets its own name in the history (the bytes differ, so the analysis is re-run).
if st.session_state.get("webcam_photo_id") != photo.file_id:
    st.session_state["webcam_photo_id"] = photo.file_id
    st.session_state["webcam_photo_name"] = f"webcam_{datetime.now():%H%M%S}.jpg"

show_analysis(st.session_state["webcam_photo_name"], photo.getvalue(), settings,
              state_key="webcam_analysis", kind="webcam")

st.caption("Mode photo : une image à la fois. Un flux vidéo continu demanderait une bibliothèque en plus "
           "(streamlit-webrtc) et un GPU pour être fluide. "
           "Prototype académique : il ne doit pas servir à contrôler un vrai véhicule.")
