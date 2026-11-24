"""Webcam: take a photo with the computer's camera and analyse it with the same pipeline."""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from components import detection as D
from components.image_view import show_analysis, show_closeup
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
SCENE, CLOSEUP = "Scène de route (détecteur + CNN)", "Panneau en gros plan (CNN seul)"
mode = st.radio("Mode", [SCENE, CLOSEUP], horizontal=True, label_visibility="collapsed")
if mode == SCENE:
    st.info("Montrez une scène avec des panneaux assez petits (photo de rue sur un téléphone, panneau "
            "tenu à 1-2 m). Le détecteur a appris sur des routes : il ne trouve pas un panneau qui "
            "remplit l'image. Le navigateur demande l'autorisation d'utiliser la caméra.",
            icon=":material/photo_camera:")
else:
    st.info("Placez UN panneau au centre, qu'il remplisse le cadre vert. Le détecteur est sauté : "
            "le centre de la photo est donné directement au CNN.", icon=":material/center_focus_strong:")

photo = st.camera_input("Prendre une photo")
if photo is None:
    st.stop()

# Each new photo gets its own name in the history (the bytes differ, so the analysis is re-run).
if st.session_state.get("webcam_photo_id") != photo.file_id:
    st.session_state["webcam_photo_id"] = photo.file_id
    st.session_state["webcam_photo_name"] = f"webcam_{datetime.now():%H%M%S}.jpg"

if mode == SCENE:
    show_analysis(st.session_state["webcam_photo_name"], photo.getvalue(), settings,
                  state_key="webcam_analysis", kind="webcam")
else:
    show_closeup(st.session_state["webcam_photo_name"], photo.getvalue(), settings)

st.caption("Mode photo : une image à la fois. Un flux vidéo continu demanderait une bibliothèque en plus "
           "(streamlit-webrtc) et un GPU pour être fluide. "
           "Prototype académique : il ne doit pas servir à contrôler un vrai véhicule.")
