"""About: project, approach, data, team and honest limitations."""

from __future__ import annotations

import streamlit as st

from components.ui import page_header

page_header("06", "À propos", "Détection et reconnaissance des panneaux de signalisation par IA",
            "Projet universitaire de vision par ordinateur pour l'aide à la conduite (ADAS).")

left, right = st.columns(2, gap="large")
with left:
    st.subheader("Approche en deux étapes")
    st.markdown(
        "1. **Détecteur YOLO26n** : trouve les panneaux dans la scène, en 4 super-catégories "
        "(interdiction, danger, obligation, autre). Entraîné sur **GTSDB**.\n"
        "2. **CNN de l'équipe** : reconnaît la classe exacte parmi 43 sur chaque panneau découpé. "
        "Entraîné sur **GTSRB**.\n"
        "3. Si la confiance du CNN est faible, seule la super-catégorie est affichée.")
    st.subheader("Données")
    st.markdown(
        "- **GTSDB** : 900 scènes routières (1360×800), 1 212 panneaux annotés.\n"
        "- **GTSRB** : 39 209 panneaux découpés, 43 classes, en séquences de 30 images.\n"
        "- Les deux jeux viennent de la Ruhr-Universität Bochum (Allemagne) et partagent les mêmes 43 classes.")
with right:
    st.subheader("Équipe")
    st.markdown("- **Malek Azzouzi** : modèles (détecteur, CNN, pipeline)\n"
                "- **Nourhene Benkhessib** : données et application\n"
                "- **Tasbih Soussi** : environnement GPU, table des panneaux, jeu de test tunisien")
    st.subheader("Limites")
    st.markdown(
        "- **Prototype académique** : ce n'est pas un système de sécurité certifié ; il ne doit pas "
        "piloter un véhicule.\n"
        "- Entraîné sur des **panneaux allemands** : les panneaux tunisiens ne sont pas encore évalués.\n"
        "- Erreurs connues dans des conditions extrêmes (nuit, contre-jour, flou, panneaux de moins de 20 px).")
