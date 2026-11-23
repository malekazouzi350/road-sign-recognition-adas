"""Dashboard: real state of the project on THIS computer (models, data, measured results)."""

from __future__ import annotations

import streamlit as st

from components import results as R
from components.ui import kpi, kpi_row, page_header, status_line

page_header("01", "Tableau de bord", "État du système",
            "Ce que cet ordinateur possède, et les résultats mesurés par nos scripts.")

det = R.load_metrics("detector")
cnn = R.load_metrics("classifier")
cnn_gtsdb = R.load_metrics("classifier_on_gtsdb")
base = R.load_metrics("baseline")

cards = [
    kpi(R.pct(base["task_a_any_sign"]["recall"]) if base else "—",
        "YOLO générique (baseline)", "panneaux trouvés, sans entraînement", "coral" if base else "grey"),
    kpi(R.pct(det["test"]["recall"]) if det else "—",
        "Notre détecteur", "rappel sur les 300 images de test" if det else "Not evaluated yet",
        "teal" if det else "grey"),
    kpi(R.pct(cnn["test_accuracy"], 2) if cnn else "—",
        "Notre CNN (GTSRB)", "accuracy, 43 classes" if cnn else "Not evaluated yet", "green" if cnn else "grey"),
    kpi(R.pct(cnn_gtsdb["exact_boxes"]["accuracy"], 2) if cnn_gtsdb else "—",
        "CNN sur GTSDB", "panneaux jamais vus" if cnn_gtsdb else "Not evaluated yet",
        "green" if cnn_gtsdb else "grey"),
]
kpi_row(cards)

left, right = st.columns(2, gap="large")

with left:
    st.subheader("Modèles sur cet ordinateur")
    w = R.weights_status()
    lines = [
        status_line(w["detector"] is not None,
                    f"Détecteur YOLO26n ({R.DETECTOR_RUN})" + ("" if w["detector"] else " : poids absents")),
        status_line(w["classifier"] is not None,
                    "CNN de reconnaissance" + ("" if w["classifier"] else " : poids absents")),
    ]
    st.html("".join(lines))
    if not all(w.values()):
        st.caption("Les poids ne sont pas dans Git : récupérez-les sur le Google Drive de l'équipe "
                   "et placez-les dans models/weights/.")

with right:
    st.subheader("Données sur cet ordinateur")
    st.html("".join(status_line(ok, name) for name, ok in R.datasets_status().items()))
    if not all(R.datasets_status().values()):
        st.caption("Voir data/README.md pour télécharger les données, puis python scripts/prepare_gtsdb.py.")

st.subheader("Analyses de cette session")
history = st.session_state.get("history", [])
if history:
    st.dataframe(list(reversed(history)), hide_index=True, width="stretch")
    st.caption("Ces analyses sont gardées tant que l'onglet du navigateur reste ouvert.")
else:
    st.info("Aucune analyse pour l'instant : utilisez les pages « Détection › Sur image » ou « Sur vidéo ».",
            icon=":material/schedule:")
