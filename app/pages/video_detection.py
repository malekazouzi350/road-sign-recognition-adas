"""Video detection: a driving video -> annotated video + list of the signs seen."""

from __future__ import annotations

import csv
import hashlib
import io
import tempfile
from pathlib import Path

import streamlit as st

from components import detection as D
from components.ui import kpi, kpi_row, page_header

page_header("03", "Détection", "Détection sur vidéo",
            "Le même pipeline que pour les images, appliqué image par image à une vidéo de conduite.")

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

uploaded = st.file_uploader("Vidéo de conduite (MP4, AVI, MOV, MKV, WebM)",
                            type=["mp4", "avi", "mov", "mkv", "webm"])
if uploaded is None:
    st.info("Importez une courte vidéo filmée depuis une voiture (10 à 30 secondes suffisent). "
            "Sur un ordinateur sans GPU, l'analyse prend environ 1 seconde par image analysée.",
            icon=":material/upload:")
    st.stop()

# OpenCV reads videos from a file path: keep the upload in a temporary folder for this session.
data = uploaded.getvalue()
digest = hashlib.sha1(data).hexdigest()[:16]
workdir = Path(tempfile.gettempdir()) / "roadsigns_video"
workdir.mkdir(exist_ok=True)
input_path = workdir / f"{digest}{Path(uploaded.name).suffix.lower()}"
if not input_path.exists():
    input_path.write_bytes(data)

from roadsigns.video import process_video, read_info  # noqa: E402  (after the weights check)

try:
    info = read_info(input_path)
except ValueError:
    st.error(f"« {uploaded.name} » n'est pas une vidéo lisible. Essayez un fichier MP4.",
             icon=":material/broken_image:")
    st.stop()

from roadsigns.video import output_size  # noqa: E402

reduced = output_size(info.width, info.height, 1280)
st.caption(f"Vidéo : {info.width} × {info.height} px · {info.fps:.0f} images/s · "
           f"{info.frame_count} images · {info.duration_s:.1f} s"
           + (f" · réduite à {reduced[0]} × {reduced[1]} px pour l'analyse" if reduced[0] != info.width else ""))

# Options -----------------------------------------------------------------------------------------
gpu = settings["device"] == "cuda" or (settings["device"] == "auto" and D.cuda_available())
c1, c2 = st.columns(2)
with c1:
    every_n = st.select_slider(
        "Analyser une image sur…", options=[1, 2, 3, 5, 10, 15], value=1 if gpu else 5,
        help="Les images non analysées gardent les boîtes de la dernière image analysée.")
with c2:
    max_seconds = st.slider("Durée maximale analysée (s)", 5, 120, min(30, max(5, int(info.duration_s) or 5)), 5)

n_frames = min(info.frame_count, int(max_seconds * info.fps)) if info.frame_count else int(max_seconds * info.fps)
st.caption(f"{n_frames} images lues, dont environ {-(-n_frames // every_n)} analysées par les modèles.")

key = (f"{digest}|{every_n}|{max_seconds}|{settings['det_conf']}|{settings['cnn_conf']}|"
       f"{settings['imgsz']}|{settings['device']}")
cached = st.session_state.get("video_analysis")

if st.button("Lancer l'analyse", type="primary", icon=":material/play_arrow:"):
    bar = st.progress(0.0, text="Analyse de la vidéo…")
    try:
        pipe = D.configured_pipeline(settings)
        summary = process_video(pipe, input_path, workdir / f"{digest}_annotated", every_n=every_n,
                                max_seconds=max_seconds,
                                progress=lambda f: bar.progress(f, text=f"Analyse de la vidéo… {f:.0%}"))
    except Exception as exc:  # shown to the user instead of a Python traceback
        bar.empty()
        st.error(f"L'analyse a échoué : {exc}", icon=":material/error:")
        st.stop()
    bar.empty()
    cached = {"key": key, "name": uploaded.name, "summary": summary, "device": pipe.device}
    st.session_state["video_analysis"] = cached
    D.add_video_to_history(uploaded.name, summary, pipe.device)

if cached is None or cached["key"] != key:
    st.stop()

# Results ------------------------------------------------------------------------------------------
summary = cached["summary"]
rows = summary.rows()
named = [r for r in rows if r["reconnu"] == "oui"]
ms = summary.mean_ms_per_analysed_frame
kpi_row([
    kpi(str(summary.analysed_frames), "Images analysées", f"une sur {summary.every_n}", "teal"),
    kpi(str(len(named)), "Panneaux différents reconnus", "sur toute la vidéo", "green"),
    kpi(str(len(rows) - len(named)), "Familles non reconnues (?)", "confiance CNN trop faible",
        "coral" if len(rows) > len(named) else "grey"),
    kpi(f"{ms:.0f} ms".replace(".", ","), "Par image analysée",
        f"≈ {1000 / ms:.1f} images/s · {cached['device']}".replace(".", ",") if ms else cached["device"], "teal"),
])

video_path = summary.output_path
if video_path and video_path.exists():
    if summary.playable_in_browser:
        st.video(str(video_path))
    else:
        st.info("Cette vidéo ne peut pas être lue dans le navigateur sur cet ordinateur : "
                "téléchargez-la ci-dessous.", icon=":material/info:")

if rows:
    st.subheader("Panneaux vus dans la vidéo")
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption("Chaque image analysée est traitée seule (pas encore de suivi d'une image à l'autre). "
               "« confirmé » : vu sur au moins 2 images analysées, et le détecteur et le CNN sont d'accord "
               "sur la famille. « à vérifier » : vu une seule fois, ou désaccord détecteur / CNN "
               "(le nom affiché vient du CNN) : regardez la vidéo à ce moment-là.")
else:
    st.info("Aucun panneau détecté dans la partie analysée de la vidéo.", icon=":material/search_off:")

stem = cached["name"].rsplit(".", 1)[0]
b1, b2, _ = st.columns([1, 1, 2])
with b1:
    if video_path and video_path.exists():
        st.download_button("Télécharger la vidéo annotée", video_path.read_bytes(),
                           f"{stem}_annotated{video_path.suffix}",
                           "video/webm" if video_path.suffix == ".webm" else "video/mp4",
                           icon=":material/download:")
with b2:
    if rows:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        st.download_button("Télécharger la liste (CSV)", buf.getvalue().encode("utf-8-sig"),
                           f"{stem}_panneaux.csv", "text/csv", icon=":material/table:")

st.caption("Prototype académique d'aide à la conduite : il ne doit pas servir à contrôler un vrai véhicule.")
