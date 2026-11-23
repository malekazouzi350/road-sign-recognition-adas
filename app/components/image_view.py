"""Shared view: analyse one image (uploaded file or webcam photo) and show the results."""

from __future__ import annotations

import json

import streamlit as st

from components import detection as D
from components.ui import kpi, kpi_row


def show_analysis(name: str, data: bytes, settings: dict, state_key: str = "image_analysis",
                  kind: str = "image") -> None:
    """Decode, run the pipeline once per (image, settings), then show numbers, image, signs, table."""
    image = D.decode_image(data)
    if image is None:
        st.error(f"« {name} » n'est pas une image lisible. Essayez un fichier JPG ou PNG.",
                 icon=":material/broken_image:")
        st.stop()

    # 3. Run the pipeline (once per image + settings) -------------------------------------------------
    key = D.image_key(data, settings)
    cached = st.session_state.get(state_key)
    if cached is None or cached["key"] != key:
        try:
            with st.spinner("Analyse en cours…"):
                result, annotated, device = D.analyse(image, settings)
        except Exception as exc:  # shown to the user instead of a Python traceback
            st.error(f"L'analyse a échoué : {exc}", icon=":material/error:")
            st.stop()
        cached = {"key": key, "name": name, "result": result, "annotated": annotated, "device": device}
        st.session_state[state_key] = cached
        D.add_to_history(name, result, device, kind)

    result, annotated = cached["result"], cached["annotated"]
    dets = result.detections
    named = [d for d in dets if d.recognised]
    unsure = [d for d in dets if not d.recognised]

    # 4. Key numbers ----------------------------------------------------------------------------------
    w, h = result.image_size
    kpi_row([
        kpi(str(len(dets)), "Panneaux détectés", f"image {w} × {h} px", "teal"),
        kpi(str(len(named)), "Reconnus", f"confiance CNN ≥ {settings['cnn_conf']:.2f}", "green"),
        kpi(str(len(unsure)), "Non reconnus (?)", "seule la famille est affichée",
            "coral" if unsure else "grey"),
        kpi(f"{result.timings_ms['total']:.0f} ms".replace(".", ","), "Temps d'analyse",
            f"détecteur {result.timings_ms['detector']:.0f} ms · CNN {result.timings_ms['classifier']:.0f} ms"
            f" · {cached['device']}", "teal"),
    ])

    # 5. Annotated image ------------------------------------------------------------------------------
    tab_result, tab_original = st.tabs(["Résultat", "Image originale"])
    with tab_result:
        st.image(D.to_rgb(annotated), width="stretch")
    with tab_original:
        st.image(D.to_rgb(image), width="stretch")

    if not dets:
        st.info("Aucun panneau détecté. Si un panneau est visible, essayez de baisser le seuil du "
                "détecteur dans « Paramètres » ou d'utiliser 1280 px pour les petits panneaux.",
                icon=":material/search_off:")
    else:
        # 6. Each sign, cropped ----------------------------------------------------------------------
        st.subheader("Panneaux trouvés")
        per_row, max_shown = 6, 24
        if len(dets) > max_shown:
            st.caption(f"{len(dets)} boîtes : seules les {max_shown} premières sont affichées ici "
                       "(toutes sont dans le tableau). Le seuil du détecteur est peut-être trop bas.")
        for start in range(0, min(len(dets), max_shown), per_row):
            cols = st.columns(per_row)
            for col, (i, d) in zip(cols, enumerate(dets[start:start + per_row], start=start + 1)):
                with col:
                    st.image(D.to_rgb(D.crop(image, d.box)), width="stretch")
                    if d.recognised:
                        st.markdown(f"**{i}. {d.class_name}**  \nCNN {d.cnn_conf:.2f}")
                    else:
                        st.markdown(f"**{i}. {d.super_category} (?)**  \nCNN {d.cnn_conf:.2f}, trop faible")

        if unsure:
            st.caption("« (?) » : le CNN n'est pas assez sûr de lui, donc l'application ne donne pas de nom "
                       "et affiche seulement la famille trouvée par le détecteur. C'est voulu : mieux vaut "
                       "« je ne sais pas » qu'un mauvais nom. Souvent, c'est un panneau hors des 43 classes.")
        disagree = [i for i, d in enumerate(dets, start=1) if d.recognised and not d.agrees_with_detector]
        if disagree:
            st.caption(f"Panneau(x) {', '.join(map(str, disagree))} : le CNN et le détecteur ne donnent pas "
                       "la même famille. Le nom affiché vient du CNN ; à vérifier visuellement.")

        # 7. Table -----------------------------------------------------------------------------------
        with st.expander("Tableau détaillé (boîtes et confiances)"):
            st.dataframe(result.to_rows(), hide_index=True, width="stretch")

    # 8. Downloads ------------------------------------------------------------------------------------
    stem = name.rsplit(".", 1)[0]
    c1, c2, _ = st.columns([1, 1, 2])
    with c1:
        st.download_button("Télécharger l'image annotée", D.to_png(annotated), f"{stem}_annotated.png",
                           "image/png", icon=":material/download:")
    with c2:
        payload = {"image": name, "settings": settings, **result.to_dict()}
        st.download_button("Télécharger les résultats (JSON)",
                           json.dumps(payload, ensure_ascii=False, indent=2, default=list),
                           f"{stem}_results.json", "application/json", icon=":material/data_object:")

