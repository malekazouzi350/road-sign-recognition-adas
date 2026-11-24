"""Evaluation: every number and figure comes from results/ (produced by our scripts)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components import results as R
from components.ui import kpi, kpi_row, not_evaluated, page_header

page_header("05", "Évaluation", "Résultats mesurés des modèles",
            "Chaque chiffre est lu dans results/ : rien n'est saisi à la main.")


def show_figures(key: str, items: list[tuple[str, str]], columns: int = 2) -> None:
    present = [(R.figure(key, f), cap) for f, cap in items if R.figure(key, f)]
    for i in range(0, len(present), columns):
        cols = st.columns(columns)
        for col, (path, caption) in zip(cols, present[i:i + columns]):
            col.image(str(path), caption=caption, width="stretch")


def _param(v) -> str:
    return " / ".join(map(str, v)) if isinstance(v, (list, tuple)) else str(v)


tab_sys, tab_rob, tab_det, tab_cnn, tab_transfer, tab_base, tab_data = st.tabs(
    ["Système complet", "Robustesse", "Détecteur YOLO", "CNN (GTSRB)", "CNN sur GTSDB",
     "Baseline générique", "Données"])

with tab_sys:
    m = R.load_metrics("pipeline")
    if not m:
        not_evaluated("système complet (détecteur + CNN)", R.command_for("pipeline"))
    else:
        d, e, b = m["detection"], m["end_to_end"], m["recognition_of_found_signs"]
        kpi_row([
            kpi(R.pct(e["recall"]), "Trouvés et bien nommés", f"{b['correct']} / {m['ground_truth_signs']} panneaux",
                "green"),
            kpi(str(b["wrong"]), "Mauvais noms", "le cas dangereux", "green" if b["wrong"] == 0 else "coral"),
            kpi(str(d["false_alarms"]), "Fausses alarmes", f"précision {R.pct(d['precision'])}", "coral"),
            kpi(f"{m['median_ms_per_image']} ms".replace(".", ","), "Par image", m["device"], "teal"),
        ])
        st.caption(f"Détecteur {m['detector']} + CNN {m['classifier']} · {m['images']} images de test GTSDB · "
                   f"seuils détecteur {m['det_conf']} / CNN {m['cnn_conf']} · un panneau compte comme trouvé "
                   f"si sa boîte recouvre la vraie (IoU ≥ {m['iou']}), et comme bien nommé si le CNN donne la "
                   "bonne classe parmi 43.")
        st.write("**Par taille de panneau (trouvés et bien nommés)**")
        st.dataframe(pd.DataFrame([{"Taille": k, "Réussite": R.pct(v)}
                                   for k, v in m["end_to_end_recall_by_size"].items()]),
                     hide_index=True, width="stretch")
        st.write("**Effet du seuil de confiance du CNN**")
        st.dataframe(pd.DataFrame([{"Seuil CNN": t, "Bien nommés": v["correct"], "Mauvais noms": v["wrong"],
                                    "Famille seulement (?)": v["fallback"]}
                                   for t, v in m["cnn_threshold_sweep"].items()]),
                     hide_index=True, width="stretch")

with tab_rob:
    m = R.load_metrics("robustness")
    if not m:
        not_evaluated("robustesse (conditions simulées)", R.command_for("robustness"))
    else:
        st.warning("Conditions **simulées** sur les images de test (assombrissement, flou, voile, bruit) : "
                   "elles montrent à quelle vitesse le système se dégrade, pas son comportement sur de "
                   "vraies photos de nuit ou à contre-jour.", icon=":material/science:")
        show_figures("robustness", [("end_to_end_recall.png", "Panneaux trouvés et bien nommés selon la "
                                     "dégradation")], columns=1)
        rows = [{"Condition": "Images d'origine", "Niveau": "—",
                 "Trouvés et bien nommés": R.pct(m["clean"]["end_to_end_recall"]),
                 "Mauvais noms": m["clean"]["wrong"], "Famille seulement (?)": m["clean"]["fallback"],
                 "Fausses alarmes": m["clean"]["false_alarms"]}]
        for cond in m["conditions"].values():
            for lv, r in cond["results"].items():
                rows.append({"Condition": cond["label"], "Niveau": f"{lv} ({_param(cond['levels'][lv])})",
                             "Trouvés et bien nommés": R.pct(r["end_to_end_recall"]), "Mauvais noms": r["wrong"],
                             "Famille seulement (?)": r["fallback"], "Fausses alarmes": r["false_alarms"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        show_figures("robustness", [("examples.jpg", "Une image de test sous chaque condition (niveau moyen)")],
                     columns=1)

with tab_det:
    m = R.load_metrics("detector")
    if not m:
        not_evaluated("détecteur entraîné", R.command_for("detector"))
    else:
        t = m["test"]
        kpi_row([
            kpi(R.pct(t["recall"]), "Rappel", "panneaux trouvés"),
            kpi(R.pct(t["precision"]), "Précision", "détections justes"),
            kpi(R.pct(t["mAP50"]), "mAP50", "score de détection", "green"),
            kpi(R.pct(t["mAP50_95"]), "mAP50-95", "score strict", "green"),
        ])
        st.caption(f"Run « {m['run']} » · {len(m['classes'])} super-catégories · imgsz {m['imgsz']} · "
                   f"{m['epochs_requested']} époques · test officiel GTSDB (300 images) · "
                   f"{m['inference_ms_per_image']} ms/image · entraînement {m['training_minutes']} min")
        rows = [{"Super-catégorie": c, "Précision": R.pct(v["precision"]), "Rappel": R.pct(v["recall"]),
                 "mAP50": R.pct(v["mAP50"]), "mAP50-95": R.pct(v["mAP50_95"])}
                for c, v in m["test_per_class"].items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        show_figures("detector", [
            ("train_results.png", "Courbes d'entraînement (pertes et métriques par époque)"),
            ("test_confusion_matrix_normalized.png", "Matrice de confusion (test)"),
            ("test_BoxPR_curve.png", "Courbe précision-rappel (test)"),
            ("test_val_batch0_pred.jpg", "Prédictions sur des images de test"),
        ])

with tab_cnn:
    m = R.load_metrics("classifier")
    if not m:
        not_evaluated("CNN sur GTSRB", R.command_for("classifier"))
    else:
        kpi_row([
            kpi(R.pct(m["test_accuracy"], 2), "Accuracy (test)", f"{m['test_errors']} erreurs", "green"),
            kpi(f"{m['test_macro_f1']:.4f}".replace(".", ","), "F1 macro", "43 classes", "green"),
            kpi(str(m["best_epoch"]), "Meilleure époque", f"sur {m['epochs']}"),
            kpi(f"{m['parameters'] / 1e6:.2f} M".replace(".", ","), "Paramètres", "TrafficSignCNN"),
        ])
        st.caption(f"Découpage : {m['split']} · images {m['images']} · entraînement {m['training_minutes']} min")
        st.write("**Classes les plus faibles (F1)**")
        st.dataframe(pd.DataFrame([{"Classe": k, "F1": f"{v:.4f}"} for k, v in m["worst_classes_f1"].items()]),
                     hide_index=True, width="stretch")
        show_figures("classifier", [
            ("training_curves.png", "Courbes d'entraînement"),
            ("per_class_f1.png", "F1 par classe"),
            ("confusion_matrix.png", "Matrice de confusion (test)"),
            ("test_errors.png", "Erreurs sur le test"),
        ])

with tab_transfer:
    m = R.load_metrics("classifier_on_gtsdb")
    if not m:
        not_evaluated("CNN testé sur les panneaux GTSDB", R.command_for("classifier_on_gtsdb"))
    else:
        e, j = m["exact_boxes"], m["jittered_boxes"]
        kpi_row([
            kpi(R.pct(e["accuracy"], 2), "Boîtes exactes", f"{e['errors']} erreurs / {e['n']}", "green"),
            kpi(R.pct(j["accuracy"], 2), f"Boîtes décalées ±{m['jitter']:.0%}", "détecteur imprécis simulé", "green"),
            kpi(f"{e['mean_confidence_correct']:.2f}".replace(".", ","), "Confiance si correct", "moyenne"),
            kpi(f"{e['mean_confidence_wrong']:.2f}".replace(".", ","), "Confiance si erreur", "moyenne", "coral"),
        ])
        size_rows = [{"Taille du panneau": k, "Nombre": v["n"], "Accuracy (exactes)": R.pct(v["accuracy"]),
                      "Accuracy (décalées)": R.pct(j["by_size"][k]["accuracy"])} for k, v in e["by_size"].items()]
        st.dataframe(pd.DataFrame(size_rows), hide_index=True, width="stretch")
        show_figures("classifier_on_gtsdb", [
            ("accuracy_by_size.png", "Accuracy par taille de panneau"),
            ("errors.png", "Les erreurs, les plus confiantes d'abord"),
        ])

with tab_base:
    m = R.load_metrics("baseline")
    if not m:
        not_evaluated("baseline COCO", R.command_for("baseline"))
    else:
        a, b = m["task_a_any_sign"], m["task_b_stop_signs_only"]
        kpi_row([
            kpi(R.pct(a["recall"]), "Rappel, tous panneaux", f"{a['tp']} / {a['ground_truth_signs']}", "coral"),
            kpi(R.pct(a["precision"]), "Précision, tous panneaux", "détections « stop sign »", "grey"),
            kpi(R.pct(b["recall"]), "Rappel, stops", f"{b['tp']} / {b['ground_truth_stop_signs']}", "grey"),
            kpi(R.pct(b["precision"]), "Précision, stops", "confusions avec d'autres panneaux", "coral"),
        ])
        st.caption("Modèle YOLO26n pré-entraîné sur COCO, non entraîné par l'équipe : il ne connaît que le "
                   f"« stop sign ». {m['num_images']} images, imgsz {m['imgsz']}.")
        st.write("**Ce que le modèle générique voit dans nos scènes**")
        st.bar_chart(pd.Series(m["coco_detections_by_class"]).head(8), color="#0b7a7c")

with tab_data:
    rep = R.load_metrics("dataset", "report.json")
    summ = R.load_metrics("exploration", "summary.json")
    if summ:
        kpi_row([
            kpi(str(summ["images"]), "Images GTSDB", f"{summ['images_without_signs']} sans panneau"),
            kpi(str(summ["annotations"] - summ["exact_duplicate_lines"]), "Annotations uniques",
                f"{summ['exact_duplicate_lines']} doublon retiré"),
            kpi(f"{summ['size_px']['median']:.0f} px", "Taille médiane", "des panneaux"),
            kpi(str(summ["classes_lt5_train"]), "Classes < 5 exemples", "sur 43 (train)", "coral"),
        ])
    else:
        not_evaluated("exploration des données", R.command_for("exploration"))
    if rep:
        st.write("**Découpage pour le détecteur (format YOLO)**")
        st.dataframe(pd.DataFrame([{"Partie": s, "Images": c.get("images", 0),
                                    "Sans panneau": c.get("images_without_signs", 0)}
                                   for s, c in rep["splits"].items()]),
                     hide_index=True, width="stretch")
    show_figures("exploration", [
        ("02_class_distribution.png", "Annotations par classe"),
        ("04_sign_sizes.png", "Taille des panneaux"),
    ])
