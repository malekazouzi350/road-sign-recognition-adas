"""Evaluate the team CNN (trained on GTSRB only) on the real signs of GTSDB scene images.

Why: the 99.69 % on GTSRB was measured on clean, centred crops from the same
dataset as training. In our pipeline the CNN will receive signs cut out of full
road scenes, with boxes produced by a detector. This script measures how well the
CNN transfers to that situation, BEFORE the detector exists:

  * "exact boxes":    crops from the ground-truth GTSDB boxes
                      -> effect of changing dataset (other scenes, other conditions);
  * "jittered boxes": each box randomly shifted and resized by up to +/-15 %
                      -> simulates an imprecise detector (fixed seed).

Results are broken down by sign size (small signs are the hard case) and by
super-category. GTSDB was never used to train the CNN: this is a true transfer test.
Duplicate annotations in gt.txt are removed first (1212 unique signs).

Outputs: results/classifier_on_gtsdb/metrics.json, accuracy_by_size.png, errors.png

Usage:
    python scripts/evaluate_classifier_gtsdb.py
    python scripts/evaluate_classifier_gtsdb.py --jitter 0.25
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import f1_score

from roadsigns.classification.data import crop_and_resize, preprocess_crop
from roadsigns.classification.model import TrafficSignCNN
from roadsigns.datasets.gtsdb import CLASS_NAMES, NUM_CLASSES, SUPER_CATEGORIES, parse_gt_file, super_category_id

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger("evaluate_classifier_gtsdb")
SIZE_BINS = ((0, 32, "< 32 px"), (32, 48, "32-48 px"), (48, 64, "48-64 px"), (64, 10_000, ">= 64 px"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gtsdb-root", type=Path, default=REPO_ROOT / "data/raw/FullIJCNN2013")
    p.add_argument("--weights", type=Path, default=REPO_ROOT / "models/weights/classifier_best.pt")
    p.add_argument("--jitter", type=float, default=0.15, help="max relative shift/resize of jittered boxes")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "results/classifier_on_gtsdb")
    return p.parse_args()


def jitter_box(box: tuple[int, int, int, int], amount: float, rng: random.Random) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    w, h = x2 - x1 + 1, y2 - y1 + 1
    cx = (x1 + x2) / 2 + rng.uniform(-amount, amount) * w
    cy = (y1 + y2) / 2 + rng.uniform(-amount, amount) * h
    s = 1 + rng.uniform(-amount, amount)
    nw, nh = w * s, h * s
    return (int(round(cx - nw / 2)), int(round(cy - nh / 2)), int(round(cx + nw / 2)), int(round(cy + nh / 2)))


def load_model(weights: Path, device: torch.device) -> tuple[TrafficSignCNN, dict]:
    ckpt = torch.load(weights, map_location=device)
    model = TrafficSignCNN(ckpt.get("num_classes", NUM_CLASSES)).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model.eval(), ckpt


@torch.no_grad()
def classify(model, crops: list[np.ndarray], device) -> tuple[np.ndarray, np.ndarray]:
    preds, confs = [], []
    for i in range(0, len(crops), 256):
        x = torch.stack([preprocess_crop(c) for c in crops[i:i + 256]]).to(device)
        prob = torch.softmax(model(x), dim=1)
        conf, pred = prob.max(1)
        preds.append(pred.cpu().numpy())
        confs.append(conf.cpu().numpy())
    return np.concatenate(preds), np.concatenate(confs)


def summarise(y: np.ndarray, pred: np.ndarray, conf: np.ndarray, sizes: np.ndarray) -> dict:
    ok = y == pred
    by_size = {}
    for lo, hi, label in SIZE_BINS:
        m = (sizes >= lo) & (sizes < hi)
        if m.any():
            by_size[label] = {"n": int(m.sum()), "accuracy": round(float(ok[m].mean()), 4)}
    super_true = np.array([super_category_id(int(c)) for c in y])
    super_pred = np.array([super_category_id(int(c)) for c in pred])
    by_super = {name: {"n": int((super_true == i).sum()), "accuracy": round(float(ok[super_true == i].mean()), 4)}
                for i, name in enumerate(SUPER_CATEGORIES) if (super_true == i).any()}
    return {
        "n": int(len(y)),
        "accuracy": round(float(ok.mean()), 4),
        "macro_f1_present_classes": round(float(f1_score(y, pred, average="macro", labels=np.unique(y),
                                                         zero_division=0)), 4),
        "super_category_accuracy": round(float((super_true == super_pred).mean()), 4),
        "errors": int((~ok).sum()),
        "mean_confidence_correct": round(float(conf[ok].mean()), 4) if ok.any() else None,
        "mean_confidence_wrong": round(float(conf[~ok].mean()), 4) if (~ok).any() else None,
        "by_size": by_size,
        "by_super_category": by_super,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if not args.weights.is_file():
        logger.error("Weights not found: %s (train with scripts/train_classifier.py or get them from Drive)",
                     args.weights)
        return 1
    try:
        raw = parse_gt_file(args.gtsdb_root / "gt.txt")
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    annotations = list(dict.fromkeys(raw))  # drop exact duplicate lines (00340 is listed twice)
    logger.info("GTSDB signs: %d unique (%d exact duplicate removed)", len(annotations), len(raw) - len(annotations))

    rng = random.Random(args.seed)
    by_image = defaultdict(list)
    for a in annotations:
        by_image[a.filename].append(a)
    exact, jittered, y, sizes = [], [], [], []
    for filename, anns in sorted(by_image.items()):
        img = cv2.imread(str(args.gtsdb_root / filename))
        if img is None:
            logger.error("Cannot read %s", filename)
            return 1
        for a in anns:
            box = (a.x1, a.y1, a.x2, a.y2)
            exact.append(crop_and_resize(img, box))
            jittered.append(crop_and_resize(img, jitter_box(box, args.jitter, rng)))
            y.append(a.class_id)
            sizes.append(max(a.width, a.height))
    y, sizes = np.array(y), np.array(sizes)

    model, ckpt = load_model(args.weights, device)
    pred_exact, conf_exact = classify(model, exact, device)
    pred_jit, conf_jit = classify(model, jittered, device)
    res_exact = summarise(y, pred_exact, conf_exact, sizes)
    res_jit = summarise(y, pred_jit, conf_jit, sizes)

    args.out.mkdir(parents=True, exist_ok=True)
    metrics = {
        "description": "Team CNN trained on GTSRB only, evaluated on all GTSDB scene signs (transfer test).",
        "trained_on_gtsdb": False,
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "weights": args.weights.name, "trained_epoch": ckpt.get("epoch"), "device": str(device),
        "classes_present": int(len(np.unique(y))), "jitter": args.jitter, "seed": args.seed,
        "exact_boxes": res_exact, "jittered_boxes": res_jit,
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    labels = [b[2] for b in SIZE_BINS if b[2] in res_exact["by_size"]]
    xs = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for off, res, name, color in ((-0.2, res_exact, "boîtes exactes", "#00798c"),
                                  (0.2, res_jit, f"boîtes décalées ±{args.jitter:.0%}", "#edae49")):
        vals = [res["by_size"][lb]["accuracy"] for lb in labels]
        bars = ax.bar(xs + off, vals, 0.4, label=name, color=color)
        ax.bar_label(bars, [f"{v:.1%}" for v in vals], fontsize=8)
    ax.set_xticks(xs, [f"{lb}\n(n={res_exact['by_size'][lb]['n']})" for lb in labels])
    ax.set(ylim=(0, 1.25), ylabel="Accuracy (43 classes)",
           title="CNN entraîné sur GTSRB, testé sur les panneaux GTSDB, par taille")
    ax.legend(loc="upper center", ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(args.out / "accuracy_by_size.png", dpi=150)
    plt.close(fig)

    wrong = np.flatnonzero(y != pred_exact)
    wrong = wrong[np.argsort(-conf_exact[wrong])][:24]
    if len(wrong):
        cols = 8
        rows = int(np.ceil(len(wrong) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.9))
        for ax in np.atleast_1d(axes).ravel():
            ax.axis("off")
        for ax, i in zip(np.atleast_1d(axes).ravel(), wrong):
            ax.imshow(exact[i])
            ax.set_title(f"vrai {y[i]} / prédit {pred_exact[i]}\n{sizes[i]} px, conf {conf_exact[i]:.2f}", fontsize=6)
        fig.suptitle("Erreurs du CNN sur GTSDB (boîtes exactes, les plus confiantes d'abord)", fontsize=9)
        fig.tight_layout()
        fig.savefig(args.out / "errors.png", dpi=150)
        plt.close(fig)

    print("\n=== Team CNN (trained on GTSRB) evaluated on GTSDB scene signs ===")
    print(f"Signs: {len(y)} ({metrics['classes_present']} classes), never seen during training")
    for name, r in (("Exact boxes", res_exact), (f"Jittered +-{args.jitter:.0%}", res_jit)):
        print(f"{name:<15}: accuracy {r['accuracy']:.2%} ({r['errors']} errors) | macro F1 "
              f"{r['macro_f1_present_classes']:.4f} | super-category accuracy {r['super_category_accuracy']:.2%}")
        print("   by size     : " + ", ".join(f"{k} {v['accuracy']:.1%} (n={v['n']})" for k, v in r["by_size"].items()))
        print(f"   confidence  : correct {r['mean_confidence_correct']} vs wrong {r['mean_confidence_wrong']}")
    if len(wrong):
        pairs = sorted({(int(y[i]), int(pred_exact[i])) for i in wrong[:6]})
        print("Most confident mistakes: " + "; ".join(f"{CLASS_NAMES[t]} -> {CLASS_NAMES[p]}" for t, p in pairs))
    print(f"Saved: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
