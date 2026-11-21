"""Train and evaluate the team's CNN on GTSRB (43 classes), the second stage of the pipeline.

Steps:
  1. read the GTSRB training set and split it by TRACK into train / val / test
     (70 / 15 / 15 per class, fixed seed, no near-duplicate frames across splits);
  2. crop each sign to its ROI and resize to 48x48 (cached in data/processed/);
  3. train TrafficSignCNN with realistic augmentation, keep the best epoch on val;
  4. evaluate that best model ONCE on the held-out test tracks.

Outputs:
  models/weights/classifier_best.pt   (git-ignored: share it via Drive)
  results/classifier/metrics.json, history.csv, classification_report.txt,
  training_curves.png, confusion_matrix.png, per_class_f1.png, test_errors.png

Usage (repository root, venv active):
    python scripts/train_classifier.py                  # GPU if available
    python scripts/train_classifier.py --epochs 3       # quick check
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no window needed: figures are written to files
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader

from roadsigns.classification.data import SignCropDataset, load_crops
from roadsigns.classification.model import INPUT_SIZE, MEAN, STD, TrafficSignCNN, count_parameters
from roadsigns.datasets.gtsdb import CLASS_NAMES, NUM_CLASSES
from roadsigns.datasets.gtsrb import load_gtsrb_training, split_by_track

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger("train_classifier")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gtsrb-root", type=Path, default=REPO_ROOT / "data/raw/GTSRB/Final_Training/Images")
    p.add_argument("--cache", type=Path, default=REPO_ROOT / "data/processed/gtsrb_crops_48.npz")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", default="auto", help="'auto', 'cpu' or 'cuda'")
    p.add_argument("--workers", type=int, default=0, help="DataLoader workers (0 is safest on Windows)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "results/classifier")
    p.add_argument("--weights-out", type=Path, default=REPO_ROOT / "models/weights/classifier_best.pt")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, criterion, device, optimizer=None) -> tuple[float, float]:
    """One pass over ``loader``. Trains if an optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
    return total_loss / seen, correct / seen


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    ys, preds, confs = [], [], []
    for x, y in loader:
        probs = torch.softmax(model(x.to(device)), dim=1)
        conf, pred = probs.max(1)
        ys.append(y.numpy())
        preds.append(pred.cpu().numpy())
        confs.append(conf.cpu().numpy())
    return np.concatenate(ys), np.concatenate(preds), np.concatenate(confs)


def plot_curves(history: list[dict], path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    a1.plot(epochs, [h["train_loss"] for h in history], "o-", label="train")
    a1.plot(epochs, [h["val_loss"] for h in history], "o-", label="validation")
    a1.set(xlabel="Epoch", ylabel="Loss", title="Loss")
    a2.plot(epochs, [h["train_acc"] for h in history], "o-", label="train")
    a2.plot(epochs, [h["val_acc"] for h in history], "o-", label="validation")
    a2.set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy")
    for a in (a1, a2):
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(y, pred, path: Path) -> None:
    cm = confusion_matrix(y, pred, labels=range(NUM_CLASSES)).astype(float)
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set(xlabel="Predicted class", ylabel="True class",
           title="Confusion matrix on test tracks (row-normalised)",
           xticks=range(NUM_CLASSES), yticks=range(NUM_CLASSES))
    ax.tick_params(labelsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_per_class_f1(per_class_f1: np.ndarray, support: np.ndarray, path: Path) -> None:
    order = np.argsort(per_class_f1)
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.barh([f"{c:>2} {CLASS_NAMES[c]} (n={support[c]})" for c in order], per_class_f1[order], color="#00798c")
    ax.set(xlim=(0, 1), xlabel="F1-score", title="Per-class F1 on test tracks (worst at the bottom)")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_errors(images, y, pred, conf, path: Path, n: int = 24) -> None:
    wrong = np.flatnonzero(y != pred)
    if len(wrong) == 0:
        return
    wrong = wrong[np.argsort(-conf[wrong])][:n]  # most confident mistakes first
    cols = 8
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.9))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, i in zip(np.atleast_1d(axes).ravel(), wrong):
        ax.imshow(images[i])
        ax.set_title(f"true {y[i]}\npred {pred[i]} ({conf[i]:.2f})", fontsize=7)
    fig.suptitle("Test errors (most confident first)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    logger.info("Device: %s%s", device, f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")

    try:
        samples = load_gtsrb_training(args.gtsrb_root)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    splits = split_by_track(samples, seed=args.seed)
    for name, part in splits.items():
        logger.info("%-5s: %5d images, %4d tracks", name, len(part), len({s.track for s in part}))

    images, labels = load_crops(samples, args.cache)
    index = {str(s.path): i for i, s in enumerate(samples)}
    idx = {name: np.array([index[str(s.path)] for s in part]) for name, part in splits.items()}

    pin = device.type == "cuda"
    loaders = {
        "train": DataLoader(SignCropDataset(images[idx["train"]], labels[idx["train"]], augment=True),
                            batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=pin),
        "val": DataLoader(SignCropDataset(images[idx["val"]], labels[idx["val"]]),
                          batch_size=256, num_workers=args.workers, pin_memory=pin),
        "test": DataLoader(SignCropDataset(images[idx["test"]], labels[idx["test"]]),
                           batch_size=256, num_workers=args.workers, pin_memory=pin),
    }

    model = TrafficSignCNN(NUM_CLASSES).to(device)
    logger.info("TrafficSignCNN: %s trainable parameters", f"{count_parameters(model):,}")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    args.out.mkdir(parents=True, exist_ok=True)
    args.weights_out.parent.mkdir(parents=True, exist_ok=True)
    history, best_val, best_epoch = [], -1.0, 0
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        tr_loss, tr_acc = run_epoch(model, loaders["train"], criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, loaders["val"], criterion, device)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": round(tr_loss, 4), "train_acc": round(tr_acc, 4),
                        "val_loss": round(va_loss, 4), "val_acc": round(va_acc, 4)})
        mark = ""
        if va_acc > best_val:
            best_val, best_epoch, mark = va_acc, epoch, "  <- best"
            torch.save({"model_state": model.state_dict(), "num_classes": NUM_CLASSES,
                        "input_size": INPUT_SIZE, "mean": MEAN, "std": STD, "class_names": CLASS_NAMES,
                        "epoch": epoch, "val_acc": va_acc, "seed": args.seed}, args.weights_out)
        logger.info("epoch %2d/%d  train loss %.3f acc %.4f | val loss %.3f acc %.4f | %.0fs%s",
                    epoch, args.epochs, tr_loss, tr_acc, va_loss, va_acc, time.perf_counter() - t0, mark)
    train_minutes = (time.perf_counter() - start) / 60

    with (args.out / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    plot_curves(history, args.out / "training_curves.png")

    # Final, single evaluation of the best checkpoint on the untouched test tracks.
    model.load_state_dict(torch.load(args.weights_out, map_location=device)["model_state"])
    y, pred, conf = predict(model, loaders["test"], device)
    test_acc = float((y == pred).mean())
    macro_f1 = float(f1_score(y, pred, average="macro", labels=range(NUM_CLASSES), zero_division=0))
    per_class = f1_score(y, pred, average=None, labels=range(NUM_CLASSES), zero_division=0)
    support = np.bincount(y, minlength=NUM_CLASSES)

    (args.out / "classification_report.txt").write_text(
        classification_report(y, pred, labels=range(NUM_CLASSES), target_names=list(CLASS_NAMES),
                              digits=4, zero_division=0), encoding="utf-8")
    plot_confusion(y, pred, args.out / "confusion_matrix.png")
    plot_per_class_f1(per_class, support, args.out / "per_class_f1.png")
    plot_errors(images[idx["test"]], y, pred, conf, args.out / "test_errors.png")

    worst = np.argsort(per_class)[:5]
    metrics = {
        "description": "Team-written CNN trained on GTSRB, evaluated on held-out tracks.",
        "trained_by_team": True,
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": str(device),
        "parameters": count_parameters(model),
        "split": "by track, 70/15/15 per class, seed %d" % args.seed,
        "images": {k: int(len(v)) for k, v in idx.items()},
        "epochs": args.epochs, "best_epoch": best_epoch, "best_val_acc": round(best_val, 4),
        "test_accuracy": round(test_acc, 4), "test_macro_f1": round(macro_f1, 4),
        "test_errors": int((y != pred).sum()),
        "worst_classes_f1": {f"{c} {CLASS_NAMES[c]}": round(float(per_class[c]), 4) for c in worst},
        "training_minutes": round(train_minutes, 1),
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n=== CNN classifier on GTSRB (held-out test tracks) ===")
    print(f"Train/val/test images : {metrics['images']}")
    print(f"Best epoch            : {best_epoch} (val acc {best_val:.2%})")
    print(f"Test accuracy         : {test_acc:.2%}  ({metrics['test_errors']} errors / {len(y)})")
    print(f"Test macro F1         : {macro_f1:.4f}")
    print("Weakest classes (F1)  : " + ", ".join(f"{c} {CLASS_NAMES[c]} {per_class[c]:.2f}" for c in worst))
    print(f"Training time         : {train_minutes:.1f} min on {device}")
    print(f"Saved: {args.weights_out}  and  {args.out}")
    return 0


if __name__ == "__main__":  # required on Windows when DataLoader uses worker processes
    sys.exit(main())
