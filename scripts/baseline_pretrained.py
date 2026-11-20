"""Baseline: run the generic COCO-pretrained YOLO26n on GTSDB, without any training.

COCO has 80 classes and only ONE of them is a traffic sign: "stop sign".
This script measures, honestly, how far a generic detector gets on our task:

  * Task A - "any traffic sign": the model's "stop sign" detections are compared
    with ALL GTSDB signs (43 classes). Expected: very low recall.
  * Task B - "stop signs only": the same detections compared with the GTSDB
    signs of class 14 (Stop) only. This is the fairest possible comparison
    for a COCO model.

It also counts what the model DOES see in road scenes (cars, people, traffic
lights...), which is useful for the presentation.

This is NOT a model trained by our team. Results are saved to
results/baseline/metrics.json and annotated samples to results/baseline/samples/.

Usage (repository root, virtual environment active):
    python scripts/baseline_pretrained.py
    python scripts/baseline_pretrained.py --split test --imgsz 1280 --device cpu
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from roadsigns.datasets.gtsdb import STOP_CLASS_ID, GTSDBDataset, SignAnnotation
from roadsigns.evaluation.matching import MatchResult, match_boxes

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GTSDB = REPO_ROOT / "data" / "raw" / "FullIJCNN2013"
DEFAULT_WEIGHTS = REPO_ROOT / "models" / "pretrained" / "yolo26n.pt"
DEFAULT_OUT = REPO_ROOT / "results" / "baseline"
OFFICIAL_TEST_START = 600  # IJCNN 2013 split: images 00000-00599 train, 00600-00899 test

GT_COLOR = (0, 200, 0)  # BGR green
PRED_COLOR = (0, 0, 255)  # BGR red

logger = logging.getLogger("baseline")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gtsdb-root", type=Path, default=DEFAULT_GTSDB, help="extracted FullIJCNN2013 folder")
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="COCO-pretrained weights")
    p.add_argument("--split", choices=["all", "test"], default="all",
                   help="'test' = official 300 test images (00600-00899)")
    p.add_argument("--imgsz", type=int, default=1280, help="inference size (GTSDB images are 1360x800)")
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    p.add_argument("--iou", type=float, default=0.5, help="IoU threshold for a true positive")
    p.add_argument("--device", default="auto", help="'auto', 'cpu' or a GPU index such as '0'")
    p.add_argument("--samples", type=int, default=8, help="number of annotated images to save")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output folder")
    return p.parse_args()


def select_images(dataset: GTSDBDataset, split: str) -> list[Path]:
    if split == "all":
        return dataset.image_paths
    return [p for p in dataset.image_paths if p.stem.isdigit() and int(p.stem) >= OFFICIAL_TEST_START]


def draw(image: np.ndarray, gts: list[SignAnnotation], preds: list[tuple[tuple, float]]) -> np.ndarray:
    """Ground truth in green, 'stop sign' predictions in red with confidence."""
    out = image.copy()
    for g in gts:
        cv2.rectangle(out, (g.x1, g.y1), (g.x2, g.y2), GT_COLOR, 2)
        cv2.putText(out, f"GT {g.class_id}", (g.x1, max(12, g.y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, GT_COLOR, 1, cv2.LINE_AA)
    for (x1, y1, x2, y2), conf in preds:
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, PRED_COLOR, 2)
        cv2.putText(out, f"stop sign {conf:.2f}", (p1[0], min(out.shape[0] - 4, p2[1] + 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, PRED_COLOR, 1, cv2.LINE_AA)
    cv2.putText(out, "green = ground truth | red = COCO 'stop sign' prediction", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def pick_samples(per_image: dict[str, dict], n: int) -> list[str]:
    """Prefer images with a real stop sign, then the images with the most signs."""
    with_stop = [k for k, v in per_image.items() if v["n_gt_stop"] > 0]
    busiest = sorted(per_image, key=lambda k: per_image[k]["n_gt"], reverse=True)
    chosen: list[str] = []
    for name in with_stop + busiest:
        if name not in chosen:
            chosen.append(name)
        if len(chosen) == n:
            break
    return chosen


def ratio(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    try:
        dataset = GTSDBDataset.load(args.gtsdb_root)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    from ultralytics import YOLO  # imported late so --help works without loading torch

    model = YOLO(str(args.weights))
    name_to_id = {name: idx for idx, name in model.names.items()}
    if "stop sign" not in name_to_id:
        logger.error("These weights have no 'stop sign' class; use COCO-pretrained weights.")
        return 1
    stop_coco_id = name_to_id["stop sign"]

    images = select_images(dataset, args.split)
    logger.info("Model: %s (COCO-pretrained, NOT trained on road signs)", args.weights.name)
    logger.info("Images: %d (%s split), imgsz=%d, conf=%.2f", len(images), args.split, args.imgsz, args.conf)

    predict_kwargs = {"imgsz": args.imgsz, "conf": args.conf, "verbose": False}
    if args.device != "auto":
        predict_kwargs["device"] = args.device

    any_sign, stop_only = MatchResult(), MatchResult()
    coco_counts: Counter[str] = Counter()
    per_image: dict[str, dict] = {}
    stop_preds_by_image: dict[str, list] = {}
    times_ms: list[float] = []
    unreadable: list[str] = []

    for i, path in enumerate(images, start=1):
        image = cv2.imread(str(path))
        if image is None:
            unreadable.append(path.name)
            continue

        start = time.perf_counter()
        result = model.predict(image, **predict_kwargs)[0]
        times_ms.append((time.perf_counter() - start) * 1000)

        classes = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
        coco_counts.update(model.names[c] for c in classes)
        stop_preds = [(tuple(map(float, b)), float(c)) for b, c, k in zip(boxes, confs, classes)
                      if k == stop_coco_id]

        gts = dataset.signs_in(path)
        gt_stop = [g for g in gts if g.class_id == STOP_CLASS_ID]
        any_sign = any_sign + match_boxes(stop_preds, [g.box for g in gts], args.iou)
        stop_only = stop_only + match_boxes(stop_preds, [g.box for g in gt_stop], args.iou)

        per_image[path.name] = {"n_gt": len(gts), "n_gt_stop": len(gt_stop), "n_stop_pred": len(stop_preds)}
        stop_preds_by_image[path.name] = stop_preds
        if i % 100 == 0:
            logger.info("  %d/%d images", i, len(images))

    if not times_ms:
        logger.error("No image could be read.")
        return 1
    if unreadable:
        logger.warning("%d unreadable image(s), e.g. %s", len(unreadable), unreadable[:3])

    # Warm-up makes the first call slow: report the median, not the mean.
    median_ms = float(np.median(times_ms))
    metrics = {
        "description": "COCO-pretrained YOLO26n evaluated on GTSDB without any training (baseline).",
        "trained_by_team": False,
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "weights": args.weights.name,
        "split": args.split,
        "num_images": len(times_ms),
        "imgsz": args.imgsz,
        "conf_threshold": args.conf,
        "iou_threshold": args.iou,
        "device": args.device,
        "median_inference_ms": round(median_ms, 1),
        "task_a_any_sign": {
            "ground_truth_signs": any_sign.tp + any_sign.fn,
            "tp": any_sign.tp, "fp": any_sign.fp, "fn": any_sign.fn,
            "precision": ratio(any_sign.precision), "recall": ratio(any_sign.recall),
        },
        "task_b_stop_signs_only": {
            "ground_truth_stop_signs": stop_only.tp + stop_only.fn,
            "tp": stop_only.tp, "fp": stop_only.fp, "fn": stop_only.fn,
            "precision": ratio(stop_only.precision), "recall": ratio(stop_only.recall),
        },
        "coco_detections_by_class": dict(coco_counts.most_common()),
        "unreadable_images": unreadable,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    samples_dir = args.out / "samples"
    samples_dir.mkdir(exist_ok=True)
    for name in pick_samples(per_image, args.samples):
        image = cv2.imread(str(dataset.root / name))
        annotated = draw(image, dataset.signs_in(Path(name)), stop_preds_by_image[name])
        cv2.imwrite(str(samples_dir / f"{Path(name).stem}.jpg"), annotated)

    def fmt(v: float | None) -> str:
        return "n/a (no predictions)" if v is None else f"{v:.1%}"

    a, b = metrics["task_a_any_sign"], metrics["task_b_stop_signs_only"]
    print("\n=== Baseline: COCO-pretrained YOLO26n on GTSDB (no training) ===")
    print(f"Images: {metrics['num_images']}  |  median inference: {median_ms:.1f} ms/image")
    print(f"Task A, any sign   : {a['tp']}/{a['ground_truth_signs']} signs found  "
          f"recall {fmt(any_sign.recall)}  precision {fmt(any_sign.precision)}")
    print(f"Task B, stop only  : {b['tp']}/{b['ground_truth_stop_signs']} stop signs found  "
          f"recall {fmt(stop_only.recall)}  precision {fmt(stop_only.precision)}")
    print("What COCO sees most:", ", ".join(f"{k} ({v})" for k, v in coco_counts.most_common(6)) or "nothing")
    print(f"Saved: {metrics_path.relative_to(REPO_ROOT) if metrics_path.is_relative_to(REPO_ROOT) else metrics_path}")
    print(f"Saved: {samples_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
