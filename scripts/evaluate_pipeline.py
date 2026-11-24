"""End-to-end evaluation of the complete pipeline (detector + CNN) on the GTSDB test images.

For each of the 300 official test images (00600-00899), the pipeline must FIND each
sign (IoU >= 0.5 with a ground-truth box) AND NAME it correctly (the CNN's class equals
the ground-truth class among 43). Every matched sign falls in exactly one bucket:

  * correct    : named, and the name is right
  * wrong      : named, but the name is wrong      <- the dangerous case for ADAS
  * fallback   : CNN not confident enough -> only the super-category is shown (safe)

and unmatched boxes are missed signs or false alarms.

Headline numbers:
  end-to-end recall    = correct / ground-truth signs   (signs found AND correctly named)
  end-to-end precision = correct / named predictions
A threshold sweep shows how the CNN threshold trades wrong names for fallbacks.

Usage (repository root, venv active, both weight files in models/weights/):
    python scripts/evaluate_pipeline.py
    python scripts/evaluate_pipeline.py --det-conf 0.25 --cnn-conf 0.6 --out results/tmp/pipeline
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from roadsigns.datasets.gtsdb import CLASS_NAMES, parse_gt_file
from roadsigns.evaluation.matching import match_pairs
from roadsigns.pipeline import DEFAULT_CLASSIFIER, DEFAULT_DETECTOR, RoadSignPipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger("evaluate_pipeline")
OFFICIAL_TEST_START = 600
SIZE_BINS = ((0, 32, "< 32 px"), (32, 48, "32-48 px"), (48, 10_000, ">= 48 px"))
SWEEP = (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gtsdb-root", type=Path, default=REPO_ROOT / "data/raw/FullIJCNN2013")
    p.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR)
    p.add_argument("--classifier", type=Path, default=DEFAULT_CLASSIFIER)
    p.add_argument("--det-conf", type=float, default=0.25)
    p.add_argument("--cnn-conf", type=float, default=0.60)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--merge-iou", type=float, default=0.6,
                   help="drop a box that overlaps a more confident one (any family); 0 = keep all boxes")
    p.add_argument("--device", default="auto")
    p.add_argument("--samples", type=int, default=6, help="annotated images to save (errors first)")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "results/pipeline")
    return p.parse_args()


def size_bin(px: int) -> str:
    return next(label for lo, hi, label in SIZE_BINS if lo <= px < hi)


def ratio(a: int, b: int) -> float | None:
    return round(a / b, 4) if b else None


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    try:
        raw = parse_gt_file(args.gtsdb_root / "gt.txt")
        pipe = RoadSignPipeline(args.detector, args.classifier, device=args.device,
                                det_conf=args.det_conf, cnn_conf=args.cnn_conf, imgsz=args.imgsz,
                                merge_iou=args.merge_iou)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1

    gt_by_image = defaultdict(list)
    for a in dict.fromkeys(raw):  # drop the exact duplicate line of gt.txt
        gt_by_image[a.filename].append(a)
    images = sorted(p for p in args.gtsdb_root.glob("*.ppm") if int(p.stem) >= OFFICIAL_TEST_START)
    logger.info("Pipeline on %d test images | device %s | det_conf %.2f | cnn_conf %.2f",
                len(images), pipe.device, args.det_conf, args.cnn_conf)

    matched_records = []  # one per matched sign: (gt_class, cnn_class, cnn_conf, size)
    n_gt = n_pred = 0
    missed_by_size, gt_by_size = defaultdict(int), defaultdict(int)
    false_alarms, per_image_issues, times = 0, {}, []
    wrong_examples = []

    for k, path in enumerate(images, start=1):
        img = cv2.imread(str(path))
        if img is None:
            logger.error("Cannot read %s", path)
            return 1
        res = pipe.predict(img)
        times.append(res.timings_ms["total"])
        gts = gt_by_image.get(path.name, [])
        n_gt += len(gts)
        n_pred += len(res.detections)
        for g in gts:
            gt_by_size[size_bin(max(g.width, g.height))] += 1
        pairs = match_pairs([(d.box, d.det_conf) for d in res.detections], [g.box for g in gts], args.iou)
        matched_pred = {i for i, _ in pairs}
        matched_gt = {j for _, j in pairs}
        issues = 0
        for i, j in pairs:
            d, g = res.detections[i], gts[j]
            matched_records.append((g.class_id, d.cnn_class_id, d.cnn_conf, max(g.width, g.height)))
            if d.recognised and d.class_id != g.class_id:
                issues += 1
                wrong_examples.append({"image": path.name, "true": CLASS_NAMES[g.class_id],
                                       "predicted": d.class_name, "cnn_conf": round(d.cnn_conf, 3)})
        for j, g in enumerate(gts):
            if j not in matched_gt:
                missed_by_size[size_bin(max(g.width, g.height))] += 1
                issues += 1
        fa = len(res.detections) - len(matched_pred)
        false_alarms += fa
        issues += fa
        if issues:
            per_image_issues[path.name] = issues
        if k % 100 == 0:
            logger.info("  %d/%d images", k, len(images))

    rec = np.array([(t, p, c) for t, p, c, _ in matched_records], dtype=float).reshape(-1, 3)

    def buckets(threshold: float) -> dict:
        named = rec[:, 2] >= threshold
        correct = int((named & (rec[:, 0] == rec[:, 1])).sum())
        wrong = int((named & (rec[:, 0] != rec[:, 1])).sum())
        return {"correct": correct, "wrong": wrong, "fallback": int((~named).sum())}

    main_b = buckets(args.cnn_conf)
    matched = len(matched_records)
    metrics = {
        "description": "Complete pipeline (team detector + team CNN) on the official GTSDB test images.",
        "trained_by_team": True,
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "detector": args.detector.name, "classifier": args.classifier.name, "device": pipe.device,
        "images": len(images), "det_conf": args.det_conf, "cnn_conf": args.cnn_conf, "iou": args.iou,
        "merge_iou": args.merge_iou,
        "ground_truth_signs": n_gt, "predicted_boxes": n_pred,
        "detection": {"found": matched, "missed": n_gt - matched, "false_alarms": false_alarms,
                      "recall": ratio(matched, n_gt), "precision": ratio(matched, n_pred)},
        "recognition_of_found_signs": main_b,
        "end_to_end": {
            "recall": ratio(main_b["correct"], n_gt),
            "precision_of_named": ratio(main_b["correct"], main_b["correct"] + main_b["wrong"]),
            "wrong_name_rate": ratio(main_b["wrong"], matched),
        },
        "end_to_end_recall_by_size": {
            lb: ratio(sum(1 for t, p, c, s in matched_records
                          if size_bin(s) == lb and c >= args.cnn_conf and t == p), gt_by_size[lb])
            for _, _, lb in SIZE_BINS},
        "missed_by_size": {lb: f"{missed_by_size[lb]}/{gt_by_size[lb]}" for _, _, lb in SIZE_BINS},
        "cnn_threshold_sweep": {f"{t:.1f}": buckets(t) for t in SWEEP},
        "median_ms_per_image": round(float(np.median(times)), 1) if times else None,
        "wrong_name_examples": wrong_examples[:20],
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    samples_dir = args.out / "samples"
    samples_dir.mkdir(exist_ok=True)
    worst = sorted(per_image_issues, key=per_image_issues.get, reverse=True)[: args.samples // 2]
    good = [p.name for p in images if p.name not in per_image_issues and gt_by_image.get(p.name)]
    for name in worst + good[: args.samples - len(worst)]:
        img = cv2.imread(str(args.gtsdb_root / name))
        tag = "issue" if name in per_image_issues else "ok"
        cv2.imwrite(str(samples_dir / f"{tag}_{Path(name).stem}.jpg"), pipe.annotate(img, pipe.predict(img)),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])

    d, e = metrics["detection"], metrics["end_to_end"]

    def pc(v):
        return "n/a" if v is None else f"{v:.1%}"

    print("\n=== Complete pipeline (detector + CNN) on GTSDB official test images ===")
    print(f"Images {len(images)} | ground-truth signs {n_gt} | device {pipe.device} | "
          f"median {metrics['median_ms_per_image']} ms/image")
    print(f"Detection      : found {d['found']}/{n_gt} (recall {pc(d['recall'])}), "
          f"false alarms {d['false_alarms']} (precision {pc(d['precision'])})")
    print(f"Found signs    : correct {main_b['correct']} | wrong name {main_b['wrong']} | "
          f"fallback to super-category {main_b['fallback']}  (CNN threshold {args.cnn_conf})")
    print(f"END-TO-END     : recall {pc(e['recall'])} (found AND correctly named) | "
          f"precision of named signs {pc(e['precision_of_named'])}")
    print("By size        : " + ", ".join(f"{k} {pc(v)}" for k, v in metrics["end_to_end_recall_by_size"].items()))
    print("Threshold sweep: " + " | ".join(f"{t}: {b['correct']}/{b['wrong']}/{b['fallback']}"
                                           for t, b in metrics["cnn_threshold_sweep"].items())
          + "   (correct/wrong/fallback)")
    print(f"Saved: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
