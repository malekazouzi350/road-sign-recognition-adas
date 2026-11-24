"""Robustness of the complete pipeline under SIMULATED bad conditions.

The 300 official GTSDB test images are degraded artificially (night, motion blur, haze /
backlight, sensor noise; 3 levels each, see roadsigns.evaluation.corruptions) and the
complete pipeline (detector + CNN, same settings as evaluate_pipeline.py) is scored on
each version: signs found AND correctly named, wrong names, false alarms.

These are simulations: they show HOW FAST the system degrades, not how it behaves on
real night or backlit photos.

Usage (repository root, venv active, weights in models/weights/):
    python scripts/evaluate_robustness.py                      # all conditions, ~3-5 min on GPU
    python scripts/evaluate_robustness.py --limit 30 --out results/tmp/robustness   # quick check
Outputs: results/robustness/metrics.json, end_to_end_recall.png, examples.jpg
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from roadsigns.datasets.gtsdb import parse_gt_file
from roadsigns.evaluation.corruptions import CORRUPTIONS, LEVELS, apply, describe
from roadsigns.evaluation.end_to_end import score_image, summarize
from roadsigns.pipeline import DEFAULT_CLASSIFIER, DEFAULT_DETECTOR, RoadSignPipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger("evaluate_robustness")
OFFICIAL_TEST_START = 600
EXAMPLE_IMAGE = "00813.ppm"   # shown in examples.jpg (several signs, one outside the 43 classes)
# Validated categorical palette (dataviz reference slots 1-4), fixed order = order of CORRUPTIONS.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gtsdb-root", type=Path, default=REPO_ROOT / "data/raw/FullIJCNN2013")
    p.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR)
    p.add_argument("--classifier", type=Path, default=DEFAULT_CLASSIFIER)
    p.add_argument("--det-conf", type=float, default=0.25)
    p.add_argument("--cnn-conf", type=float, default=0.60)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--device", default="auto")
    p.add_argument("--conditions", nargs="+", default=list(CORRUPTIONS), choices=list(CORRUPTIONS))
    p.add_argument("--limit", type=int, default=None, help="only the first N test images (quick check)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "results/robustness")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    try:
        raw = parse_gt_file(args.gtsdb_root / "gt.txt")
        pipe = RoadSignPipeline(args.detector, args.classifier, device=args.device,
                                det_conf=args.det_conf, cnn_conf=args.cnn_conf, imgsz=args.imgsz)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1

    gt_by_image = defaultdict(list)
    for a in dict.fromkeys(raw):  # drop the exact duplicate line of gt.txt
        gt_by_image[a.filename].append(a)
    images = sorted(p for p in args.gtsdb_root.glob("*.ppm") if int(p.stem) >= OFFICIAL_TEST_START)
    if args.limit:
        images = images[: args.limit]
    runs = [("clean", 0)] + [(c, lv) for c in args.conditions for lv in LEVELS]
    logger.info("%d images x %d versions | device %s | detector %s",
                len(images), len(runs), pipe.device, args.detector.name)

    totals: dict[tuple[str, int], Counter] = {run: Counter() for run in runs}
    times = []
    for k, path in enumerate(images, start=1):
        img = cv2.imread(str(path))
        if img is None:
            logger.error("Cannot read %s", path)
            return 1
        gts = gt_by_image.get(path.name, [])
        for name, level in runs:
            # the seed depends on the image, so the noise differs between images but not between runs
            degraded = img if level == 0 else apply(img, name, level, seed=args.seed * 1000 + k)
            res = pipe.predict(degraded)
            times.append(res.timings_ms["total"])
            totals[(name, level)] += score_image(res.detections, gts, args.iou)
        if k % 50 == 0:
            logger.info("  %d/%d images", k, len(images))

    clean = summarize(totals[("clean", 0)])
    metrics = {
        "description": "Complete pipeline on SIMULATED degradations of the official GTSDB test images.",
        "warning": "Simulations (darkening, blur, haze, noise), not real night or backlit photos.",
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "detector": args.detector.name, "classifier": args.classifier.name, "device": pipe.device,
        "images": len(images), "det_conf": args.det_conf, "cnn_conf": args.cnn_conf, "imgsz": args.imgsz,
        "iou": args.iou, "seed": args.seed,
        "clean": clean,
        "conditions": {name: {**describe(name),
                              "results": {str(lv): summarize(totals[(name, lv)]) for lv in LEVELS}}
                       for name in args.conditions},
        "median_ms_per_image": round(float(np.median(times)), 1) if times else None,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    save_chart(metrics, args.out / "end_to_end_recall.png")
    save_examples(pipe, args, args.out / "examples.jpg")

    def pc(v):
        return "n/a" if v is None else f"{v:.1%}"

    print("\n=== Robustness (SIMULATED conditions) on GTSDB official test images ===")
    print(f"Images {len(images)} | signs {clean['signs']} | device {pipe.device}")
    print(f"{'condition':<38} {'level':>5} {'found+named':>12} {'wrong':>6} {'fallback':>9} {'false al.':>10}")
    print(f"{'Images d origine':<38} {'-':>5} {pc(clean['end_to_end_recall']):>12} {clean['wrong']:>6} "
          f"{clean['fallback']:>9} {clean['false_alarms']:>10}")
    for name, cond in metrics["conditions"].items():
        for lv, r in cond["results"].items():
            print(f"{cond['label']:<38} {lv:>5} {pc(r['end_to_end_recall']):>12} {r['wrong']:>6} "
                  f"{r['fallback']:>9} {r['false_alarms']:>10}")
    print(f"Saved: {args.out}")
    return 0


def save_chart(metrics: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    clean = metrics["clean"]["end_to_end_recall"] or 0.0
    for color, (name, cond) in zip(SERIES_COLORS, metrics["conditions"].items()):
        ys = [clean] + [cond["results"][str(lv)]["end_to_end_recall"] or 0.0 for lv in LEVELS]
        ax.plot([0, *LEVELS], [y * 100 for y in ys], color=color, linewidth=2, marker="o", markersize=6,
                label=cond["label"])
        ax.annotate(f"{ys[-1]:.0%}", (LEVELS[-1], ys[-1] * 100), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color="#333333")
    ax.set_xticks([0, *LEVELS], ["origine", "léger", "moyen", "fort"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Panneaux trouvés et bien nommés (%)")
    ax.set_title(f"Robustesse du système complet (conditions simulées, {metrics['images']} images de test)",
                 fontsize=11)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_examples(pipe, args, path: Path) -> None:
    """One test image under each condition (level 2), annotated: for slides and the report."""
    src = args.gtsdb_root / EXAMPLE_IMAGE
    img = cv2.imread(str(src))
    if img is None:
        return
    ascii_titles = {"night": "Nuit", "motion_blur": "Flou de bouge", "haze": "Voile clair", "noise": "Bruit"}
    panels = [("Origine", img)] + [(ascii_titles.get(n, n) + " (niveau moyen)", apply(img, n, 2, seed=args.seed))
                                   for n in args.conditions]
    tiles = []
    for title, im in panels:
        tile = cv2.resize(pipe.annotate(im, pipe.predict(im)), (680, 400), interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile, (0, 0), (680, 30), (30, 30, 30), -1)
        cv2.putText(tile, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    while len(tiles) % 2:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    sys.exit(main())
