"""Convert GTSDB to a YOLO dataset (images + labels + data.yaml), with checks.

What it does:
  1. reads data/raw/FullIJCNN2013 (900 .ppm images + gt.txt);
  2. removes duplicate annotations (exact, or same class with IoU >= 0.9);
  3. splits: official test = images 00600-00899; official train (00000-00599)
     split again into train / val (85 / 15, fixed seed);
  4. writes PNG images (lossless) and YOLO label files;
  5. re-reads the written label files and checks the counts;
  6. draws a few images FROM THE WRITTEN LABELS (round-trip check) into results/dataset/.

Output (git-ignored, rebuildable at any time):
  data/processed/gtsdb_yolo_<mode>/images/{train,val,test}/*.png
  data/processed/gtsdb_yolo_<mode>/labels/{train,val,test}/*.txt
  data/processed/gtsdb_yolo_<mode>/data.yaml
Committed report: results/dataset/<mode>/report.json and overlays/*.jpg

Usage:
    python scripts/prepare_gtsdb.py                  # 4 super-categories (our detector)
    python scripts/prepare_gtsdb.py --classes full   # 43 classes (comparison experiment)
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import yaml

from roadsigns.datasets.gtsdb import GTSDBDataset, parse_gt_file
from roadsigns.datasets.yolo_export import CLASS_MODES, class_names, deduplicate, split_images, to_yolo_line

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger("prepare_gtsdb")
COLORS = [(91, 73, 209), (73, 174, 237), (140, 121, 0), (125, 117, 108)]  # BGR


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gtsdb-root", type=Path, default=REPO_ROOT / "data/raw/FullIJCNN2013")
    p.add_argument("--classes", choices=CLASS_MODES, default="super")
    p.add_argument("--out", type=Path, default=None, help="default: data/processed/gtsdb_yolo_<classes>")
    p.add_argument("--report-dir", type=Path, default=None, help="default: results/dataset/<classes>")
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overlays", type=int, default=12)
    return p.parse_args()


def draw_from_label_file(image_path: Path, label_path: Path, names: tuple[str, ...]):
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    for line in label_path.read_text(encoding="utf-8").splitlines():
        c, cx, cy, bw, bh = line.split()
        c = int(c)
        cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
        p1 = (int(round(cx - bw / 2)), int(round(cy - bh / 2)))
        p2 = (int(round(cx + bw / 2)), int(round(cy + bh / 2)))
        color = COLORS[c % len(COLORS)] if len(names) == 4 else (0, 200, 0)
        cv2.rectangle(img, p1, p2, color, 3)
        label = names[c]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        y = max(th + 6, p1[1] - 4)
        cv2.rectangle(img, (p1[0], y - th - 6), (p1[0] + tw + 6, y + 2), (0, 0, 0), -1)
        cv2.putText(img, label, (p1[0] + 3, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return img


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    out = args.out or REPO_ROOT / f"data/processed/gtsdb_yolo_{args.classes}"
    report_dir = args.report_dir or REPO_ROOT / f"results/dataset/{args.classes}"
    names = class_names(args.classes)

    try:
        ds = GTSDBDataset.load(args.gtsdb_root)
        raw = parse_gt_file(args.gtsdb_root / "gt.txt")
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    kept, dedup = deduplicate(raw)
    logger.info("Annotations: %d read, %d exact duplicates and %d near-duplicates removed -> %d kept",
                len(raw), dedup.exact_duplicates, dedup.near_duplicates, len(kept))
    by_file: dict[str, list] = {}
    for ann in kept:
        by_file.setdefault(ann.filename, []).append(ann)

    splits = split_images([p.name for p in ds.image_paths], val_fraction=args.val_fraction, seed=args.seed)

    if out.exists():
        logger.info("Removing previous output %s", out)
        shutil.rmtree(out)
    stats: dict[str, Counter] = {}
    for split, files in splits.items():
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
        counter: Counter = Counter()
        for name in files:
            img = cv2.imread(str(args.gtsdb_root / name))
            if img is None:
                logger.error("Cannot read %s", name)
                return 1
            h, w = img.shape[:2]
            stem = Path(name).stem
            cv2.imwrite(str(out / "images" / split / f"{stem}.png"), img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            lines = [to_yolo_line(a, w, h, args.classes) for a in by_file.get(name, [])]
            # An empty label file = "this image contains no sign" (a useful negative example).
            (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""),
                                                                 encoding="utf-8")
            counter["images"] += 1
            counter["images_without_signs"] += int(not lines)
            for line in lines:
                counter[names[int(line.split()[0])]] += 1
        stats[split] = counter
        logger.info("%-5s: %3d images, %4d objects", split, counter["images"],
                    sum(v for k, v in counter.items() if k in names))

    # Round-trip check: re-read every written label file.
    written = sum(len(p.read_text(encoding="utf-8").splitlines()) for p in (out / "labels").rglob("*.txt"))
    if written != len(kept):
        logger.error("Check failed: %d label lines written, %d expected", written, len(kept))
        return 1
    logger.info("Check OK: %d label lines written = %d annotations kept", written, len(kept))

    data_yaml = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(names)),
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8")

    report_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = report_dir / "overlays"
    if overlay_dir.exists():
        shutil.rmtree(overlay_dir)
    overlay_dir.mkdir()
    busiest = sorted(by_file, key=lambda n: len(by_file[n]), reverse=True)
    for name in busiest[: args.overlays]:
        split = next(s for s, files in splits.items() if name in files)
        stem = Path(name).stem
        img = draw_from_label_file(out / "images" / split / f"{stem}.png", out / "labels" / split / f"{stem}.txt", names)
        cv2.imwrite(str(overlay_dir / f"{split}_{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])

    report = {
        "classes_mode": args.classes,
        "class_names": list(names),
        "annotations_read": len(raw),
        "exact_duplicates_removed": dedup.exact_duplicates,
        "near_duplicates_removed": dedup.near_duplicates,
        "overlapping_different_class_kept": dedup.overlapping_different_class,
        "annotations_kept": len(kept),
        "split_rule": f"test = official 00600-00899; train/val = official 00000-00599 split "
                      f"{1 - args.val_fraction:.0%}/{args.val_fraction:.0%}, seed {args.seed}",
        "splits": {s: dict(c) for s, c in stats.items()},
        "val_images": splits["val"],
    }
    (report_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n=== GTSDB -> YOLO ({args.classes}: {len(names)} classes) ===")
    print(f"Annotations: {len(raw)} read | {dedup.exact_duplicates} exact + {dedup.near_duplicates} near duplicates "
          f"removed | {len(kept)} kept | {dedup.overlapping_different_class} overlapping boxes with different classes")
    for s, c in stats.items():
        objects = {k: v for k, v in c.items() if k in names}
        print(f"  {s:<5}: {c['images']:3d} images ({c['images_without_signs']} without sign), "
              f"{sum(objects.values()):4d} objects" + (f" {objects}" if len(names) == 4 else ""))
    print(f"data.yaml : {out / 'data.yaml'}")
    print(f"Report    : {report_dir / 'report.json'}  (+ {len(list(overlay_dir.glob('*.jpg')))} overlays)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
