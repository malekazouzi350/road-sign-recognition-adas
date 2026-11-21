"""Convert GTSDB annotations to the YOLO format used by Ultralytics.

YOLO label format: one ``.txt`` per image, one line per object::

    class_id x_center y_center width height      (all normalised to [0, 1])

GTSDB boxes are inclusive pixel coordinates (x1, y1, x2, y2), so the box width
is ``x2 - x1 + 1``.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from roadsigns.datasets.gtsdb import CLASS_NAMES, SUPER_CATEGORIES, SignAnnotation, super_category_id
from roadsigns.evaluation.matching import iou

CLASS_MODES = ("super", "full")


def class_names(mode: str) -> tuple[str, ...]:
    """Names of the YOLO classes for a mode: 4 super-categories or the 43 GTSDB classes."""
    if mode == "super":
        return SUPER_CATEGORIES
    if mode == "full":
        return CLASS_NAMES
    raise ValueError(f"mode must be one of {CLASS_MODES}, got {mode!r}")


def yolo_class_id(ann: SignAnnotation, mode: str) -> int:
    return super_category_id(ann.class_id) if mode == "super" else ann.class_id


def to_yolo_line(ann: SignAnnotation, img_w: int, img_h: int, mode: str) -> str:
    """One YOLO label line for an annotation, coordinates clipped to the image."""
    x1, y1 = max(0, ann.x1), max(0, ann.y1)
    x2, y2 = min(img_w - 1, ann.x2), min(img_h - 1, ann.y2)
    w, h = x2 - x1 + 1, y2 - y1 + 1
    if w <= 0 or h <= 0:
        raise ValueError(f"box outside the image: {ann}")
    cx, cy = x1 + w / 2, y1 + h / 2
    return f"{yolo_class_id(ann, mode)} {cx / img_w:.6f} {cy / img_h:.6f} {w / img_w:.6f} {h / img_h:.6f}"


@dataclass
class DedupReport:
    exact_duplicates: int = 0
    near_duplicates: int = 0  # same class, IoU >= threshold
    overlapping_different_class: int = 0  # kept, only reported


def deduplicate(annotations: Sequence[SignAnnotation], iou_threshold: float = 0.9) -> tuple[list[SignAnnotation], DedupReport]:
    """Remove annotations that describe the same sign twice within one image.

    * exact duplicates (identical line) are removed;
    * same class + IoU >= ``iou_threshold`` are removed (the first one is kept);
    * strongly overlapping boxes with DIFFERENT classes are kept but counted, because
      deciding which label is right needs a human look.
    """
    report = DedupReport()
    kept: list[SignAnnotation] = []
    seen: set[SignAnnotation] = set()
    for ann in annotations:
        if ann in seen:
            report.exact_duplicates += 1
            continue
        seen.add(ann)
        same_image = [k for k in kept if k.filename == ann.filename]
        if any(k.class_id == ann.class_id and iou(k.box, ann.box) >= iou_threshold for k in same_image):
            report.near_duplicates += 1
            continue
        if any(k.class_id != ann.class_id and iou(k.box, ann.box) >= iou_threshold for k in same_image):
            report.overlapping_different_class += 1
        kept.append(ann)
    return kept, report


def split_images(
    image_names: Iterable[str],
    test_start: int = 600,
    val_fraction: float = 0.15,
    seed: int = 0,
) -> dict[str, list[str]]:
    """Official IJCNN 2013 split for test (images >= ``test_start``); the official training
    images are split again into train / val with a fixed seed."""
    names = sorted(image_names)
    test = [n for n in names if int(n.split(".")[0]) >= test_start]
    train_pool = [n for n in names if int(n.split(".")[0]) < test_start]
    rng = random.Random(seed)
    shuffled = train_pool[:]
    rng.shuffle(shuffled)
    n_val = round(len(shuffled) * val_fraction)
    return {"train": sorted(shuffled[n_val:]), "val": sorted(shuffled[:n_val]), "test": test}
