"""Reader for the German Traffic Sign Recognition Benchmark (GTSRB) training set.

Layout after extraction (see data/README.md)::

    GTSRB/Final_Training/Images/00000/GT-00000.csv
    GTSRB/Final_Training/Images/00000/00000_00000.ppm   <- track 00000, frame 00000
    ...

Each CSV row: ``Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId``.
The image contains a ~10 % margin around the sign; the ROI gives the sign itself.

Images are grouped in *tracks*: about 30 consecutive frames of the SAME physical
sign. Splitting image by image would put near-identical frames in both training
and validation and inflate the score, so we split whole tracks.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from roadsigns.datasets.gtsdb import NUM_CLASSES


@dataclass(frozen=True)
class GTSRBSample:
    path: Path
    class_id: int
    track: str  # "<class>_<track>", unique over the whole dataset
    roi: tuple[int, int, int, int]  # x1, y1, x2, y2 (inclusive)


def load_gtsrb_training(images_root: Path) -> list[GTSRBSample]:
    """Read every GT-XXXXX.csv under ``GTSRB/Final_Training/Images``."""
    images_root = Path(images_root)
    if not images_root.is_dir():
        raise FileNotFoundError(
            f"GTSRB folder not found: {images_root}. Download and extract it first (see data/README.md)."
        )
    samples: list[GTSRBSample] = []
    for csv_path in sorted(images_root.glob("*/GT-*.csv")):
        with csv_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter=";"):
                class_id = int(row["ClassId"])
                if not 0 <= class_id < NUM_CLASSES:
                    raise ValueError(f"{csv_path}: class {class_id} out of range")
                track = row["Filename"].split("_")[0]
                roi = (int(row["Roi.X1"]), int(row["Roi.Y1"]), int(row["Roi.X2"]), int(row["Roi.Y2"]))
                samples.append(GTSRBSample(csv_path.parent / row["Filename"], class_id,
                                           f"{class_id:05d}_{track}", roi))
    if not samples:
        raise FileNotFoundError(f"No GT-*.csv annotation files found under {images_root}")
    return samples


def split_by_track(
    samples: list[GTSRBSample],
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 0,
) -> dict[str, list[GTSRBSample]]:
    """Split into train/val/test by whole tracks, class by class (stratified).

    Every class with at least 3 tracks gets at least one track in val and in test.
    The same seed always gives the same split.
    """
    if not 0 < val_fraction < 1 or not 0 < test_fraction < 1 or val_fraction + test_fraction >= 1:
        raise ValueError("fractions must be in (0, 1) and sum to less than 1")

    tracks_by_class: dict[int, list[str]] = defaultdict(list)
    by_track: dict[str, list[GTSRBSample]] = defaultdict(list)
    for s in samples:
        if s.track not in by_track:
            tracks_by_class[s.class_id].append(s.track)
        by_track[s.track].append(s)

    rng = random.Random(seed)
    splits: dict[str, list[GTSRBSample]] = {"train": [], "val": [], "test": []}
    for class_id in sorted(tracks_by_class):
        tracks = sorted(tracks_by_class[class_id])
        rng.shuffle(tracks)
        n = len(tracks)
        n_val = max(1, round(n * val_fraction)) if n >= 3 else 0
        n_test = max(1, round(n * test_fraction)) if n >= 3 else 0
        for i, track in enumerate(tracks):
            part = "val" if i < n_val else "test" if i < n_val + n_test else "train"
            splits[part].extend(by_track[track])
    return splits
