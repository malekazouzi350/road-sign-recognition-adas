"""Reader for the German Traffic Sign Detection Benchmark (GTSDB).

The annotation file ``gt.txt`` has one line per sign, fields separated by ``;``::

    filename;x_left;y_top;x_right;y_bottom;class_id
    00000.ppm;774;411;815;446;11

Images that contain no sign do not appear in ``gt.txt``, so the list of images
must come from the folder itself, not from the annotation file.

Class IDs 0-42 are shared with GTSRB. The four super-categories are the ones
used by the official IJCNN 2013 competition (prohibitory, danger, mandatory)
plus "other" for the remaining classes.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

NUM_CLASSES = 43

# Official GTSRB/GTSDB class names, indexed by class ID.
CLASS_NAMES: tuple[str, ...] = (
    "Speed limit (20km/h)",
    "Speed limit (30km/h)",
    "Speed limit (50km/h)",
    "Speed limit (60km/h)",
    "Speed limit (70km/h)",
    "Speed limit (80km/h)",
    "End of speed limit (80km/h)",
    "Speed limit (100km/h)",
    "Speed limit (120km/h)",
    "No passing",
    "No passing for vehicles over 3.5 t",
    "Right-of-way at the next intersection",
    "Priority road",
    "Yield",
    "Stop",
    "No vehicles",
    "Vehicles over 3.5 t prohibited",
    "No entry",
    "General caution",
    "Dangerous curve to the left",
    "Dangerous curve to the right",
    "Double curve",
    "Bumpy road",
    "Slippery road",
    "Road narrows on the right",
    "Road work",
    "Traffic signals",
    "Pedestrians",
    "Children crossing",
    "Bicycles crossing",
    "Beware of ice/snow",
    "Wild animals crossing",
    "End of all speed and passing limits",
    "Turn right ahead",
    "Turn left ahead",
    "Ahead only",
    "Go straight or right",
    "Go straight or left",
    "Keep right",
    "Keep left",
    "Roundabout mandatory",
    "End of no passing",
    "End of no passing by vehicles over 3.5 t",
)

SUPER_CATEGORIES: tuple[str, ...] = ("prohibitory", "danger", "mandatory", "other")

_PROHIBITORY = {0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16}
_DANGER = {11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}
_MANDATORY = {33, 34, 35, 36, 37, 38, 39, 40}

STOP_CLASS_ID = 14


def super_category_id(class_id: int) -> int:
    """Map a GTSDB class ID (0-42) to its super-category index in SUPER_CATEGORIES."""
    if not 0 <= class_id < NUM_CLASSES:
        raise ValueError(f"class_id must be in [0, {NUM_CLASSES - 1}], got {class_id}")
    if class_id in _PROHIBITORY:
        return 0
    if class_id in _DANGER:
        return 1
    if class_id in _MANDATORY:
        return 2
    return 3


@dataclass(frozen=True)
class SignAnnotation:
    """One annotated traffic sign. Coordinates are pixels, inclusive, as in gt.txt."""

    filename: str
    x1: int
    y1: int
    x2: int
    y2: int
    class_id: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1 + 1

    @property
    def height(self) -> int:
        return self.y2 - self.y1 + 1

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (float(self.x1), float(self.y1), float(self.x2), float(self.y2))

    @property
    def class_name(self) -> str:
        return CLASS_NAMES[self.class_id]

    @property
    def super_category(self) -> str:
        return SUPER_CATEGORIES[super_category_id(self.class_id)]


def parse_gt_file(gt_path: Path) -> list[SignAnnotation]:
    """Parse a GTSDB ``gt.txt`` file.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if a line is malformed (with its line number).
    """
    gt_path = Path(gt_path)
    if not gt_path.is_file():
        raise FileNotFoundError(f"GTSDB annotation file not found: {gt_path}")

    annotations: list[SignAnnotation] = []
    with gt_path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) != 6:
                raise ValueError(f"{gt_path}:{line_no}: expected 6 fields, got {len(parts)}: {line!r}")
            filename = parts[0]
            try:
                x1, y1, x2, y2, class_id = (int(p) for p in parts[1:])
            except ValueError as exc:
                raise ValueError(f"{gt_path}:{line_no}: non-integer field in {line!r}") from exc
            if x2 < x1 or y2 < y1:
                raise ValueError(f"{gt_path}:{line_no}: invalid box (right<left or bottom<top): {line!r}")
            if not 0 <= class_id < NUM_CLASSES:
                raise ValueError(f"{gt_path}:{line_no}: class_id {class_id} out of range")
            annotations.append(SignAnnotation(filename, x1, y1, x2, y2, class_id))
    return annotations


@dataclass
class GTSDBDataset:
    """All GTSDB images in a folder with their annotations (possibly none)."""

    root: Path
    image_paths: list[Path]
    annotations: dict[str, list[SignAnnotation]]  # filename -> signs (missing key = no sign)

    @classmethod
    def load(cls, root: Path) -> GTSDBDataset:
        """Load from an extracted FullIJCNN2013 folder (containing *.ppm and gt.txt)."""
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(
                f"GTSDB folder not found: {root}. Download and extract it first (see data/README.md)."
            )
        image_paths = sorted(root.glob("*.ppm"))
        if not image_paths:
            raise FileNotFoundError(f"No .ppm images found directly in {root}.")

        by_file: dict[str, list[SignAnnotation]] = defaultdict(list)
        for ann in parse_gt_file(root / "gt.txt"):
            by_file[ann.filename].append(ann)

        known = {p.name for p in image_paths}
        orphans = sorted(set(by_file) - known)
        if orphans:
            logger.warning("%d annotated file(s) have no image, e.g. %s", len(orphans), orphans[:3])

        return cls(root=root, image_paths=image_paths, annotations=dict(by_file))

    def signs_in(self, image_path: Path) -> list[SignAnnotation]:
        return self.annotations.get(Path(image_path).name, [])

    @property
    def num_signs(self) -> int:
        return sum(len(v) for v in self.annotations.values())
