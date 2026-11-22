"""Match predicted boxes to ground-truth boxes and count TP / FP / FN.

Boxes are (x1, y1, x2, y2) in pixels. A prediction is a true positive if its
IoU with a not-yet-matched ground-truth box is >= the threshold. Matching is
greedy by descending prediction confidence (the standard rule used by PASCAL VOC
and COCO-style evaluation), so each ground-truth box is matched at most once.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

Box = tuple[float, float, float, float]


def iou(a: Box, b: Box) -> float:
    """Intersection over union of two boxes. Returns 0.0 if they do not overlap."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


@dataclass
class MatchResult:
    """Counts for one image (or summed over many)."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    matched_gt: tuple[int, ...] = ()  # indices of matched ground-truth boxes (single image only)

    def __add__(self, other: MatchResult) -> MatchResult:
        return MatchResult(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)

    @property
    def precision(self) -> float | None:
        """None when there are no predictions (precision is undefined, not 0)."""
        n = self.tp + self.fp
        return self.tp / n if n else None

    @property
    def recall(self) -> float | None:
        """None when there is no ground truth."""
        n = self.tp + self.fn
        return self.tp / n if n else None


def match_boxes(
    predictions: Sequence[tuple[Box, float]],
    ground_truth: Sequence[Box],
    iou_threshold: float = 0.5,
) -> MatchResult:
    """Greedy one-to-one matching of (box, confidence) predictions to ground-truth boxes."""
    order = sorted(range(len(predictions)), key=lambda i: predictions[i][1], reverse=True)
    used: set[int] = set()
    tp = fp = 0
    for i in order:
        pred_box = predictions[i][0]
        best_j, best_iou = -1, iou_threshold
        for j, gt_box in enumerate(ground_truth):
            if j in used:
                continue
            value = iou(pred_box, gt_box)
            if value >= best_iou:
                best_j, best_iou = j, value
        if best_j >= 0:
            used.add(best_j)
            tp += 1
        else:
            fp += 1
    return MatchResult(tp=tp, fp=fp, fn=len(ground_truth) - len(used), matched_gt=tuple(sorted(used)))


def match_pairs(
    predictions: Sequence[tuple[Box, float]],
    ground_truth: Sequence[Box],
    iou_threshold: float = 0.5,
) -> list[tuple[int, int]]:
    """Same greedy rule as match_boxes, but return the (prediction index, ground-truth index)
    pairs, so the caller can compare the classes of matched boxes."""
    order = sorted(range(len(predictions)), key=lambda i: predictions[i][1], reverse=True)
    used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for i in order:
        best_j, best_iou = -1, iou_threshold
        for j, gt_box in enumerate(ground_truth):
            if j in used:
                continue
            value = iou(predictions[i][0], gt_box)
            if value >= best_iou:
                best_j, best_iou = j, value
        if best_j >= 0:
            used.add(best_j)
            pairs.append((i, best_j))
    return pairs
