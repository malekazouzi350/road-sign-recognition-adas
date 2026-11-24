"""Count the end-to-end outcome of the pipeline on one image (shared by the evaluation scripts).

A predicted box and a ground-truth sign are matched one-to-one (IoU >= ``iou``, greedy, most
confident box first). Each matched sign is then:
  correct  : CNN confident (>= cnn_conf) and right   wrong : confident but wrong (dangerous)
  fallback : CNN not confident -> only the family is shown (safe)
Unmatched ground truth = missed; unmatched boxes = false alarms.
"""

from __future__ import annotations

from collections import Counter

from roadsigns.evaluation.matching import match_pairs

KEYS = ("signs", "boxes", "found", "correct", "wrong", "fallback", "missed", "false_alarms")


def score_image(detections, ground_truth, iou: float = 0.5) -> Counter:
    pairs = match_pairs([(d.box, d.det_conf) for d in detections], [g.box for g in ground_truth], iou)
    c = Counter({k: 0 for k in KEYS})
    c["signs"], c["boxes"], c["found"] = len(ground_truth), len(detections), len(pairs)
    for i, j in pairs:
        d, g = detections[i], ground_truth[j]
        if not d.recognised:
            c["fallback"] += 1
        elif d.class_id == g.class_id:
            c["correct"] += 1
        else:
            c["wrong"] += 1
    c["missed"] = len(ground_truth) - len(pairs)
    c["false_alarms"] = len(detections) - len(pairs)
    return c


def summarize(c: Counter) -> dict:
    """Counts + the headline ratios (None when undefined)."""
    def r(a, b):
        return round(a / b, 4) if b else None
    out = {k: int(c[k]) for k in KEYS}
    out["end_to_end_recall"] = r(c["correct"], c["signs"])
    out["detection_recall"] = r(c["found"], c["signs"])
    out["detection_precision"] = r(c["found"], c["boxes"])
    return out
