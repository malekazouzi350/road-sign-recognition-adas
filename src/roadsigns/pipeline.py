"""The complete road-sign pipeline: detector -> crop -> CNN -> named sign.

This is the ONLY entry point the application (Streamlit image/video/webcam pages)
and the evaluation scripts use, so every part of the project runs the same code.

    from roadsigns.pipeline import RoadSignPipeline
    pipe = RoadSignPipeline.from_default_weights()
    result = pipe.predict(image_bgr)          # OpenCV BGR image
    annotated = pipe.annotate(image_bgr, result)

Stage 1  YOLO26n detector (trained by the team on GTSDB): where are the signs?
         -> boxes + one of 4 super-categories + detector confidence.
Stage 2  TrafficSignCNN (trained by the team on GTSRB): which of the 43 signs?
         -> class + CNN confidence, on a 48x48 crop of each box.
Safety   if the CNN confidence is below ``cnn_conf``, the sign is NOT named: only the
         detector's super-category is shown. Measured on GTSDB: mean confidence 0.98
         when the CNN is right, 0.41 when it is wrong.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

from roadsigns.classification.data import crop_and_resize, preprocess_crop
from roadsigns.classification.model import TrafficSignCNN
from roadsigns.datasets.gtsdb import CLASS_NAMES, NUM_CLASSES, SUPER_CATEGORIES, super_category_id
from roadsigns.evaluation.matching import iou

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DETECTOR = REPO_ROOT / "models" / "weights" / "detector_v2_1280.pt"  # v2: trained at 1280 px
DEFAULT_CLASSIFIER = REPO_ROOT / "models" / "weights" / "classifier_blur.pt"  # v2: trained with motion blur

# BGR colours per super-category (same palette as the slides and the app).
SUPER_COLORS_BGR = {
    "prohibitory": (74, 81, 196),   # coral red
    "danger": (59, 169, 242),       # amber
    "mandatory": (219, 111, 47),    # blue
    "other": (124, 122, 11),        # teal
}


@dataclass
class SignDetection:
    """One sign found in an image."""

    box: tuple[int, int, int, int]          # x1, y1, x2, y2 in pixels
    det_conf: float                         # detector confidence
    super_category: str                     # from the detector
    class_id: int | None                    # CNN class (None if not recognised)
    class_name: str | None
    cnn_conf: float                         # CNN confidence for its best class
    cnn_class_id: int                       # CNN best class, even if below threshold
    recognised: bool                        # cnn_conf >= threshold
    agrees_with_detector: bool              # CNN class lies in the detector's super-category

    @property
    def label(self) -> str:
        if self.recognised:
            return f"{self.class_name} ({self.cnn_conf:.2f})"
        return f"{self.super_category} (?)"


@dataclass
class PipelineResult:
    detections: list[SignDetection] = field(default_factory=list)
    image_size: tuple[int, int] = (0, 0)    # width, height
    timings_ms: dict[str, float] = field(default_factory=dict)

    def to_rows(self) -> list[dict]:
        """Flat rows for a table (Streamlit dataframe, CSV)."""
        rows = []
        for i, d in enumerate(self.detections, start=1):
            rows.append({
                "#": i,
                "panneau": d.class_name if d.recognised else "non reconnu",
                "classe": d.class_id if d.recognised else None,
                "super-catégorie": d.super_category,
                "confiance détecteur": round(d.det_conf, 3),
                "confiance CNN": round(d.cnn_conf, 3),
                "x1": d.box[0], "y1": d.box[1], "x2": d.box[2], "y2": d.box[3],
            })
        return rows

    def to_dict(self) -> dict:
        return {"image_size": self.image_size, "timings_ms": self.timings_ms,
                "detections": [asdict(d) for d in self.detections]}


def resolve_device(device: str) -> str:
    """'auto' -> 'cuda' if an NVIDIA GPU is usable, else 'cpu'."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but no usable NVIDIA GPU was found")
    return device


class RoadSignPipeline:
    def __init__(
        self,
        detector_weights: Path,
        classifier_weights: Path,
        device: str = "auto",
        det_conf: float = 0.25,
        cnn_conf: float = 0.60,
        imgsz: int = 1280,
        merge_iou: float | None = 0.6,
    ) -> None:
        detector_weights, classifier_weights = Path(detector_weights), Path(classifier_weights)
        for p, what in ((detector_weights, "detector"), (classifier_weights, "classifier")):
            if not p.is_file():
                raise FileNotFoundError(f"{what} weights not found: {p} (get them from the team Drive, "
                                        "put them in models/weights/)")
        from ultralytics import YOLO  # heavy import, only when a pipeline is built

        self.device = resolve_device(device)
        self.det_conf, self.cnn_conf, self.imgsz = det_conf, cnn_conf, imgsz
        self.merge_iou = merge_iou  # None/0 = keep every detector box (behaviour before 2026-11-23)
        self.detector = YOLO(str(detector_weights))
        self.detector_names = {int(k): v for k, v in self.detector.names.items()}
        unknown = set(self.detector_names.values()) - set(SUPER_CATEGORIES)
        if unknown:
            raise ValueError(f"detector classes {sorted(unknown)} are not super-categories {SUPER_CATEGORIES}")

        ckpt = torch.load(classifier_weights, map_location=self.device)
        self.classifier = TrafficSignCNN(ckpt.get("num_classes", NUM_CLASSES)).to(self.device).eval()
        self.classifier.load_state_dict(ckpt["model_state"])
        self.detector_weights, self.classifier_weights = detector_weights, classifier_weights

    @classmethod
    def from_default_weights(cls, **kwargs) -> RoadSignPipeline:
        return cls(DEFAULT_DETECTOR, DEFAULT_CLASSIFIER, **kwargs)

    @torch.no_grad()
    def _classify(self, crops: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        if not crops:
            return np.empty(0, dtype=int), np.empty(0, dtype=float)
        x = torch.stack([preprocess_crop(c) for c in crops]).to(self.device)
        prob = torch.softmax(self.classifier(x), dim=1)
        conf, cls = prob.max(1)
        return cls.cpu().numpy(), conf.cpu().numpy()

    def classify_closeup(self, image_bgr: np.ndarray, top_k: int = 3) -> list[tuple[int, str, float]]:
        """CNN only, for a photo where ONE sign fills the centre (e.g. webcam close-up).

        The detector is skipped: it was trained on small signs in road scenes and misses
        signs that fill the image. The central square of the photo is given to the CNN.
        Returns the ``top_k`` (class_id, class_name, probability), best first. The CNN always
        answers among the 43 classes, so an unknown sign simply gets a low probability.
        """
        if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("expected a colour image (H x W x 3, BGR)")
        box = central_square(image_bgr.shape[1], image_bgr.shape[0])
        x = preprocess_crop(crop_and_resize(image_bgr, box)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prob = torch.softmax(self.classifier(x), dim=1)[0].cpu().numpy()
        best = prob.argsort()[::-1][:top_k]
        return [(int(c), CLASS_NAMES[int(c)], float(prob[c])) for c in best]

    def predict(self, image_bgr: np.ndarray) -> PipelineResult:
        if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("expected a colour image (H x W x 3, BGR)")
        h, w = image_bgr.shape[:2]
        t0 = time.perf_counter()
        yolo_device = "cpu" if self.device == "cpu" else 0
        det = self.detector.predict(image_bgr, imgsz=self.imgsz, conf=self.det_conf,
                                    device=yolo_device, verbose=False)[0]
        t1 = time.perf_counter()

        boxes = det.boxes.xyxy.cpu().numpy().round().astype(int)
        det_confs = det.boxes.conf.cpu().numpy()
        det_cls = det.boxes.cls.cpu().numpy().astype(int)
        # One sign can get two boxes of different families (e.g. "other" + "prohibitory"):
        # keep only the most confident one (class-agnostic suppression).
        order = merge_overlapping(boxes, det_confs, getattr(self, "merge_iou", None))
        crops, kept = [], []
        for i in order:
            x1, y1, x2, y2 = boxes[i]
            try:
                crops.append(crop_and_resize(image_bgr, (int(x1), int(y1), int(x2), int(y2))))
                kept.append(i)
            except ValueError:
                continue  # degenerate box (outside or zero-size): skip it
        cnn_cls, cnn_conf = self._classify(crops)
        t2 = time.perf_counter()

        result = PipelineResult(image_size=(w, h))
        for k, i in enumerate(kept):
            sup = self.detector_names[int(det_cls[i])]
            c, conf = int(cnn_cls[k]), float(cnn_conf[k])
            recognised = conf >= self.cnn_conf
            result.detections.append(SignDetection(
                box=tuple(int(v) for v in boxes[i]), det_conf=float(det_confs[i]), super_category=sup,
                class_id=c if recognised else None, class_name=CLASS_NAMES[c] if recognised else None,
                cnn_conf=conf, cnn_class_id=c, recognised=recognised,
                agrees_with_detector=SUPER_CATEGORIES[super_category_id(c)] == sup,
            ))
        result.detections.sort(key=lambda d: d.box[0])  # left to right
        result.timings_ms = {"detector": round((t1 - t0) * 1000, 1), "classifier": round((t2 - t1) * 1000, 1),
                             "total": round((t2 - t0) * 1000, 1)}
        return result

    @staticmethod
    def annotate(image_bgr: np.ndarray, result: PipelineResult) -> np.ndarray:
        """Draw boxes and readable labels (dark background behind the text).

        Labels always stay inside the image, never cover each other and, when there is
        room, do not cover the other boxes either (see ``place_label``).
        """
        out = image_bgr.copy()
        h, w = out.shape[:2]
        scale = max(0.5, min(1.0, w / 1600))
        font, font_scale = cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale
        # Boxes first, so that no box line is drawn over a label.
        for d in result.detections:
            color = SUPER_COLORS_BGR.get(d.super_category, (200, 200, 200))
            x1, y1, x2, y2 = d.box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, max(2, int(3 * scale)))
        placed: list[tuple[int, int, int, int]] = []
        all_boxes = [d.box for d in result.detections]
        for d in result.detections:
            color = SUPER_COLORS_BGR.get(d.super_category, (200, 200, 200))
            (tw, th), base = cv2.getTextSize(d.label, font, font_scale, 2)
            lw, lh = tw + 8, th + base + 8
            others = [b for b in all_boxes if b is not d.box]
            lx, ly = place_label(d.box, (lw, lh), (w, h), placed, others)
            placed.append((lx, ly, lx + lw, ly + lh))
            cv2.rectangle(out, (lx, ly), (lx + lw, ly + lh), (30, 30, 30), -1)
            cv2.putText(out, d.label, (lx + 4, ly + th + 4), font, font_scale, color, 2, cv2.LINE_AA)
        return out


def merge_overlapping(boxes, confidences, merge_iou: float | None = 0.6,
                      containment: float = 0.85) -> list[int]:
    """Indices of the boxes to keep, most confident first.

    A box is dropped when a MORE confident box covers the same sign, whatever their
    families: IoU above ``merge_iou``, or one box lying almost entirely inside the
    other (intersection / smaller area above ``containment``). ``merge_iou`` None or
    <= 0 keeps every box.
    """
    order = sorted(range(len(boxes)), key=lambda i: -float(confidences[i]))
    if not merge_iou or merge_iou <= 0:
        return order
    kept: list[int] = []
    for i in order:
        a = tuple(float(v) for v in boxes[i])
        duplicate = False
        for j in kept:
            b = tuple(float(v) for v in boxes[j])
            if iou(a, b) > merge_iou or _containment(a, b) > containment:
                duplicate = True
                break
        if not duplicate:
            kept.append(i)
    return kept


def _containment(a, b) -> float:
    """Intersection area / area of the smaller box (1.0 = one box is inside the other)."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return (iw * ih) / smaller if smaller > 0 else 0.0


def _overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def place_label(
    box: tuple[int, int, int, int],
    label_size: tuple[int, int],
    image_size: tuple[int, int],
    placed: list[tuple[int, int, int, int]],
    other_boxes: list[tuple[int, int, int, int]] = (),
) -> tuple[int, int]:
    """Top-left corner for a label of ``label_size`` (w, h) next to ``box``.

    Rules: the label is kept fully inside the image (clamped at the borders); it goes
    above the box if possible, else below; if that spot covers an already placed
    label (or, when possible, another sign's box), it is moved further up / down.
    """
    x1, y1, x2, y2 = box
    lw, lh = label_size
    w, h = image_size
    lx = min(max(0, x1), max(0, w - lw))

    def clamp_y(y: int) -> int:
        return min(max(0, y), max(0, h - lh))

    candidates = []
    for step in range(8):
        candidates.append(y1 - lh - 2 - step * (lh + 2))   # above, then further up
        candidates.append(y2 + 2 + step * (lh + 2))        # below, then further down
    # First try a spot that hides neither labels nor boxes, then one that hides no label.
    for obstacles in (list(placed) + list(other_boxes), list(placed)):
        for y in candidates:
            if y < 0 or y + lh > h:
                continue
            rect = (lx, y, lx + lw, y + lh)
            if not any(_overlaps(rect, o) for o in obstacles):
                return lx, y
    return lx, clamp_y(y1 - lh - 2)  # crowded image: accept an overlap, but stay inside


def central_square(width: int, height: int, fraction: float = 0.8) -> tuple[int, int, int, int]:
    """Box (x1, y1, x2, y2) of the central square covering ``fraction`` of the shorter side."""
    side = max(1, int(min(width, height) * fraction))
    x1, y1 = (width - side) // 2, (height - side) // 2
    return x1, y1, x1 + side, y1 + side
