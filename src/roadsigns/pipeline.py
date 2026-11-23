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

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DETECTOR = REPO_ROOT / "models" / "weights" / "detector_v1_quick.pt"
DEFAULT_CLASSIFIER = REPO_ROOT / "models" / "weights" / "classifier_best.pt"

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
        imgsz: int = 1024,
    ) -> None:
        detector_weights, classifier_weights = Path(detector_weights), Path(classifier_weights)
        for p, what in ((detector_weights, "detector"), (classifier_weights, "classifier")):
            if not p.is_file():
                raise FileNotFoundError(f"{what} weights not found: {p} (get them from the team Drive, "
                                        "put them in models/weights/)")
        from ultralytics import YOLO  # heavy import, only when a pipeline is built

        self.device = resolve_device(device)
        self.det_conf, self.cnn_conf, self.imgsz = det_conf, cnn_conf, imgsz
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
        crops, kept = [], []
        for i, (x1, y1, x2, y2) in enumerate(boxes):
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
        """Draw boxes and readable labels (dark background behind the text)."""
        out = image_bgr.copy()
        scale = max(0.5, min(1.0, out.shape[1] / 1600))
        for d in result.detections:
            color = SUPER_COLORS_BGR.get(d.super_category, (200, 200, 200))
            x1, y1, x2, y2 = d.box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, max(2, int(3 * scale)))
            text = d.label
            (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, 2)
            ty = y1 - 6 if y1 - th - 10 > 0 else y2 + th + 10
            cv2.rectangle(out, (x1, ty - th - 6), (x1 + tw + 8, ty + base), (30, 30, 30), -1)
            cv2.putText(out, text, (x1 + 4, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, color, 2, cv2.LINE_AA)
        return out
