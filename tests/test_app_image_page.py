"""Headless test of the Streamlit 'Détection sur image' page (no browser, no real weights).

The real models are replaced by a tiny fake pipeline, so this checks the PAGE (upload/example
flow, numbers, table, history, error messages), not the models: those are evaluated by
scripts/evaluate_pipeline.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")

from roadsigns.pipeline import PipelineResult, RoadSignPipeline, SignDetection  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
APP_DIR = REPO / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from components import detection as D  # noqa: E402

EXAMPLE_CHOICE = "Exemple GTSDB (images de test)"


class FakePipeline:
    device = "cpu"
    det_conf, cnn_conf, imgsz = 0.25, 0.6, 1024

    def __init__(self, detections):
        self.detections = detections

    def predict(self, image_bgr):
        h, w = image_bgr.shape[:2]
        return PipelineResult(detections=list(self.detections), image_size=(w, h),
                              timings_ms={"detector": 12.0, "classifier": 3.0, "total": 15.0})

    annotate = staticmethod(RoadSignPipeline.annotate)


def sign(box, recognised=True):
    return SignDetection(box=box, det_conf=0.9, super_category="other" if recognised else "prohibitory",
                         class_id=14 if recognised else None, class_name="Stop" if recognised else None,
                         cnn_conf=0.99 if recognised else 0.35, cnn_class_id=14, recognised=recognised,
                         agrees_with_detector=True)


@pytest.fixture
def example(tmp_path, monkeypatch):
    img = tmp_path / "00700.ppm"
    cv2.imwrite(str(img), np.full((200, 340, 3), 120, dtype=np.uint8))
    monkeypatch.setattr(D, "missing_weights", lambda: [])
    monkeypatch.setattr(D, "example_images", lambda: [img])
    return img


def run_page(monkeypatch, detections):
    monkeypatch.setattr(D, "load_pipeline", lambda device: FakePipeline(detections))
    at = st_testing.AppTest.from_file(str(APP_DIR / "app.py"), default_timeout=60)
    at.run()
    at.switch_page("pages/image_detection.py").run()
    at.radio[0].set_value(EXAMPLE_CHOICE).run()
    assert not at.exception, at.exception
    return at


def test_page_shows_signs_table_and_history(example, monkeypatch):
    at = run_page(monkeypatch, [sign((20, 30, 60, 70)), sign((300, 20, 338, 60), recognised=False)])
    html = " ".join(str(h.proto.body) for h in at.get("html"))
    assert "Panneaux détectés" in html and "Non reconnus (?)" in html
    assert any("Stop" in m.value for m in at.markdown)
    assert any("prohibitory (?)" in m.value for m in at.markdown)
    assert len(at.session_state["history"]) == 1
    assert at.session_state["history"][0]["reconnus"] == 1


def test_same_image_is_not_analysed_twice(example, monkeypatch):
    at = run_page(monkeypatch, [sign((20, 30, 60, 70))])
    at.run()
    assert len(at.session_state["history"]) == 1


def test_no_sign_gives_a_helpful_message(example, monkeypatch):
    at = run_page(monkeypatch, [])
    assert any("Aucun panneau détecté" in i.value for i in at.info)


def test_missing_weights_stop_the_page(monkeypatch):
    monkeypatch.setattr(D, "missing_weights", lambda: ["models/weights/classifier_best.pt"])
    at = st_testing.AppTest.from_file(str(APP_DIR / "app.py"), default_timeout=60)
    at.run()
    at.switch_page("pages/image_detection.py").run()
    assert not at.exception
    assert any("classifier_best.pt" in e.value for e in at.error)


def test_decode_image_rejects_non_images():
    assert D.decode_image(b"not an image") is None
    ok, buf = cv2.imencode(".png", np.zeros((10, 10, 3), dtype=np.uint8))
    assert D.decode_image(buf.tobytes()).shape == (10, 10, 3)


def test_upload_and_webcam_photo_use_the_same_analysis(monkeypatch):
    monkeypatch.setattr(D, "missing_weights", lambda: [])
    monkeypatch.setattr(D, "example_images", lambda: [])
    monkeypatch.setattr(D, "load_pipeline", lambda device: FakePipeline([sign((20, 30, 60, 70))]))
    ok, buf = cv2.imencode(".jpg", np.full((200, 340, 3), 120, dtype=np.uint8))
    at = st_testing.AppTest.from_file(str(APP_DIR / "app.py"), default_timeout=60)
    at.run()
    at.switch_page("pages/image_detection.py").run()
    at.file_uploader[0].upload("road.jpg", buf.tobytes(), "image/jpeg").run()
    assert not at.exception
    at.file_uploader[0].upload("bad.jpg", b"not an image", "image/jpeg").run()
    assert any("n'est pas une image lisible" in e.value for e in at.error)

    at.switch_page("pages/webcam_detection.py").run()
    at.camera_input[0].upload("photo.jpg", buf.tobytes(), "image/jpeg").run()
    assert not at.exception, at.exception
    assert [h["type"] for h in at.session_state["history"]] == ["image", "webcam"]


def test_webcam_closeup_mode_uses_the_cnn_only(monkeypatch):
    class CloseupPipeline(FakePipeline):
        def classify_closeup(self, image_bgr, top_k=3):
            return [(14, "Stop", 0.93), (17, "No entry", 0.04), (13, "Yield", 0.01)]

    monkeypatch.setattr(D, "missing_weights", lambda: [])
    monkeypatch.setattr(D, "load_pipeline", lambda device: CloseupPipeline([]))
    ok, buf = cv2.imencode(".jpg", np.full((200, 340, 3), 120, dtype=np.uint8))
    at = st_testing.AppTest.from_file(str(APP_DIR / "app.py"), default_timeout=60)
    at.run()
    at.switch_page("pages/webcam_detection.py").run()
    at.radio[0].set_value("Panneau en gros plan (CNN seul)").run()
    at.camera_input[0].upload("photo.jpg", buf.tobytes(), "image/jpeg").run()
    assert not at.exception, at.exception
    assert any("Stop" in s.value for s in at.success)
    assert at.session_state["history"][-1]["type"] == "webcam (gros plan)"
