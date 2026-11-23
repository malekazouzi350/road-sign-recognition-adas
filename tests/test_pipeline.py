from types import SimpleNamespace

import numpy as np
import pytest

from roadsigns.evaluation.matching import match_pairs
from roadsigns.pipeline import PipelineResult, RoadSignPipeline, SignDetection, resolve_device


def det(recognised=True, cls=14, conf=0.95, sup="other", box=(10, 10, 40, 40)):
    return SignDetection(box=box, det_conf=0.9, super_category=sup, class_id=cls if recognised else None,
                         class_name="Stop" if recognised else None, cnn_conf=conf, cnn_class_id=cls,
                         recognised=recognised, agrees_with_detector=True)


def test_match_pairs_returns_prediction_and_gt_indices():
    preds = [((0, 0, 10, 10), 0.5), ((50, 50, 60, 60), 0.9)]
    gts = [(50, 50, 60, 60), (0, 0, 10, 10), (100, 100, 110, 110)]
    assert sorted(match_pairs(preds, gts)) == [(0, 1), (1, 0)]


def test_label_shows_name_or_super_category():
    assert det().label.startswith("Stop")
    assert det(recognised=False, conf=0.3).label == "other (?)"


def test_result_rows_and_dict():
    r = PipelineResult(detections=[det(), det(recognised=False, conf=0.2)], image_size=(100, 50))
    rows = r.to_rows()
    assert rows[0]["panneau"] == "Stop" and rows[1]["panneau"] == "non reconnu"
    assert r.to_dict()["image_size"] == (100, 50)


def test_resolve_device_cpu_and_auto():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in {"cpu", "cuda"}


def test_missing_weights_give_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="models/weights"):
        RoadSignPipeline(tmp_path / "nope.pt", tmp_path / "nope2.pt")


def test_annotate_draws_without_changing_input():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    r = PipelineResult(detections=[det(box=(20, 30, 60, 70))], image_size=(200, 100))
    out = RoadSignPipeline.annotate(img, r)
    assert out.shape == img.shape and out.any() and not img.any()


def test_predict_rejects_non_colour_image():
    pipe = RoadSignPipeline.__new__(RoadSignPipeline)  # no weights needed for this check
    pipe.detector = SimpleNamespace()
    with pytest.raises(ValueError):
        pipe.predict(np.zeros((10, 10), dtype=np.uint8))
