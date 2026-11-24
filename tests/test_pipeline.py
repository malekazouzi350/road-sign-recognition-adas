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


def test_label_stays_inside_image_at_the_right_border():
    from roadsigns.pipeline import place_label
    x, y = place_label((1330, 200, 1360, 260), (180, 24), (1360, 800), [])
    assert 0 <= x and x + 180 <= 1360 and 0 <= y and y + 24 <= 800


def test_label_goes_below_when_box_touches_the_top():
    from roadsigns.pipeline import place_label
    _, y = place_label((100, 0, 140, 40), (80, 24), (400, 300), [])
    assert y >= 40


def test_labels_do_not_overlap_for_neighbour_boxes():
    from roadsigns.pipeline import _overlaps, place_label
    placed = []
    for box in [(900, 150, 1020, 250), (930, 250, 1010, 335), (1020, 230, 1075, 290)]:
        x, y = place_label(box, (190, 24), (1360, 800), placed)
        placed.append((x, y, x + 190, y + 24))
    assert not any(_overlaps(a, b) for i, a in enumerate(placed) for b in placed[i + 1:])


def test_label_avoids_other_boxes_when_there_is_room():
    from roadsigns.pipeline import _overlaps, place_label
    yield_box, arrow_box = (915, 150, 1022, 250), (930, 252, 1010, 335)
    x, y = place_label(arrow_box, (190, 24), (1360, 800), [], [yield_box])
    assert not _overlaps((x, y, x + 190, y + 24), yield_box)


def test_central_square_is_centred_and_inside():
    from roadsigns.pipeline import central_square
    assert central_square(1000, 500) == (300, 50, 700, 450)


def test_classify_closeup_returns_top3_sorted():
    import numpy as np
    from roadsigns.classification.model import TrafficSignCNN
    from roadsigns.pipeline import RoadSignPipeline

    pipe = object.__new__(RoadSignPipeline)  # CNN only: no detector weights needed
    pipe.device = "cpu"
    pipe.classifier = TrafficSignCNN(43).eval()
    top = pipe.classify_closeup(np.random.default_rng(0).integers(0, 255, (240, 320, 3), dtype=np.uint8))
    assert len(top) == 3 and top[0][2] >= top[1][2] >= top[2][2]
    assert all(0 <= c < 43 and isinstance(n, str) for c, n, _ in top)
