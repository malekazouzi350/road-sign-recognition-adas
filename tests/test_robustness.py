"""Simulated degradations and end-to-end scoring used by scripts/evaluate_robustness.py."""

import numpy as np
import pytest

from roadsigns.datasets.gtsdb import SignAnnotation
from roadsigns.evaluation.corruptions import CORRUPTIONS, LEVELS, apply
from roadsigns.evaluation.end_to_end import score_image, summarize
from roadsigns.pipeline import SignDetection

IMG = np.random.default_rng(0).integers(40, 220, (60, 80, 3), dtype=np.uint8)


@pytest.mark.parametrize("name", list(CORRUPTIONS))
def test_every_corruption_keeps_shape_and_type_and_is_deterministic(name):
    for level in LEVELS:
        out = apply(IMG, name, level, seed=3)
        assert out.shape == IMG.shape and out.dtype == np.uint8
        assert np.array_equal(out, apply(IMG, name, level, seed=3))
    assert np.array_equal(apply(IMG, name, 0), IMG)


def test_night_gets_darker_with_the_level():
    means = [apply(IMG, "night", lv).mean() for lv in (0, *LEVELS)]
    assert means == sorted(means, reverse=True)


def test_haze_reduces_contrast_and_blur_reduces_sharpness():
    assert apply(IMG, "haze", 3).std() < apply(IMG, "haze", 1).std() < IMG.std()
    def sharpness(im):
        return np.abs(np.diff(im.astype(float), axis=1)).mean()
    assert sharpness(apply(IMG, "motion_blur", 3)) < sharpness(apply(IMG, "motion_blur", 1)) < sharpness(IMG)


def test_unknown_corruption_or_level_is_refused():
    with pytest.raises(ValueError):
        apply(IMG, "rain", 1)
    with pytest.raises(ValueError):
        apply(IMG, "night", 4)


def gt(box, cls):
    return SignAnnotation("00700.ppm", *box, cls)


def det(box, cls, recognised=True):
    return SignDetection(box, 0.9, "prohibitory", cls if recognised else None, "x" if recognised else None,
                         0.9 if recognised else 0.3, cls, recognised, True)


def test_score_image_counts_every_outcome():
    truth = [gt((0, 0, 40, 40), 2), gt((100, 0, 140, 40), 14), gt((200, 0, 240, 40), 1), gt((300, 0, 340, 40), 5)]
    preds = [det((1, 1, 40, 40), 2),            # correct
             det((101, 0, 140, 40), 13),        # wrong name
             det((200, 1, 240, 40), 1, False),  # fallback "(?)"
             det((500, 0, 540, 40), 7)]         # false alarm; sign 4 is missed
    s = summarize(score_image(preds, truth))
    assert (s["correct"], s["wrong"], s["fallback"], s["missed"], s["false_alarms"]) == (1, 1, 1, 1, 1)
    assert s["end_to_end_recall"] == 0.25 and s["detection_precision"] == 0.75
