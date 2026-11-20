import pytest

from roadsigns.evaluation.matching import iou, match_boxes


def test_iou_identical_disjoint_and_half():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    # 10x10 and 10x5 sharing 50 px: 50 / (100 + 50 - 50) = 0.5
    assert iou((0, 0, 10, 10), (0, 0, 10, 5)) == pytest.approx(0.5)


def test_perfect_match():
    r = match_boxes([((0, 0, 10, 10), 0.9)], [(0, 0, 10, 10)])
    assert (r.tp, r.fp, r.fn) == (1, 0, 0)
    assert r.precision == 1.0 and r.recall == 1.0


def test_each_ground_truth_matched_once():
    # Two predictions on the same sign: one TP, one FP (duplicate detection).
    preds = [((0, 0, 10, 10), 0.6), ((0, 0, 10, 10), 0.9)]
    r = match_boxes(preds, [(0, 0, 10, 10)])
    assert (r.tp, r.fp, r.fn) == (1, 1, 0)


def test_low_iou_is_false_positive_and_missed_gt():
    r = match_boxes([((0, 0, 10, 10), 0.9)], [(8, 8, 20, 20)])
    assert (r.tp, r.fp, r.fn) == (0, 1, 1)


def test_no_predictions_precision_undefined():
    r = match_boxes([], [(0, 0, 10, 10)])
    assert (r.tp, r.fp, r.fn) == (0, 0, 1)
    assert r.precision is None and r.recall == 0.0


def test_results_add_up():
    a = match_boxes([((0, 0, 10, 10), 0.9)], [(0, 0, 10, 10)])
    b = match_boxes([], [(0, 0, 5, 5)])
    total = a + b
    assert (total.tp, total.fp, total.fn) == (1, 0, 1)
