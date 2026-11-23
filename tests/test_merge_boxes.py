"""Class-agnostic suppression of duplicate detector boxes (pipeline.merge_overlapping)."""

from roadsigns.pipeline import merge_overlapping


def test_two_boxes_on_the_same_sign_keep_only_the_most_confident():
    # 00813 case: "other (?)" and "prohibitory (?)" almost on the same sign
    boxes = [(1022, 232, 1077, 290), (1024, 230, 1078, 292), (915, 150, 1022, 250)]
    assert merge_overlapping(boxes, [0.40, 0.55, 0.95]) == [2, 1]


def test_box_inside_a_bigger_box_is_dropped():
    assert merge_overlapping([(0, 0, 100, 100), (10, 10, 60, 60)], [0.9, 0.5]) == [0]


def test_neighbour_signs_are_all_kept():
    stacked = [(915, 150, 1022, 250), (930, 252, 1010, 335)]  # yield above turn-right: touching only
    assert sorted(merge_overlapping(stacked, [0.9, 0.8])) == [0, 1]


def test_merge_can_be_disabled():
    same = [(0, 0, 10, 10), (0, 0, 10, 10)]
    assert sorted(merge_overlapping(same, [0.9, 0.8], merge_iou=0)) == [0, 1]
    assert sorted(merge_overlapping(same, [0.9, 0.8], merge_iou=None)) == [0, 1]
