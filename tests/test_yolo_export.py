import pytest

from roadsigns.datasets.gtsdb import SignAnnotation
from roadsigns.datasets.yolo_export import class_names, deduplicate, split_images, to_yolo_line


def ann(x1, y1, x2, y2, c=1, f="00000.ppm"):
    return SignAnnotation(f, x1, y1, x2, y2, c)


def test_to_yolo_line_inclusive_coordinates():
    # 10x10 px box (0..9 inclusive) in a 100x50 image -> centre (5, 5)
    line = to_yolo_line(ann(0, 0, 9, 9, c=14), 100, 50, "full")
    assert line == "14 0.050000 0.100000 0.100000 0.200000"


def test_to_yolo_line_super_mode_and_clipping():
    line = to_yolo_line(ann(95, 45, 120, 70, c=25), 100, 50, "super")  # 25 Road work -> danger (1)
    cls, cx, cy, w, h = line.split()
    assert cls == "1"
    assert 0 < float(cx) <= 1 and 0 < float(cy) <= 1 and float(w) == pytest.approx(0.05)


def test_class_names_modes():
    assert class_names("super") == ("prohibitory", "danger", "mandatory", "other")
    assert len(class_names("full")) == 43
    with pytest.raises(ValueError):
        class_names("bad")


def test_deduplicate():
    a = ann(10, 10, 40, 40, c=1)
    near = ann(10, 10, 40, 41, c=1)  # same sign, IoU ~0.97
    other_cls = ann(10, 10, 40, 40, c=2)  # same place, different class: kept, reported
    other_img = ann(10, 10, 40, 40, c=1, f="00001.ppm")
    kept, rep = deduplicate([a, a, near, other_cls, other_img])
    assert (rep.exact_duplicates, rep.near_duplicates, rep.overlapping_different_class) == (1, 1, 1)
    assert kept == [a, other_cls, other_img]


def test_split_images_official_test_and_disjoint():
    names = [f"{i:05d}.ppm" for i in range(900)]
    s = split_images(names, seed=0)
    assert len(s["test"]) == 300 and all(int(n[:5]) >= 600 for n in s["test"])
    assert len(s["val"]) == 90 and len(s["train"]) == 510
    assert not set(s["train"]) & set(s["val"])
    assert split_images(names, seed=0) == s
