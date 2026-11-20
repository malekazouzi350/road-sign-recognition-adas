from pathlib import Path

import pytest

from roadsigns.datasets.gtsdb import (
    CLASS_NAMES,
    NUM_CLASSES,
    SUPER_CATEGORIES,
    GTSDBDataset,
    parse_gt_file,
    super_category_id,
)


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_class_tables_are_complete():
    assert len(CLASS_NAMES) == NUM_CLASSES == 43
    assert CLASS_NAMES[14] == "Stop"


def test_every_class_has_exactly_one_super_category():
    counts = [0] * len(SUPER_CATEGORIES)
    for class_id in range(NUM_CLASSES):
        counts[super_category_id(class_id)] += 1
    assert counts == [12, 15, 8, 8]  # prohibitory, danger, mandatory, other


@pytest.mark.parametrize("class_id,expected", [(1, "prohibitory"), (14, "other"), (25, "danger"), (38, "mandatory")])
def test_super_category_examples(class_id, expected):
    assert SUPER_CATEGORIES[super_category_id(class_id)] == expected


def test_super_category_rejects_out_of_range():
    with pytest.raises(ValueError):
        super_category_id(43)


def test_parse_gt_file(tmp_path):
    gt = write(tmp_path / "gt.txt", "00000.ppm;774;411;815;446;11\n00001.ppm;10;20;29;39;14\n\n")
    anns = parse_gt_file(gt)
    assert len(anns) == 2
    assert anns[0].box == (774.0, 411.0, 815.0, 446.0)
    assert anns[1].width == 20 and anns[1].height == 20
    assert anns[1].class_name == "Stop"


@pytest.mark.parametrize("line", ["a;1;2;3;4", "a;1;2;x;4;5", "a;10;2;3;4;5", "a;1;2;3;4;99"])
def test_parse_gt_file_rejects_bad_lines(tmp_path, line):
    gt = write(tmp_path / "gt.txt", line + "\n")
    with pytest.raises(ValueError, match="gt.txt:1"):
        parse_gt_file(gt)


def test_dataset_includes_images_without_signs(tmp_path):
    for name in ("00000.ppm", "00001.ppm", "00002.ppm"):
        (tmp_path / name).write_bytes(b"")
    write(tmp_path / "gt.txt", "00000.ppm;1;1;20;20;1\n00000.ppm;30;30;50;50;14\n")
    ds = GTSDBDataset.load(tmp_path)
    assert len(ds.image_paths) == 3
    assert ds.num_signs == 2
    assert len(ds.signs_in(tmp_path / "00000.ppm")) == 2
    assert ds.signs_in(tmp_path / "00002.ppm") == []


def test_dataset_missing_folder_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="data/README.md"):
        GTSDBDataset.load(tmp_path / "nope")
