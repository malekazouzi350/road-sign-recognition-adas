"""Video processing with a fake pipeline (no weights needed): frame skipping, output video, summary."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from roadsigns.pipeline import PipelineResult, RoadSignPipeline, SignDetection
from roadsigns.video import process_video, read_info


class FakePipeline:
    """Sees a 'Stop' in every frame and an unrecognised 'danger (?)' in frames 0-4."""

    def __init__(self):
        self.calls = 0

    def predict(self, frame):
        self.calls += 1
        dets = [SignDetection((10, 10, 30, 30), 0.9, "other", 14, "Stop", 0.99, 14, True, True)]
        if self.calls <= 5:
            dets.append(SignDetection((40, 10, 60, 30), 0.5, "danger", None, None, 0.3, 20, False, True))
        return PipelineResult(dets, (frame.shape[1], frame.shape[0]), {"total": 10.0})

    annotate = staticmethod(RoadSignPipeline.annotate)


@pytest.fixture
def video(tmp_path) -> Path:
    path = tmp_path / "drive.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (96, 64))
    for i in range(30):  # 3 s at 10 fps
        writer.write(np.full((64, 96, 3), i * 8, dtype=np.uint8))
    writer.release()
    return path


def test_read_info(video):
    info = read_info(video)
    assert (info.width, info.height, info.frame_count) == (96, 64, 30)
    assert info.duration_s == pytest.approx(3.0)


def test_read_info_rejects_non_video(tmp_path):
    bad = tmp_path / "x.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(ValueError):
        read_info(bad)


def test_every_n_frames_are_analysed_and_all_are_written(video, tmp_path):
    pipe = FakePipeline()
    summary = process_video(pipe, video, tmp_path / "out", every_n=3)
    assert pipe.calls == summary.analysed_frames == 10
    assert summary.written_frames == 30
    assert summary.output_path.exists() and read_info(summary.output_path).frame_count in (0, 30)
    assert summary.mean_ms_per_analysed_frame == pytest.approx(10.0)


def test_summary_counts_frames_per_sign(video):
    summary = process_video(FakePipeline(), video, None, every_n=3)
    rows = {r["panneau"]: r for r in summary.rows()}
    assert rows["Stop"]["images analysées où il apparaît"] == 10
    assert rows["danger (?)"]["images analysées où il apparaît"] == 5
    assert rows["danger (?)"]["reconnu"] == "non (?)"
    assert summary.rows()[0]["panneau"] == "Stop"  # most frequent first


def test_max_seconds_limits_the_analysis(video):
    summary = process_video(FakePipeline(), video, None, every_n=1, max_seconds=1)
    assert summary.analysed_frames == 10


def test_video_page_without_weights_and_without_upload(monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    app_dir = Path(__file__).resolve().parents[1] / "app"
    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))
    from components import detection as D

    monkeypatch.setattr(D, "missing_weights", lambda: ["models/weights/detector_v1_quick.pt"])
    at = testing.AppTest.from_file(str(app_dir / "app.py"), default_timeout=60)
    at.run()
    at.switch_page("pages/video_detection.py").run()
    assert not at.exception and any("detector_v1_quick.pt" in e.value for e in at.error)

    monkeypatch.setattr(D, "missing_weights", lambda: [])
    at.run()
    assert not at.exception and any("Importez une courte vidéo" in i.value for i in at.info)
