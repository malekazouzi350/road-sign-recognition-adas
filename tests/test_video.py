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

    monkeypatch.setattr(D, "missing_weights", lambda: ["models/weights/detector_v2_1280.pt"])
    at = testing.AppTest.from_file(str(app_dir / "app.py"), default_timeout=60)
    at.run()
    at.switch_page("pages/video_detection.py").run()
    assert not at.exception and any("detector_v2_1280.pt" in e.value for e in at.error)

    monkeypatch.setattr(D, "missing_weights", lambda: [])
    at.run()
    assert not at.exception and any("Importez une courte vidéo" in i.value for i in at.info)


def test_4k_is_reduced_to_1280_keeping_the_aspect_ratio():
    from roadsigns.video import output_size
    assert output_size(3840, 2160, 1280) == (1280, 720)
    assert output_size(1280, 720, 1280) == (1280, 720)
    assert output_size(1920, 1080, None) == (1920, 1080)


def test_large_video_is_written_reduced(tmp_path):
    path = tmp_path / "big.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (1600, 900))
    for _ in range(4):
        writer.write(np.zeros((900, 1600, 3), dtype=np.uint8))
    writer.release()
    summary = process_video(FakePipeline(), path, tmp_path / "out", max_width=800)
    assert summary.output_size == (800, 450)
    assert read_info(summary.output_path).width == 800


def test_disagreement_and_single_frame_signs_are_marked_to_check(video):
    class Disagree(FakePipeline):
        def predict(self, frame):
            result = super().predict(frame)
            if self.calls == 1:  # one frame: CNN says "Priority road" (other) inside a prohibitory box
                result.detections.append(SignDetection((70, 10, 90, 30), 0.6, "prohibitory", 12,
                                                       "Priority road", 0.89, 12, True, False))
            return result

    rows = {r["panneau"]: r for r in process_video(Disagree(), video, None, every_n=3).rows()}
    assert rows["Stop"]["fiabilité"] == "confirmé" and rows["Stop"]["accord détecteur / CNN"] == "oui"
    assert rows["Priority road"]["fiabilité"] == "à vérifier"
    assert rows["Priority road"]["accord détecteur / CNN"] == "non (1 fois)"
    assert rows["danger (?)"]["accord détecteur / CNN"] == "—"


def test_tracking_gives_one_row_per_physical_sign_with_a_voted_name(video):
    class Flicker(FakePipeline):
        def predict(self, frame):
            result = super().predict(frame)
            if self.calls == 2:  # the CNN hesitates once on the same Stop sign
                d = result.detections[0]
                result.detections[0] = SignDetection(d.box, d.det_conf, d.super_category, 13, "Yield",
                                                     0.7, 13, True, True)
            return result

    summary = process_video(Flicker(), video, None, every_n=3, track=True)
    rows = summary.rows()
    stop = [r for r in rows if r["panneau"] == "Stop"]
    assert len(stop) == 1 and stop[0]["suivi n°"] >= 1
    assert stop[0]["images analysées où il apparaît"] == 10       # the "Yield" frame is part of the Stop track
    assert not any(r["panneau"] == "Yield" for r in rows)
    assert stop[0]["vote CNN"] == "93%"                            # 9 x 0.99 vs 1 x 0.7
    assert stop[0]["fiabilité"] == "confirmé"


def test_without_tracking_the_old_per_name_summary_is_kept(video):
    summary = process_video(FakePipeline(), video, None, every_n=3, track=False)
    assert "suivi n°" not in summary.rows()[0]
