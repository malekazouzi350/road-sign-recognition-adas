"""Run the road-sign pipeline on a video file, frame by frame.

    from roadsigns.pipeline import RoadSignPipeline
    from roadsigns.video import process_video
    summary = process_video(RoadSignPipeline.from_default_weights(), "drive.mp4", "drive_annotated.webm")

Only one frame out of ``every_n`` is analysed (the CPU is too slow for every frame);
the frames in between are written with the boxes of the last analysed frame, so the
output video keeps the original length and speed.

Frames wider than ``max_width`` (e.g. 4K) are reduced first: the detector works at
``imgsz`` (1024 px) anyway, and writing a 4K video is very slow on a CPU.

No temporal smoothing yet: each analysed frame is independent. The per-sign summary
counts in how many analysed frames each sign appeared and how often the detector and
the CNN disagreed on its family; a sign seen once, or with a disagreement, is marked
"à vérifier".
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2

# (fourcc, extension): WebM/VP8 plays in every browser; MP4/mp4v is a fallback for download only.
VIDEO_FORMATS = (("VP80", ".webm"), ("mp4v", ".mp4"))


@dataclass
class VideoInfo:
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


@dataclass
class SignSeen:
    """One sign (or one unrecognised super-category) across the video."""

    label: str                  # class name, or "<super-category> (?)"
    super_category: str
    recognised: bool
    frames: int = 0             # number of ANALYSED frames where it appears
    first_time_s: float = 0.0
    best_cnn_conf: float = 0.0
    disagreements: int = 0      # frames where the CNN class is not in the detector's family

    @property
    def reliable(self) -> bool:
        """Seen on at least 2 analysed frames, and detector + CNN always agree."""
        return self.frames >= 2 and self.disagreements == 0


@dataclass
class VideoSummary:
    info: VideoInfo
    output_path: Path | None
    playable_in_browser: bool
    analysed_frames: int = 0
    written_frames: int = 0
    every_n: int = 1
    output_size: tuple[int, int] = (0, 0)   # (w, h) after the optional reduction
    mean_ms_per_analysed_frame: float = 0.0
    signs: dict[str, SignSeen] = field(default_factory=dict)

    def rows(self) -> list[dict]:
        """One row per sign, most frequent first (for a table / CSV)."""
        ordered = sorted(self.signs.values(), key=lambda s: (-s.frames, s.first_time_s))
        return [{
            "panneau": s.label,
            "super-catégorie": s.super_category,
            "reconnu": "oui" if s.recognised else "non (?)",
            "images analysées où il apparaît": s.frames,
            "première apparition (s)": round(s.first_time_s, 1),
            "meilleure confiance CNN": round(s.best_cnn_conf, 3),
            "accord détecteur / CNN": ("—" if not s.recognised else
                                       "oui" if s.disagreements == 0 else f"non ({s.disagreements} fois)"),
            "fiabilité": "confirmé" if s.reliable else "à vérifier",
        } for s in ordered]


def read_info(path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"cannot open video: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        info = VideoInfo(fps=fps if fps > 0 else 25.0, frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                         width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                         height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        cap.release()
    if info.width <= 0 or info.height <= 0:
        raise ValueError(f"no readable frames in: {path}")
    return info


def open_writer(stem_path: Path, fps: float, size: tuple[int, int]) -> tuple[cv2.VideoWriter, Path, bool]:
    """Try WebM (browser-playable) first, then MP4. Returns (writer, path, playable_in_browser)."""
    for i, (fourcc, ext) in enumerate(VIDEO_FORMATS):
        path = stem_path.with_suffix(ext)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if writer.isOpened():
            return writer, path, i == 0
        writer.release()
    raise RuntimeError("OpenCV cannot write WebM or MP4 video on this computer")


def process_video(
    pipeline,
    input_path: str | Path,
    output_path: str | Path | None,
    every_n: int = 1,
    max_seconds: float | None = None,
    progress: Callable[[float], None] | None = None,
    max_width: int | None = 1280,
) -> VideoSummary:
    """Analyse ``input_path``; write the annotated video next to ``output_path`` (extension chosen
    automatically, see VIDEO_FORMATS). ``output_path=None`` -> only the summary is computed."""
    every_n = max(1, int(every_n))
    info = read_info(input_path)
    last_frame = info.frame_count
    if max_seconds is not None:
        last_frame = min(last_frame, int(max_seconds * info.fps)) if last_frame > 0 else int(max_seconds * info.fps)

    size = output_size(info.width, info.height, max_width)
    resize = size != (info.width, info.height)

    writer, out_path, playable = (None, None, False)
    if output_path is not None:
        writer, out_path, playable = open_writer(Path(output_path), info.fps, size)

    summary = VideoSummary(info=info, output_path=out_path, playable_in_browser=playable, every_n=every_n,
                           output_size=size)
    cap = cv2.VideoCapture(str(input_path))
    last_result, total_ms, index = None, 0.0, 0
    try:
        while last_frame <= 0 or index < last_frame:
            ok, frame = cap.read()
            if not ok:
                break
            if resize:
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            if index % every_n == 0:
                last_result = pipeline.predict(frame)
                total_ms += last_result.timings_ms.get("total", 0.0)
                summary.analysed_frames += 1
                _update_signs(summary.signs, last_result, index / info.fps)
            if writer is not None:
                writer.write(pipeline.annotate(frame, last_result) if last_result else frame)
                summary.written_frames += 1
            index += 1
            if progress is not None and last_frame > 0:
                progress(min(1.0, index / last_frame))
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if summary.analysed_frames:
        summary.mean_ms_per_analysed_frame = total_ms / summary.analysed_frames
    if progress is not None:
        progress(1.0)
    return summary


def output_size(width: int, height: int, max_width: int | None) -> tuple[int, int]:
    """(w, h) after reducing to ``max_width`` (aspect ratio kept, even numbers for the codecs)."""
    if not max_width or width <= max_width:
        return width, height
    scale = max_width / width
    return max_width - max_width % 2, max(2, int(round(height * scale / 2)) * 2)


def _update_signs(signs: dict[str, SignSeen], result, time_s: float) -> None:
    seen_here = set()
    for d in result.detections:
        label = d.class_name if d.recognised else f"{d.super_category} (?)"
        if label in seen_here:          # same sign twice in one frame counts once
            continue
        seen_here.add(label)
        s = signs.get(label)
        if s is None:
            s = signs[label] = SignSeen(label=label, super_category=d.super_category,
                                        recognised=d.recognised, first_time_s=time_s)
        s.frames += 1
        s.best_cnn_conf = max(s.best_cnn_conf, d.cnn_conf)
        if d.recognised and not d.agrees_with_detector:
            s.disagreements += 1
