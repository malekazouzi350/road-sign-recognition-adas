"""Run the road-sign pipeline on a video file, frame by frame.

    from roadsigns.pipeline import RoadSignPipeline
    from roadsigns.video import process_video
    summary = process_video(RoadSignPipeline.from_default_weights(), "drive.mp4", "drive_annotated.webm")

Only one frame out of ``every_n`` is analysed (the CPU is too slow for every frame);
the frames in between are written with the boxes of the last analysed frame, so the
output video keeps the original length and speed.

Frames wider than ``max_width`` (e.g. 4K) are reduced first: the detector works at
``imgsz`` (1024 px) anyway, and writing a 4K video is very slow on a CPU.

With ``track=True`` (default) each sign is followed from one analysed frame to the next
(roadsigns.tracking): its name is decided by a vote over all its frames, the video shows
that stable name with the track number, and the summary has one row per physical sign.
With ``track=False`` each analysed frame is judged alone and the summary has one row per
name. In both cases a sign seen once, or with a detector/CNN disagreement, is marked
"à vérifier" (with tracking: also when the CNN changed its mind too often).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2

from roadsigns.pipeline import SignDetection
from roadsigns.tracking import SignTracker, Track

MIN_VOTE_SHARE = 0.7  # with tracking: below this, the CNN changed its mind too often -> "à vérifier"

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
    track_id: int | None = None         # set when tracking is on
    last_time_s: float | None = None
    vote_share: float | None = None     # with tracking: part of the CNN votes for the chosen name

    @property
    def reliable(self) -> bool:
        """Seen on at least 2 analysed frames, detector + CNN always agree, and (with tracking)
        the CNN mostly gave the same name."""
        consistent = self.vote_share is None or not self.recognised or self.vote_share >= MIN_VOTE_SHARE
        return self.frames >= 2 and self.disagreements == 0 and consistent


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
    tracking: bool = False
    signs: dict[str, SignSeen] = field(default_factory=dict)

    def rows(self) -> list[dict]:
        """One row per sign (per physical sign with tracking), most frequent first (table / CSV)."""
        ordered = sorted(self.signs.values(), key=lambda s: (-s.frames, s.first_time_s))
        return [{
            **({"suivi n°": s.track_id} if self.tracking else {}),
            "panneau": s.label,
            "super-catégorie": s.super_category,
            "reconnu": "oui" if s.recognised else "non (?)",
            "images analysées où il apparaît": s.frames,
            "première apparition (s)": round(s.first_time_s, 1),
            "meilleure confiance CNN": round(s.best_cnn_conf, 3),
            "accord détecteur / CNN": ("—" if not s.recognised else
                                       "oui" if s.disagreements == 0 else f"non ({s.disagreements} fois)"),
            **({"dernière apparition (s)": round(s.last_time_s, 1),
                "vote CNN": "—" if not s.recognised else f"{s.vote_share:.0%}"} if self.tracking else {}),
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
    track: bool = True,
) -> VideoSummary:
    """Analyse ``input_path``; write the annotated video next to ``output_path`` (extension chosen
    automatically, see VIDEO_FORMATS). ``output_path=None`` -> only the summary is computed.
    ``track`` -> follow each sign across frames (see module docstring)."""
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
                           output_size=size, tracking=track)
    tracker = SignTracker() if track else None
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
                if tracker is not None:
                    tracks = tracker.update(last_result.detections, index / info.fps)
                    last_result = replace(last_result, detections=[
                        _with_track_label(d, t) for d, t in zip(last_result.detections, tracks)])
                else:
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
    if tracker is not None:
        summary.signs = {f"#{t.track_id}": _seen_from_track(t) for t in tracker.all_tracks()}
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


@dataclass
class TrackedDetection(SignDetection):
    """A detection drawn with its track: voted name (or family + "?") and track number."""

    track_label: str = ""

    @property
    def label(self) -> str:
        return self.track_label


def _with_track_label(d, track: Track) -> TrackedDetection:
    """The detection as drawn in the video: the track's voted name and its number."""
    text = (f"#{track.track_id} {track.name} ({d.cnn_conf:.2f})" if track.name
            else f"#{track.track_id} {track.family} (?)")
    fields = {k: getattr(d, k) for k in SignDetection.__dataclass_fields__}
    fields.update(super_category=track.family, recognised=track.name is not None,
                  class_name=track.name or d.class_name)
    return TrackedDetection(**fields, track_label=text)


def _seen_from_track(t: Track) -> SignSeen:
    return SignSeen(label=t.label, super_category=t.family, recognised=t.name is not None,
                    frames=t.hits, first_time_s=t.first_time_s, best_cnn_conf=t.best_cnn_conf,
                    disagreements=t.disagreements, track_id=t.track_id, last_time_s=t.last_time_s,
                    vote_share=t.vote_share if t.name else None)
