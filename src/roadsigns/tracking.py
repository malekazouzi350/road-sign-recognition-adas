"""Follow each sign from one analysed video frame to the next (simple tracking by position).

Why: in a video the same sign is seen on many frames. Without tracking, each frame is
judged alone, so the name can flicker (e.g. "Speed limit 50" then "Speed limit 30" once)
and a box seen on a single frame looks as trustworthy as a sign seen ten times.

How (no learning, no extra library):
1. Prediction: from its last two positions, each track predicts where its sign should be
   now (constant speed). Near the car a sign moves and grows fast, so comparing with the
   predicted box works better than with the last box.
2. Association: a new box is linked to a track when it overlaps the predicted box
   (IoU >= ``min_iou``) or its centre is within ``max_center_dist`` x the diagonal of the
   larger of the two boxes. Pairs are matched greedily, closest first; each track and each
   box is used once.
3. Name by vote: every frame where the CNN recognised the sign adds its CNN confidence to
   that class. The track's name is the class with the largest total; its "vote share" says
   how consistent the CNN was.
4. A track is "confirmed" after ``min_hits`` frames. It is closed when it has not been
   seen for more than ``max_gap_s`` SECONDS (a duration, not a number of frames, so the
   behaviour is the same whether 1 frame in 1 or 1 frame in 5 is analysed). Short gaps
   happen when the detector misses the sign on a few frames.

Limits: this is position-only tracking. Two signs side by side that cross each other, or a
sign hidden longer than ``max_gap_s``, can be mixed up or split into two tracks. We have no
video ground truth, so the tracker is checked with unit tests and by watching videos, not
measured.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from roadsigns.evaluation.matching import iou

Box = tuple[float, float, float, float]


@dataclass
class Track:
    track_id: int
    box: Box
    super_category: str
    first_time_s: float
    last_time_s: float
    hits: int = 0                   # analysed frames where this sign was seen
    prev_box: Box | None = None     # previous observation, for the speed estimate
    prev_time_s: float | None = None
    best_cnn_conf: float = 0.0
    disagreements: int = 0          # frames where the CNN class is not in the detector's family
    votes: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    family_votes: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @property
    def name(self) -> str | None:
        """Class name with the largest summed CNN confidence, or None if never recognised."""
        return max(self.votes, key=self.votes.get) if self.votes else None

    @property
    def family(self) -> str:
        return max(self.family_votes, key=self.family_votes.get) if self.family_votes else self.super_category

    @property
    def vote_share(self) -> float:
        """Part of the votes that went to the winning name (1.0 = the CNN never changed its mind)."""
        total = sum(self.votes.values())
        return self.votes[self.name] / total if total else 0.0

    @property
    def label(self) -> str:
        return self.name if self.name else f"{self.family} (?)"

    def predicted_box(self, time_s: float) -> Box:
        """Last box shifted by the last observed speed (no prediction after one observation)."""
        if self.prev_box is None or self.prev_time_s is None or self.last_time_s <= self.prev_time_s:
            return self.box
        k = (time_s - self.last_time_s) / (self.last_time_s - self.prev_time_s)
        k = min(k, 3.0)  # do not extrapolate far after a long gap
        dx = (_cx(self.box) - _cx(self.prev_box)) * k
        dy = (_cy(self.box) - _cy(self.prev_box)) * k
        x1, y1, x2, y2 = self.box
        return x1 + dx, y1 + dy, x2 + dx, y2 + dy


class SignTracker:
    def __init__(self, min_iou: float = 0.2, max_center_dist: float = 1.0,
                 max_gap_s: float = 0.6, min_hits: int = 2) -> None:
        self.min_iou, self.max_center_dist = min_iou, max_center_dist
        self.max_gap_s, self.min_hits = max_gap_s, min_hits
        self.active: list[Track] = []
        self.finished: list[Track] = []
        self._next_id = 1

    def update(self, detections, time_s: float) -> list[Track]:
        """Add the detections of one analysed frame. Returns the track of each detection (same order)."""
        # 1. close the tracks not seen for too long
        still_active = []
        for track in self.active:
            (self.finished if time_s - track.last_time_s > self.max_gap_s else still_active).append(track)
        self.active = still_active

        # 2. associate boxes with tracks, closest pairs first
        pairs = []
        for t_idx, track in enumerate(self.active):
            predicted = track.predicted_box(time_s)
            for d_idx, d in enumerate(detections):
                cost = self._cost(predicted, tuple(map(float, d.box)))
                if cost is not None:
                    pairs.append((cost, t_idx, d_idx))
        pairs.sort()
        used_tracks, assigned = set(), {}
        for _, t_idx, d_idx in pairs:
            if t_idx in used_tracks or d_idx in assigned:
                continue
            used_tracks.add(t_idx)
            assigned[d_idx] = self.active[t_idx]

        # 3. update matched tracks, start new ones
        out: list[Track] = []
        for d_idx, d in enumerate(detections):
            track = assigned.get(d_idx)
            if track is None:
                track = Track(self._next_id, tuple(map(float, d.box)), d.super_category, time_s, time_s)
                self._next_id += 1
                self.active.append(track)
            self._add(track, d, time_s)
            out.append(track)
        return out

    def all_tracks(self) -> list[Track]:
        return sorted(self.finished + self.active, key=lambda t: t.track_id)

    def confirmed(self, track: Track) -> bool:
        return track.hits >= self.min_hits

    # ------------------------------------------------------------------
    def _cost(self, a: Box, b: Box) -> float | None:
        """Lower = closer. None = too far to be the same sign."""
        overlap = iou(a, b)
        diag = max(math.hypot(a[2] - a[0], a[3] - a[1]), math.hypot(b[2] - b[0], b[3] - b[1])) or 1.0
        dist = math.hypot(_cx(a) - _cx(b), _cy(a) - _cy(b)) / diag
        if overlap >= self.min_iou or dist <= self.max_center_dist:
            return dist - overlap
        return None

    @staticmethod
    def _add(track: Track, d, time_s: float) -> None:
        if track.hits:  # keep the previous observation for the speed estimate
            track.prev_box, track.prev_time_s = track.box, track.last_time_s
        track.box = tuple(map(float, d.box))
        track.hits += 1
        track.last_time_s = time_s
        track.best_cnn_conf = max(track.best_cnn_conf, d.cnn_conf)
        track.family_votes[d.super_category] += 1
        if d.recognised:
            track.votes[d.class_name] += d.cnn_conf
            if not d.agrees_with_detector:
                track.disagreements += 1


def _cx(b: Box) -> float:
    return (b[0] + b[2]) / 2


def _cy(b: Box) -> float:
    return (b[1] + b[3]) / 2
