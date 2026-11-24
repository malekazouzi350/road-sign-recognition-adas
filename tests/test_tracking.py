"""Tracking of signs across analysed video frames (roadsigns.tracking)."""

from roadsigns.pipeline import SignDetection
from roadsigns.tracking import SignTracker


def det(box, name="Speed limit (50km/h)", conf=0.95, family="prohibitory", recognised=True, agrees=True):
    return SignDetection(box, 0.9, family, 2 if recognised else None, name if recognised else None,
                         conf, 2, recognised, agrees)


def test_a_moving_growing_sign_keeps_one_track():
    tr = SignTracker()
    # the car approaches: the sign moves right and grows between analysed frames
    for t, box in enumerate([(800, 300, 830, 330), (815, 295, 850, 330), (835, 288, 878, 331), (860, 280, 912, 332)]):
        tracks = tr.update([det(box)], t * 0.2)
    assert len(tr.all_tracks()) == 1
    assert tracks[0].hits == 4 and tr.confirmed(tracks[0])


def test_name_is_decided_by_vote():
    tr = SignTracker()
    boxes = [(100, 100, 140, 140), (102, 100, 142, 140), (104, 100, 144, 140), (106, 100, 146, 140)]
    names = ["Speed limit (50km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)", "Speed limit (50km/h)"]
    for t, (box, name) in enumerate(zip(boxes, names)):
        tr.update([det(box, name=name, conf=0.9)], t * 0.2)
    track = tr.all_tracks()[0]
    assert track.name == "Speed limit (50km/h)"
    assert abs(track.vote_share - 0.75) < 1e-9


def test_two_signs_far_apart_get_two_tracks():
    tr = SignTracker()
    for t in range(3):
        tr.update([det((100 + t, 100, 140 + t, 140)), det((900 + t, 120, 940 + t, 160), name="Stop")], t * 0.2)
    tracks = tr.all_tracks()
    assert len(tracks) == 2 and {t.name for t in tracks} == {"Speed limit (50km/h)", "Stop"}


def test_track_is_closed_after_a_long_gap_and_a_new_one_starts():
    tr = SignTracker(max_gap_s=0.6)
    tr.update([det((100, 100, 140, 140))], 0.0)
    for t in range(1, 5):  # no sign for 0.8 s
        tr.update([], t * 0.2)
    tr.update([det((100, 100, 140, 140))], 1.0)
    assert [t.track_id for t in tr.all_tracks()] == [1, 2]


def test_short_detector_misses_do_not_split_the_track_even_with_every_frame_analysed():
    # 25 fps, every frame analysed, the detector misses the sign on 5 frames (0.2 s)
    tr = SignTracker(max_gap_s=0.6)
    for i in range(30):
        t = i / 25
        tr.update([] if 10 <= i < 15 else [det((300 + i, 200, 340 + i, 240))], t)
    assert len(tr.all_tracks()) == 1 and tr.all_tracks()[0].hits == 25


def test_fast_sign_close_to_the_car_is_followed_thanks_to_the_speed_estimate():
    # near the car the sign accelerates: +30, +40, +60, +80 px per analysed frame (box 40 px).
    # The last jumps are larger than the box diagonal: only the prediction can link them.
    tr = SignTracker()
    for i, x in enumerate([1000, 1030, 1070, 1130, 1210]):
        tr.update([det((x, 300, x + 40, 340))], i * 0.2)
    assert len(tr.all_tracks()) == 1


def test_unrecognised_sign_keeps_its_family_and_question_mark():
    tr = SignTracker()
    for t in range(2):
        tr.update([det((50, 50, 90, 90), recognised=False, conf=0.3, family="danger")], t * 0.2)
    track = tr.all_tracks()[0]
    assert track.name is None and track.label == "danger (?)"
