from pathlib import Path

import numpy as np
import pytest
import torch

from roadsigns.classification.data import SignCropDataset, crop_and_resize, preprocess_crop
from roadsigns.classification.model import INPUT_SIZE, TrafficSignCNN
from roadsigns.datasets.gtsrb import GTSRBSample, split_by_track


def fake_samples(n_classes=3, tracks_per_class=10, frames=30):
    out = []
    for c in range(n_classes):
        for t in range(tracks_per_class):
            for f in range(frames):
                out.append(GTSRBSample(Path(f"{c:05d}/{t:05d}_{f:05d}.ppm"), c, f"{c:05d}_{t:05d}", (0, 0, 9, 9)))
    return out


def test_split_has_no_track_leakage_and_covers_everything():
    samples = fake_samples()
    splits = split_by_track(samples, seed=0)
    tracks = {name: {s.track for s in part} for name, part in splits.items()}
    assert not tracks["train"] & tracks["val"]
    assert not tracks["train"] & tracks["test"]
    assert not tracks["val"] & tracks["test"]
    assert sum(len(p) for p in splits.values()) == len(samples)


def test_split_is_stratified_and_reproducible():
    samples = fake_samples()
    a = split_by_track(samples, seed=1)
    b = split_by_track(samples, seed=1)
    assert [s.path for s in a["val"]] == [s.path for s in b["val"]]
    for part in ("train", "val", "test"):
        assert {s.class_id for s in a[part]} == {0, 1, 2}


def test_split_rejects_bad_fractions():
    with pytest.raises(ValueError):
        split_by_track(fake_samples(), val_fraction=0.6, test_fraction=0.5)


def test_crop_and_resize_shape_and_clipping():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    out = crop_and_resize(img, (-5, -5, 250, 150))  # box larger than the image is clipped
    assert out.shape == (INPUT_SIZE, INPUT_SIZE, 3) and out.dtype == np.uint8


def test_preprocess_and_model_output_shape():
    x = preprocess_crop(np.full((INPUT_SIZE, INPUT_SIZE, 3), 128, dtype=np.uint8))
    assert x.shape == (3, INPUT_SIZE, INPUT_SIZE)
    model = TrafficSignCNN(43).eval()
    with torch.no_grad():
        logits = model(x.unsqueeze(0))
    assert logits.shape == (1, 43)


def test_dataset_returns_normalised_tensors():
    imgs = np.random.default_rng(0).integers(0, 255, (4, INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    ds = SignCropDataset(imgs, np.arange(4), augment=True)
    x, y = ds[2]
    assert x.shape == (3, INPUT_SIZE, INPUT_SIZE) and x.dtype == torch.float32 and y == 2


def test_motion_blur_kernel_is_normalised_and_horizontal():
    from roadsigns.classification.data import motion_blur_kernel
    k = motion_blur_kernel(7, 0)
    assert k.shape == (7, 7) and abs(k.sum() - 1) < 1e-5
    assert k[3].sum() > 0.99  # all the weight on the middle row
    assert abs(motion_blur_kernel(7, 20).sum() - 1) < 1e-5


def test_random_motion_blur_keeps_type_and_blurs():
    import numpy as np
    import torch
    from roadsigns.classification.data import RandomMotionBlur
    t = torch.from_numpy(np.random.default_rng(0).integers(0, 255, (3, 48, 48), dtype=np.uint8))
    out = RandomMotionBlur(p=1.0, lengths=(9, 9), max_angle=0)(t)
    assert out.dtype == torch.uint8 and out.shape == t.shape
    sharp = lambda x: x.float().diff(dim=2).abs().mean()  # noqa: E731
    assert sharp(out) < sharp(t) / 2
    assert torch.equal(RandomMotionBlur(p=0.0)(t), t)


def test_dataset_with_motion_blur_augmentation():
    import numpy as np
    from roadsigns.classification.data import SignCropDataset
    ds = SignCropDataset(np.zeros((2, 48, 48, 3), np.uint8), np.array([1, 2]), augment=True, motion_blur=True)
    x, y = ds[0]
    assert tuple(x.shape) == (3, 48, 48) and y == 1
