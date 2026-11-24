"""Crops, caching, augmentation and PyTorch datasets for the sign classifier.

The same ``preprocess_crop`` function is used for training images and, later,
for the crops produced by the YOLO detector, so both go through identical
preprocessing (same size, same normalisation).
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from roadsigns.classification.model import INPUT_SIZE, MEAN, STD
from roadsigns.datasets.gtsrb import GTSRBSample

logger = logging.getLogger(__name__)

_NORMALIZE = v2.Normalize(MEAN, STD)


def crop_and_resize(image_bgr: np.ndarray, box: tuple[int, int, int, int], size: int = INPUT_SIZE) -> np.ndarray:
    """Crop ``box`` (x1, y1, x2, y2 inclusive) from a BGR image and resize to ``size`` x ``size`` RGB."""
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"empty crop {box} in image of size {w}x{h}")
    crop = image_bgr[y1:y2 + 1, x1:x2 + 1]
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def preprocess_crop(crop_rgb_uint8: np.ndarray) -> torch.Tensor:
    """HxWx3 uint8 RGB (already resized) -> normalised 3xHxW float tensor."""
    t = torch.from_numpy(np.ascontiguousarray(crop_rgb_uint8)).permute(2, 0, 1).float().div_(255.0)
    return _NORMALIZE(t)


def load_crops(samples: list[GTSRBSample], cache_path: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (images N x 48 x 48 x 3 uint8 RGB, labels N int64), using a .npz cache if valid.

    The cache stores the file paths it was built from; it is rebuilt automatically
    if the list of samples changes.
    """
    keys = np.array([str(s.path) for s in samples])
    if cache_path is not None and Path(cache_path).is_file():
        with np.load(cache_path, allow_pickle=False) as data:
            if data["images"].shape[1] == INPUT_SIZE and np.array_equal(data["keys"], keys):
                logger.info("Loaded %d crops from cache %s", len(keys), cache_path)
                return data["images"], data["labels"]
        logger.info("Cache %s is outdated: rebuilding", cache_path)

    images = np.empty((len(samples), INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    labels = np.empty(len(samples), dtype=np.int64)
    for i, s in enumerate(samples):
        img = cv2.imread(str(s.path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image {s.path}")
        images[i] = crop_and_resize(img, s.roi)
        labels[i] = s.class_id
        if (i + 1) % 5000 == 0:
            logger.info("  cropped %d/%d images", i + 1, len(samples))

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, images=images, labels=labels, keys=keys)
        logger.info("Saved crop cache to %s", cache_path)
    return images, labels


class RandomMotionBlur(torch.nn.Module):
    """Directional (motion) blur on a uint8 3xHxW tensor, with probability ``p``.

    The blur length is drawn in ``lengths`` (pixels at 48x48) and the direction mostly
    horizontal (angle in ``max_angle`` degrees around it), like a car moving or turning.
    Added after the robustness test showed motion blur is the only condition that makes
    the CNN give WRONG names.
    """

    def __init__(self, p: float = 0.3, lengths: tuple[int, int] = (3, 11), max_angle: float = 20.0) -> None:
        super().__init__()
        self.p, self.lengths, self.max_angle = p, lengths, max_angle

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.p:
            return t
        length = int(torch.randint(self.lengths[0], self.lengths[1] + 1, (1,)).item())
        angle = float((torch.rand(1).item() * 2 - 1) * self.max_angle)
        kernel = motion_blur_kernel(length, angle)
        img = t.permute(1, 2, 0).numpy()
        out = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REFLECT)
        return torch.from_numpy(np.ascontiguousarray(out)).permute(2, 0, 1)


def motion_blur_kernel(length: int, angle_deg: float) -> np.ndarray:
    """Normalised line kernel of ``length`` pixels at ``angle_deg`` (0 = horizontal)."""
    length = max(1, int(length))
    kernel = np.zeros((length, length), np.float32)
    kernel[length // 2, :] = 1.0
    if angle_deg:
        rot = cv2.getRotationMatrix2D(((length - 1) / 2, (length - 1) / 2), angle_deg, 1.0)
        kernel = cv2.warpAffine(kernel, rot, (length, length))
    return kernel / max(kernel.sum(), 1e-6)


def train_augmentation(motion_blur: bool = False) -> v2.Compose:
    """Realistic variations only. No horizontal/vertical flip: it would change a sign's
    meaning ("turn left" <-> "turn right") or produce signs that do not exist.
    ``motion_blur`` adds RandomMotionBlur (classifier v2, see scripts/train_classifier.py)."""
    steps = [
        v2.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.9, 1.1)),  # imprecise detector boxes, viewpoint
        v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.02),  # lighting, weather
        v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.2),  # focus blur
    ]
    if motion_blur:
        steps.append(RandomMotionBlur())
    return v2.Compose(steps)


class SignCropDataset(Dataset):
    """In-memory dataset of 48x48 RGB crops."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, augment: bool = False,
                 motion_blur: bool = False) -> None:
        if len(images) != len(labels):
            raise ValueError("images and labels must have the same length")
        self.images = images
        self.labels = labels
        self.augment = train_augmentation(motion_blur) if augment else None

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        t = torch.from_numpy(self.images[idx]).permute(2, 0, 1)  # uint8 3xHxW
        if self.augment is not None:
            t = self.augment(t)
        t = _NORMALIZE(t.float().div(255.0))
        return t, int(self.labels[idx])
