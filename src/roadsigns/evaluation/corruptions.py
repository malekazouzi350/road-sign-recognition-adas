"""Artificial image degradations to test robustness (night, motion blur, haze, sensor noise).

These are SIMULATIONS applied to the GTSDB test images. They say how fast the system
degrades when the image gets darker, blurrier, paler or noisier; they do NOT replace
real photos taken at night or against the sun (no headlight glare, no reflections,
no real camera behaviour). Report them as "simulated conditions".

Each corruption has 3 levels (1 = light, 3 = strong) and is deterministic for a given
``seed``, so two runs give the same numbers.
"""

from __future__ import annotations

import cv2
import numpy as np

LEVELS = (1, 2, 3)

# name -> (French label for tables/slides, parameters for levels 1, 2, 3, description of the parameter)
CORRUPTIONS: dict[str, tuple[str, tuple, str]] = {
    "night": ("Nuit (image sombre + bruit)", ((0.5, 1.2, 4), (0.3, 1.4, 6), (0.15, 1.6, 8)),
              "(luminosité x, gamma, bruit σ)"),
    "motion_blur": ("Flou de bougé", (5, 9, 15), "longueur du flou horizontal (px)"),
    "haze": ("Voile clair (brouillard, contre-jour)", (0.3, 0.5, 0.7), "part de blanc mélangée"),
    "noise": ("Bruit du capteur", (10, 20, 35), "bruit gaussien σ (niveaux de gris 0-255)"),
}


def apply(image_bgr: np.ndarray, name: str, level: int, seed: int = 0) -> np.ndarray:
    """Return a degraded copy of ``image_bgr`` (uint8 BGR). ``level`` 0 returns the image unchanged."""
    if level == 0:
        return image_bgr.copy()
    if name not in CORRUPTIONS:
        raise ValueError(f"unknown corruption {name!r}; choose from {sorted(CORRUPTIONS)}")
    if level not in LEVELS:
        raise ValueError(f"level must be 0, 1, 2 or 3, got {level}")
    param = CORRUPTIONS[name][1][level - 1]
    rng = np.random.default_rng(seed)
    img = image_bgr.astype(np.float32) / 255.0

    if name == "night":
        brightness, gamma, sigma = param
        img = np.power(img, gamma) * brightness
        img = img + rng.normal(0.0, sigma / 255.0, img.shape)
    elif name == "motion_blur":
        kernel = np.zeros((param, param), np.float32)
        kernel[param // 2, :] = 1.0 / param
        img = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REFLECT)
    elif name == "haze":
        img = img * (1.0 - param) + param
    elif name == "noise":
        img = img + rng.normal(0.0, param / 255.0, img.shape)

    return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def describe(name: str) -> dict:
    label, params, meaning = CORRUPTIONS[name]
    return {"label": label, "parameter": meaning,
            "levels": {str(lv): params[lv - 1] for lv in LEVELS}}
