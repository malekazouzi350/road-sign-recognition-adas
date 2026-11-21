"""A small CNN for 43-class traffic sign recognition, written by the team.

Input: RGB crops resized to 48x48, normalised. Output: 43 logits.

Architecture (about 1.5 M parameters)::

    [Conv3x3-BN-ReLU] x2, 32 ch  -> MaxPool  (48 -> 24)
    [Conv3x3-BN-ReLU] x2, 64 ch  -> MaxPool  (24 -> 12)
    [Conv3x3-BN-ReLU] x2, 128 ch -> MaxPool  (12 -> 6)
    Dropout -> Linear(128*6*6, 256) -> ReLU -> Dropout -> Linear(256, 43)

Design choices to explain to the jury:
  * two 3x3 convolutions per block: same receptive field as one 5x5 with fewer
    parameters and one more non-linearity (the VGG idea);
  * BatchNorm: stabilises and speeds up training;
  * Dropout: limits over-fitting, GTSRB has many near-identical frames;
  * 48x48 input: most GTSRB signs are 30-60 px, so 48 keeps detail without
    wasting computation.
"""

from __future__ import annotations

import torch
from torch import nn

INPUT_SIZE = 48
# Channel statistics used for normalisation (ImageNet values: a standard, documented choice).
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class TrafficSignCNN(nn.Module):
    def __init__(self, num_classes: int = 43, dropout: float = 0.4) -> None:
        super().__init__()
        self.features = nn.Sequential(_block(3, 32), _block(32, 64), _block(64, 128))
        spatial = INPUT_SIZE // 8
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128 * spatial * spatial, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
