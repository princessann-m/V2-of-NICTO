from __future__ import annotations

from math import log2

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, image_channels: int = 3, image_size: int = 64, dropout: float = 0.3) -> None:
        super().__init__()
        self.image_channels = image_channels
        self.image_size = image_size

        num_blocks = int(log2(image_size)) - 2
        base_channels = 64

        layers: list[nn.Module] = []

        in_ch = image_channels
        out_ch = base_channels
        layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Dropout2d(dropout))

        in_ch = out_ch
        for _ in range(num_blocks - 1):
            out_ch = min(base_channels * 2, in_ch * 2)
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout2d(dropout))
            in_ch = out_ch

        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_ch * 4 * 4, 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
