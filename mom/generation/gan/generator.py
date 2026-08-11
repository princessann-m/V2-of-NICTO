from __future__ import annotations

from math import log2

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim: int = 100, image_channels: int = 3, image_size: int = 64) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.image_channels = image_channels
        self.image_size = image_size

        num_blocks = int(log2(image_size)) - 2
        base_channels = 64
        out_ch = base_channels * (2 ** (num_blocks - 1))

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, out_ch * 4 * 4),
            nn.BatchNorm1d(out_ch * 4 * 4),
            nn.ReLU(inplace=True),
        )
        self.reshape_size = (out_ch, 4, 4)

        conv_layers: list[nn.Module] = []
        in_ch = out_ch
        for _ in range(num_blocks - 1):
            out_ch = max(32, in_ch // 2)
            conv_layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False))
            conv_layers.append(nn.BatchNorm2d(out_ch))
            conv_layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch

        conv_layers.append(nn.ConvTranspose2d(in_ch, image_channels, kernel_size=4, stride=2, padding=1, bias=False))
        conv_layers.append(nn.Tanh())
        self.conv = nn.Sequential(*conv_layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(x.size(0), *self.reshape_size)
        x = self.conv(x)
        return x
