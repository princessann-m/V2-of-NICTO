from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GANConfig:
    latent_dim: int = 100
    image_channels: int = 3
    image_size: int = 64
    num_epochs: int = 50
    batch_size: int = 64
    lr: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999
    loss_type: str = "bce"
    device: str = "cuda"
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints/gan"))
    sample_dir: Path = field(default_factory=lambda: Path("samples/gan"))
    lambda_gp: float = 10.0
    critic_iters: int = 5
