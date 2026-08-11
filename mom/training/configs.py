"""Training configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainingConfig:
    model_name: str = "nicto"
    epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    mixed_precision: bool = True
    distributed: bool = False
    checkpoint_dir: str = "checkpoints"
    resume_from: str | None = None
    log_interval: int = 10
    save_interval: int = 500
    val_interval: int = 500
    device: str = "cpu"
    seed: int = 42
    extra: dict[str, Any] = field(default_factory=dict)
