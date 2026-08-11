from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .config import GANConfig
from .generator import Generator
from .trainer import GANTrainer


class GenerationPipeline:
    UNTRAINED_STATUS = "UNTRAINED"

    def __init__(self, config: GANConfig | None = None, weights_path: str | Path | None = None) -> None:
        self.config = config or GANConfig()
        self.trainer = GANTrainer(self.config)
        self.is_trained = False
        if weights_path:
            self.load_weights(weights_path)

    def load_weights(self, path: str | Path) -> None:
        self.trainer.load_checkpoint(path)
        self.is_trained = True

    def generate(self, specifications: dict[str, Any], num_samples: int = 4) -> dict[str, Any]:
        if not self.is_trained:
            return {
                "status": self.UNTRAINED_STATUS,
                "message": "Model is untrained. Outputs will be noise.",
                "images": None,
                "specifications": specifications,
            }
        result = self.trainer.evaluate(num_samples=num_samples)
        return {
            "status": "SUCCESS",
            "images": result["generated_images"],
            "fid": result["fid"],
            "specifications": specifications,
        }

    def train(self, dataloader: torch.utils.data.DataLoader) -> None:
        self.trainer.fit(dataloader)
        self.is_trained = True
