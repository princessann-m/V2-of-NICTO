from __future__ import annotations

from typing import Any

from ..gan.pipeline import GenerationPipeline
from ..gan.config import GANConfig


class ImagePipeline:
    def __init__(self, config: GANConfig | None = None, weights_path: str | None = None) -> None:
        self.gan_pipeline = GenerationPipeline(config=config, weights_path=weights_path)

    def generate(self, specifications: dict[str, Any], num_samples: int = 4, iterations: int = 1) -> dict[str, Any]:
        result = self.gan_pipeline.generate(specifications, num_samples=num_samples)
        output = {
            "specifications": specifications,
            "modality": "image",
            "candidates": [],
        }

        if result.get("status") == "UNTRAINED":
            output["status"] = "UNTRAINED"
            output["message"] = result.get("message", "Image pipeline is untrained.")
            output["images"] = None
            output["iterations"] = iterations
            return output

        output["status"] = "SUCCESS"
        output["images"] = result.get("images")
        output["fid"] = result.get("fid")
        output["iterations"] = iterations
        output["candidates"].append(result)
        return output
