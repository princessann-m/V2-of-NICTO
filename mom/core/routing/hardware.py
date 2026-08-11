"""Hardware detection and model placement strategy."""

from __future__ import annotations

import logging
import os
import psutil
from dataclasses import dataclass, field
from typing import Any

LOG = logging.getLogger(__name__)


@dataclass
class HardwareProfile:
    gpu_available: bool = False
    gpu_count: int = 0
    gpu_names: list[str] = field(default_factory=list)
    vram_total_gb: float = 0.0
    cpu_cores: int = 1
    cpu_memory_gb: float = 0.0
    recommended_quantization: str = "none"
    multi_gpu: bool = False
    placement_strategy: str = "cpu"


class HardwareDetector:
    def __init__(self) -> None:
        self.profile = self._detect()

    def _detect(self) -> HardwareProfile:
        profile = HardwareProfile()
        profile.cpu_cores = os.cpu_count() or 1
        mem = psutil.virtual_memory()
        profile.cpu_memory_gb = mem.total / (1024 ** 3)

        try:
            import torch
            if torch.cuda.is_available():
                profile.gpu_available = True
                profile.gpu_count = torch.cuda.device_count()
                profile.gpu_names = [torch.cuda.get_device_name(i) for i in range(profile.gpu_count)]
                total_vram = sum(torch.cuda.get_device_properties(i).total_memory for i in range(profile.gpu_count))
                profile.vram_total_gb = total_vram / (1024 ** 3)
                profile.multi_gpu = profile.gpu_count > 1
                profile.placement_strategy = "cuda"
                if profile.vram_total_gb < 16:
                    profile.recommended_quantization = "int8"
                elif profile.vram_total_gb < 32:
                    profile.recommended_quantization = "fp16"
                else:
                    profile.recommended_quantization = "bf16"
        except Exception:
            LOG.debug("torch not available or CUDA unavailable")

        if not profile.gpu_available and profile.cpu_memory_gb < 8:
            profile.recommended_quantization = "int4"
        elif not profile.gpu_available:
            profile.recommended_quantization = "int8"

        return profile

    def get_profile(self) -> HardwareProfile:
        return self.profile

    def can_fit_model(self, model_size_gb: float, quantized_size_gb: float | None = None) -> bool:
        if self.profile.gpu_available:
            return (quantized_size_gb or model_size_gb) <= self.profile.vram_total_gb * 0.9
        return (quantized_size_gb or model_size_gb) <= self.profile.cpu_memory_gb * 0.7

    def suggest_distribution(self, model_count: int) -> list[int]:
        if not self.profile.multi_gpu:
            return [0] * model_count
        return [i % self.profile.gpu_count for i in range(model_count)]
