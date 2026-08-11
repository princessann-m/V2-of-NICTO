"""Mixture of Models (MoM) prototype package."""

from __future__ import annotations

__version__ = "0.1.0"

from .config import MoMConfig, load_config

_LAZY_IMPORTS: dict[str, list[str]] = {
    "Orchestrator": [".core.orchestrator", "Orchestrator"],
    "VideoPipeline": [".generation.video.pipeline", "VideoPipeline"],
    "VideoPipelineConfig": [".generation.video.config", "VideoPipelineConfig"],
    "ImagePipeline": [".generation.image.pipeline", "ImagePipeline"],
    "DirectorModel": [".creative.directors.model", "DirectorModel"],
    "LightingExpert": [".creative.experts.lighting", "LightingExpert"],
    "CompositionExpert": [".creative.experts.composition", "CompositionExpert"],
    "TextureExpert": [".creative.experts.texture", "TextureExpert"],
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path, __name__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MoMConfig",
    "load_config",
    "Orchestrator",
    "VideoPipeline",
    "VideoPipelineConfig",
    "ImagePipeline",
    "DirectorModel",
    "LightingExpert",
    "CompositionExpert",
    "TextureExpert",
]
