from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class VideoPipelineConfig:
    story_breakdown_steps: int = 3
    scene_count: int = 5
    segment_duration_seconds: float = 4.0
    fps: int = 24
    resolution: tuple[int, int] = (512, 512)
    output_format: str = "mp4"
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints/video"))
    sample_dir: Path = field(default_factory=lambda: Path("samples/video"))
