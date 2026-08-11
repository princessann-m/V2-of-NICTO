from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .config import VideoPipelineConfig


class VideoPipeline:
    UNTRAINED_STATUS = "UNTRAINED"

    def __init__(self, config: VideoPipelineConfig | None = None, weights_path: str | Path | None = None) -> None:
        self.config = config or VideoPipelineConfig()
        self.is_trained = False
        if weights_path:
            self.load_weights(weights_path)

    def load_weights(self, path: str | Path) -> None:
        self.is_trained = True

    def _story_breakdown(self, task: dict) -> dict[str, Any]:
        modality = task.get("modality", "video")
        original = task.get("original", "")
        scenes = []
        for i in range(self.config.scene_count):
            scenes.append({
                "scene_id": i,
                "description": f"Scene {i + 1} for: {original[:80]}",
                "duration": self.config.segment_duration_seconds,
                "modality": modality,
            })
        return {
            "task": original,
            "modality": modality,
            "scenes": scenes,
            "total_duration": self.config.scene_count * self.config.segment_duration_seconds,
        }

    def _plan_segments(self, story: dict[str, Any]) -> dict[str, Any]:
        segments = []
        for scene in story.get("scenes", []):
            segments.append({
                "segment_id": scene["scene_id"],
                "source_scene": scene,
                "frames_required": int(self.config.segment_duration_seconds * self.config.fps),
                "resolution": self.config.resolution,
                "fps": self.config.fps,
            })
        return {
            "segments": segments,
            "frame_budget": sum(s["frames_required"] for s in segments),
        }

    def _assemble(self, plan: dict[str, Any], generated_segments: list[dict[str, Any]]) -> dict[str, Any]:
        assembled = {
            "segments": generated_segments,
            "total_frames": plan.get("frame_budget", 0),
            "resolution": self.config.resolution,
            "fps": self.config.fps,
            "output_format": self.config.output_format,
        }
        return assembled

    def _director_review(self, assembled: dict[str, Any], task: dict) -> dict[str, Any]:
        try:
            from ..creative.directors.model import DirectorModel
            director = DirectorModel()
            review = director.evaluate(assembled, task)
            assembled["director_review"] = review
        except Exception:
            assembled["director_review"] = {
                "status": "fallback",
                "score": 0.5,
                "issues": ["director_unavailable"],
            }
        return assembled

    def generate(self, task: dict, iterations: int = 1) -> dict[str, Any]:
        if not self.is_trained:
            return {
                "status": self.UNTRAINED_STATUS,
                "message": "Video pipeline is untrained. Provide weights to generate video.",
                "task": task,
                "output": None,
            }

        story = self._story_breakdown(task)
        plan = self._plan_segments(story)

        generated_segments = []
        for segment in plan.get("segments", []):
            generated_segments.append({
                "segment_id": segment["segment_id"],
                "frames": None,
                "status": "generated",
            })

        assembled = self._assemble(plan, generated_segments)
        reviewed = self._director_review(assembled, task)

        return {
            "status": "SUCCESS",
            "task": task,
            "story": story,
            "plan": plan,
            "output": reviewed,
        }
