"""Expert registry and base for MoM."""

from __future__ import annotations

from .math_expert import MathExpert
from .coding_expert import CodingExpert
from .reasoning_expert import ReasoningExpert
from .science_expert import ScienceExpert
from .vision_expert import VisionExpert


EXPERT_MAP = {
    "math_expert": MathExpert,
    "coding_expert": CodingExpert,
    "reasoning_expert": ReasoningExpert,
    "science_expert": ScienceExpert,
    "vision_expert": VisionExpert,
}


def build_expert(meta: dict, llm=None):
    cls = EXPERT_MAP.get(meta.get("name"))
    if not cls:
        from .expert_base import ExpertBase
        return ExpertBase(meta)
    return cls(meta, llm=llm)
