"""Training recipes for MoM models."""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class Recipe:
    name: str
    config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str) -> "Recipe":
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        name = data.pop("name", os.path.splitext(os.path.basename(path))[0])
        return cls(name=name, config=data)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.safe_dump({"name": self.name, **self.config}, f)
