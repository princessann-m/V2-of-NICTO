"""Fallback package for graceful degradation in MoM."""

from .handlers import FallbackHandlers
from .strategies import FallbackStrategies

__all__ = ["FallbackHandlers", "FallbackStrategies"]
