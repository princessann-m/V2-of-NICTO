"""Tests for multimodal fusion."""

from __future__ import annotations

import numpy as np

import pytest

from mom.models.multimodal.config import MultimodalConfig
from mom.models.multimodal.fusion import MultimodalFusion


def test_fusion_config_defaults():
    cfg = MultimodalConfig()
    assert cfg.vision_dim == 512
    assert cfg.text_dim == 768
    assert cfg.fusion_dim == 512


def test_fuse_output_shape():
    fusion = MultimodalFusion()
    image = np.zeros((1, 3, 224, 224), dtype=np.float32)
    text = np.zeros((1, 10), dtype=np.float32)
    out = fusion.fuse(image, text)
    assert out.shape == (1, 512)


def test_fusion_marked_untrained():
    fusion = MultimodalFusion()
    assert fusion.trained is False


def test_cross_attention_returns_array():
    fusion = MultimodalFusion()
    v = np.zeros((2, 512), dtype=np.float32)
    t = np.zeros((2, 768), dtype=np.float32)
    out = fusion.cross_attention(v, t)
    assert isinstance(out, np.ndarray)
