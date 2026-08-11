"""Tests for GAN components."""

from __future__ import annotations

import pytest
import torch

from mom.generation.gan.generator import Generator
from mom.generation.gan.discriminator import Discriminator
from mom.generation.gan.pipeline import GenerationPipeline
from mom.generation.gan.config import GANConfig
from mom.training.pipelines import get_pipeline


class TestGenerator:
    def test_generate_shape(self):
        g = Generator(latent_dim=32, image_channels=1, image_size=16)
        z = torch.randn(4, 32)
        out = g(z)
        assert out.shape == (4, 1, 16, 16)

    def test_generator_default(self):
        g = Generator()
        assert g.latent_dim == 100


class TestDiscriminator:
    def test_discriminate_shape(self):
        d = Discriminator(image_channels=1, image_size=16)
        x = torch.randn(4, 1, 16, 16)
        out = d(x)
        assert out.shape == (4, 1)

    def test_discriminator_default(self):
        d = Discriminator()
        assert d.image_channels == 3


class TestPipeline:
    def test_pipeline_untrained(self):
        pipeline = GenerationPipeline()
        result = pipeline.generate({})
        assert result["status"] == "UNTRAINED"
        assert result["images"] is None

    def test_pipeline_from_config(self):
        cfg = GANConfig(latent_dim=16, image_channels=1, image_size=8)
        pipeline = GenerationPipeline(config=cfg)
        result = pipeline.generate({"style": "abstract"})
        assert result["status"] == "UNTRAINED"


def test_gan_training_pipeline():
    fn = get_pipeline("gan")
    result = fn(model=None, train_loader=iter([]), val_loader=None)
    assert "model" in result
