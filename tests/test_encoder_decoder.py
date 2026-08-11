"""Tests for encoder-decoder components."""

from __future__ import annotations

import pytest
import torch

from mom.training.pipelines import get_pipeline


class DummyEncoderDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(8, 4)
        self.decoder = torch.nn.Linear(4, 8)

    def forward(self, x=None, labels=None, **kwargs):
        if x is None:
            x = torch.randn(2, 8)
        z = self.encoder(x)
        out = self.decoder(z)
        loss = torch.nn.functional.mse_loss(out, x)
        return {"logits": out, "loss": loss}


class DummyDataset:
    def __iter__(self):
        for _ in range(4):
            yield {"x": torch.randn(2, 8), "labels": torch.zeros(2, 8)}


def test_encoder_decoder_pipeline_exists():
    fn = get_pipeline("encoder_decoder")
    assert callable(fn)


def test_encoder_decoder_forward():
    model = DummyEncoderDecoder()
    out = model(x=torch.randn(2, 8))
    assert "logits" in out
    assert "loss" in out
    assert out["logits"].shape == (2, 8)


def test_encoder_decoder_pipeline_run():
    fn = get_pipeline("encoder_decoder")
    result = fn(model=None, train_loader=iter([]), val_loader=None)
    assert "model" in result
