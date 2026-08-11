"""Tests for training infrastructure."""

from __future__ import annotations

import pytest
import torch

from mom.training.configs import TrainingConfig
from mom.training.trainer import BaseTrainer
from mom.training.pipelines import get_pipeline


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, x=None, labels=None, **kwargs):
        if x is None:
            x = torch.randn(2, 4)
        logits = self.linear(x)
        loss = torch.nn.functional.cross_entropy(logits, labels if labels is not None else torch.zeros(2, dtype=torch.long))
        return {"logits": logits, "loss": loss}


class DummyDataset:
    def __init__(self, n=8):
        self.n = n

    def __iter__(self):
        for _ in range(self.n):
            yield {"x": torch.randn(2, 4), "labels": torch.tensor([0, 1])}


def test_training_config_defaults():
    cfg = TrainingConfig()
    assert cfg.epochs == 3
    assert cfg.batch_size == 8
    assert cfg.mixed_precision is True


def test_training_config_custom():
    cfg = TrainingConfig(epochs=10, batch_size=4, device="cpu")
    assert cfg.epochs == 10
    assert cfg.batch_size == 4
    assert cfg.device == "cpu"


def test_base_trainer_train_cpu():
    cfg = TrainingConfig(epochs=1, batch_size=2, device="cpu", mixed_precision=False)
    trainer = BaseTrainer(cfg)
    model = DummyModel()
    loader = DummyDataset(n=4)
    train_loader = iter(loader)
    trainer.train(model, train_loader)


def test_base_trainer_save_and_resume(tmp_path):
    cfg = TrainingConfig(
        epochs=1,
        batch_size=2,
        device="cpu",
        mixed_precision=False,
        checkpoint_dir=str(tmp_path),
        save_interval=2,
    )
    trainer = BaseTrainer(cfg)
    model = DummyModel()
    loader = DummyDataset(n=4)
    train_loader = iter(loader)
    trainer.train(model, train_loader)
    files = list(tmp_path.iterdir())
    assert len(files) >= 1


def test_pipeline_lookup():
    fn = get_pipeline("mamba")
    assert callable(fn)
    with pytest.raises(ValueError):
        get_pipeline("unknown_pipeline")


def test_mamba_pipeline():
    fn = get_pipeline("mamba")
    result = fn(model=None, train_loader=iter([]), val_loader=None)
    assert "model" in result
    assert "loader" in result
