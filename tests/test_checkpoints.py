"""Tests for checkpoint registry and loader."""

from __future__ import annotations

import os
import tempfile

import pytest
import torch

from mom.models.checkpoints import CheckpointRegistry, CheckpointLoader
from mom.models.checkpoints.registry import CheckpointMetadata


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    return str(tmp_path)


def test_registry_initialization(temp_checkpoint_dir):
    registry = CheckpointRegistry(base_dir=temp_checkpoint_dir)
    assert registry.base_dir == temp_checkpoint_dir


def test_register_checkpoint(temp_checkpoint_dir):
    path = os.path.join(temp_checkpoint_dir, "model.pt")
    torch.save({"state_dict": {}}, path)
    registry = CheckpointRegistry(base_dir=temp_checkpoint_dir)
    meta = registry.register(
        model_id="test-model",
        path=path,
        architecture="transformer",
        task="language-modeling",
        metrics={"loss": 0.5},
    )
    assert meta.model_id == "test-model"
    assert meta.architecture == "transformer"


def test_get_checkpoint(temp_checkpoint_dir):
    path = os.path.join(temp_checkpoint_dir, "m1.pt")
    torch.save({"state_dict": {}}, path)
    registry = CheckpointRegistry(base_dir=temp_checkpoint_dir)
    registry.register("m1", path=path)
    meta = registry.get("m1")
    assert meta is not None
    assert meta.model_id == "m1"


def test_list_available(temp_checkpoint_dir):
    for i in range(3):
        path = os.path.join(temp_checkpoint_dir, f"m{i}.pt")
        torch.save({"state_dict": {}}, path)
    registry = CheckpointRegistry(base_dir=temp_checkpoint_dir)
    registry.register("m0", path=os.path.join(temp_checkpoint_dir, "m0.pt"))
    registry.register("m1", path=os.path.join(temp_checkpoint_dir, "m1.pt"))
    registry.register("m2", path=os.path.join(temp_checkpoint_dir, "m2.pt"))
    available = registry.list_available()
    assert len(available) >= 3


def test_loader_load(temp_checkpoint_dir):
    path = os.path.join(temp_checkpoint_dir, "ckpt.pt")
    data = {"state_dict": {"layer.weight": torch.randn(2, 2)}}
    torch.save(data, path)
    registry = CheckpointRegistry(base_dir=temp_checkpoint_dir)
    registry.register("ckpt", path=path)
    loader = CheckpointLoader(registry)
    loaded = loader.load(path)
    assert "state_dict" in loaded


def test_loader_missing_raises(temp_checkpoint_dir):
    loader = CheckpointLoader()
    with pytest.raises(FileNotFoundError):
        loader.load(os.path.join(temp_checkpoint_dir, "nonexistent.pt"))


def test_loader_verify(temp_checkpoint_dir):
    path = os.path.join(temp_checkpoint_dir, "verify.pt")
    torch.save({"state_dict": {}}, path)
    registry = CheckpointRegistry(base_dir=temp_checkpoint_dir)
    registry.register("verify", path=path)
    loader = CheckpointLoader(registry)
    assert loader.verify(path) is True


def test_checkpoint_metadata_to_dict(temp_checkpoint_dir):
    path = os.path.join(temp_checkpoint_dir, "md.pt")
    torch.save({}, path)
    registry = CheckpointRegistry(base_dir=temp_checkpoint_dir)
    meta = registry.register(
        model_id="md",
        path=path,
        version="v2.0",
        metrics={"accuracy": 0.95},
    )
    d = meta.to_dict()
    assert d["model_id"] == "md"
    assert d["version"] == "v2.0"
    assert d["metrics"]["accuracy"] == 0.95
