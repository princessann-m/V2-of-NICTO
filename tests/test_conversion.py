"""Tests for model conversion utilities."""

from __future__ import annotations

import os
import json
import tempfile

import pytest
import torch
import torch.nn as nn

from mom.models.conversion import HFConverter


class DummyHFModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 2)

    def forward(self, x):
        return self.layer(x)


def test_hf_converter_init():
    conv = HFConverter(hf_model_id="dummy/model")
    assert conv.hf_model_id == "dummy/model"


def test_convert_from_hf():
    conv = HFConverter()
    model = DummyHFModel()
    config = {"hidden_size": 4, "num_layers": 1}
    out = conv.convert_from_hf(model, config)
    assert "state_dict" in out
    assert "layer.weight" in out["state_dict"]


def test_convert_to_hf():
    conv = HFConverter()
    state = {"module.layers.0.weight": torch.randn(2, 4)}
    hf_config = {"hidden_size": 4}
    out = conv.convert_to_hf(state, hf_config)
    assert "state_dict" in out
    assert "layers.0.weight" in out["state_dict"]


def test_map_config():
    conv = HFConverter()
    hf = {"hidden_size": 768, "num_hidden_layers": 12, "vocab_size": 32000}
    mapped = conv.map_config(hf)
    assert mapped["d_model"] == 768
    assert mapped["n_layers"] == 12
    assert mapped["vocab_size"] == 32000


def test_save_and_load_hf_format(tmp_path):
    conv = HFConverter()
    state = {"layer.weight": torch.randn(2, 4)}
    config = {"hidden_size": 4}
    out_dir = str(tmp_path / "hf_model")
    conv.save_hf_format(state, config, out_dir)
    assert os.path.exists(os.path.join(out_dir, "pytorch_model.bin"))
    assert os.path.exists(os.path.join(out_dir, "config.json"))
    loaded_state, loaded_config = conv.load_hf_format(out_dir)
    assert "layer.weight" in loaded_state
    assert loaded_config["hidden_size"] == 4


def test_map_key():
    conv = HFConverter()
    assert conv._map_key("module.layer.weight") == "layer.weight"
    assert conv._reverse_map_key("module.layer.weight") == "layer.weight"
