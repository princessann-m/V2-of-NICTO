"""Tests for quantization and ONNX export."""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
import torch.nn as nn

from mom.models.quantization import ModelQuantizer, ONNXExporter
from mom.models.quantization.quantizer import QuantizationResult


class SimpleModel(nn.Module):
    def __init__(self, in_features: int = 16, out_features: int = 8):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def simple_model():
    return SimpleModel()


class TestModelQuantizer:
    def test_initialization(self, simple_model):
        quantizer = ModelQuantizer(simple_model)
        assert quantizer.model is not None

    def test_original_size_computed(self, simple_model):
        quantizer = ModelQuantizer(simple_model)
        assert quantizer._original_size > 0

    def test_quantize_to_fp16_returns_result(self, simple_model):
        quantizer = ModelQuantizer(simple_model)
        result = quantizer.quantize_to_fp16()
        assert isinstance(result, QuantizationResult)
        assert result.original_size_mb > 0

    def test_quantize_to_int8_returns_result(self, simple_model):
        quantizer = ModelQuantizer(simple_model)
        result = quantizer.quantize_to_int8()
        assert isinstance(result, QuantizationResult)

    def test_quantize_to_gguf_returns_result(self, simple_model):
        quantizer = ModelQuantizer(simple_model)
        result = quantizer.quantize_to_gguf()
        assert isinstance(result, QuantizationResult)

    def test_estimate_memory(self, simple_model):
        quantizer = ModelQuantizer(simple_model)
        mem = quantizer.estimate_memory(batch_size=1, seq_length=64)
        assert "parameters_mb" in mem
        assert "total_mb" in mem
        assert mem["total_mb"] > 0

    def test_estimate_latency(self, simple_model):
        quantizer = ModelQuantizer(simple_model)
        latency = quantizer.estimate_latency(batch_size=1, seq_length=64)
        assert latency > 0


class TestONNXExporter:
    def test_initialization(self, simple_model):
        exporter = ONNXExporter(simple_model)
        assert exporter.input_names == ["input_ids"]
        assert exporter.output_names == ["logits"]

    def test_get_input_spec(self, simple_model):
        exporter = ONNXExporter(simple_model)
        spec = exporter.get_input_spec()
        assert len(spec) == 1
        assert spec[0]["name"] == "input_ids"

    def test_get_output_spec(self, simple_model):
        exporter = ONNXExporter(simple_model)
        spec = exporter.get_output_spec()
        assert len(spec) == 1

    def test_create_optimization_profile(self, simple_model):
        exporter = ONNXExporter(simple_model)
        profile = exporter.create_optimization_profile(
            min_shape=(1, 32),
            opt_shape=(1, 128),
            max_shape=(1, 512),
        )
        assert "input_shapes" in profile

    def test_export_to_onnx(self, simple_model):
        exporter = ONNXExporter(simple_model, input_shape=(1, 16))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.onnx")
            try:
                result = exporter.export_to_onnx(path)
                assert os.path.exists(result)
            except ImportError:
                pytest.skip("onnxscript not installed")
