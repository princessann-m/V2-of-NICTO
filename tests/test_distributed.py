"""Tests for distributed training utilities."""

from __future__ import annotations

import os
import pytest

from mom.training.distributed import (
    TorchrunLauncher,
    TorchrunConfig,
    AccelerateLauncher,
    AccelerateConfig,
    get_world_size,
    get_rank,
    is_main_process,
    setup_distributed,
    cleanup_distributed,
)


def test_torchrun_config_defaults():
    cfg = TorchrunConfig()
    assert cfg.nnodes == 1
    assert cfg.master_port == 29500
    assert cfg.script_args == []


def test_torchrun_launcher_builds_command():
    launcher = TorchrunLauncher()
    cfg = TorchrunConfig(script="train.py", script_args=["--epochs", "1"])
    launcher.config = cfg
    assert launcher.config.cmd[0] == "torchrun"
    assert launcher.config.cmd[-1] == "1"
    assert launcher.config.cmd[-2] == "--epochs"


def test_torchrun_validate_hardware():
    launcher = TorchrunLauncher()
    info = launcher.validate_hardware()
    assert "torch_version" in info
    assert "cuda_available" in info


def test_accelerate_config_defaults():
    cfg = AccelerateConfig()
    assert cfg.mixed_precision == "no"
    assert cfg.num_machines == 1


def test_accelerate_config_custom():
    cfg = AccelerateConfig(mixed_precision="fp16", num_processes=4)
    assert cfg.mixed_precision == "fp16"
    assert cfg.num_processes == 4


def test_accelerate_launcher_generate_config(tmp_path):
    launcher = AccelerateLauncher()
    out = tmp_path / "acc_config.json"
    path = launcher.generate_config(str(out))
    assert os.path.exists(path)
    import json
    with open(path) as f:
        data = json.load(f)
    assert data["mixed_precision"] == "no"


def test_accelerate_launcher_launch_missing_tool():
    launcher = AccelerateLauncher()
    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = ""
    ret = launcher.launch("nonexistent_script.py")
    os.environ["PATH"] = old_path
    assert ret == 127


def test_world_size_default():
    assert get_world_size() == 1


def test_rank_default():
    assert get_rank() == 0


def test_is_main_process_default():
    assert is_main_process() is True


def test_local_rank_default():
    from mom.training.distributed.utils import get_local_rank
    assert get_local_rank() == 0
