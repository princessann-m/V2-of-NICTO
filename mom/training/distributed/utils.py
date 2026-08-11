"""Distributed training utilities."""

from __future__ import annotations

import os
import logging

logger = logging.getLogger(__name__)


def setup_distributed(backend: str = "nccl") -> int:
    import torch

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend=backend)
    return torch.distributed.get_rank()


def cleanup_distributed() -> None:
    import torch

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def get_world_size() -> int:
    import torch

    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def get_rank() -> int:
    import torch

    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def is_main_process() -> bool:
    return get_rank() == 0


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def set_device_from_env() -> None:
    import torch

    local_rank = get_local_rank()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
