"""Torchrun-based distributed training launcher."""

from __future__ import annotations

import os
import shutil
import logging
import subprocess
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TorchrunConfig:
    nnodes: int = 1
    nproc_per_node: int | str = "auto"
    node_rank: int = 0
    master_addr: str = "localhost"
    master_port: int = 29500
    rdzv_backend: str = "c10d"
    rdzv_endpoint: str = ""
    max_restarts: int = 3
    env: dict[str, str] = field(default_factory=dict)
    script: str = ""
    script_args: list[str] = field(default_factory=list)

    @property
    def cmd(self) -> list[str]:
        parts = ["torchrun"]
        parts += ["--nnodes", str(self.nnodes)]
        parts += ["--node_rank", str(self.node_rank)]
        parts += ["--master_addr", self.master_addr]
        parts += ["--master_port", str(self.master_port)]
        parts += ["--rdzv_backend", self.rdzv_backend]
        if self.rdzv_endpoint:
            parts += ["--rdzv_endpoint", self.rdzv_endpoint]
        parts += ["--max_restarts", str(self.max_restarts)]
        if self.script:
            parts.append(self.script)
        parts.extend(self.script_args)
        return parts


class TorchrunLauncher:
    def __init__(self, config: TorchrunConfig | None = None) -> None:
        self.config = config or TorchrunConfig()

    def validate_hardware(self) -> dict[str, Any]:
        import torch

        info: dict[str, Any] = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "nccl_available": torch.distributed.is_nccl_available() if hasattr(torch.distributed, "is_nccl_available") else False,
            "gloo_available": torch.distributed.is_gloo_available() if hasattr(torch.distributed, "is_gloo_available") else False,
            "mpi_available": torch.distributed.is_mpi_available() if hasattr(torch.distributed, "is_mpi_available") else False,
        }
        if info["cuda_available"]:
            for i in range(info["cuda_device_count"]):
                info[f"cuda:{i}_name"] = torch.cuda.get_device_name(i)
        return info

    def launch(self) -> int:
        self._configure_env()
        cmd = self.config.cmd
        logger.info("Launching: %s", " ".join(cmd))
        try:
            result = subprocess.run(cmd, check=False)
            return result.returncode
        except FileNotFoundError:
            logger.error("torchrun executable not found in PATH")
            return 127

    def _configure_env(self) -> None:
        env = self.config.env
        if env:
            os.environ.update(env)
        if self.config.nproc_per_node == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(torch.cuda.device_count()))
            except Exception:
                pass
        if self.config.rdzv_endpoint:
            os.environ.setdefault("MASTER_ADDR", self.config.master_addr)
            os.environ.setdefault("MASTER_PORT", str(self.config.master_port))
