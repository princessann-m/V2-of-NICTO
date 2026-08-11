"""Accelerate-based distributed training launcher."""

from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AccelerateConfig:
    mixed_precision: str = "no"
    gradient_accumulation_steps: int = 1
    cpu: bool = False
    num_processes: int | None = None
    num_machines: int = 1
    machine_rank: int = 0
    main_process_ip: str = "127.0.0.1"
    main_process_port: int = 29500
    rdzv_backend: str = "static"
    same_network: bool = True
    fsdp: str | None = None
    fsdp_config: str | None = None
    deepspeed: str | None = None
    dynamo_backend: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mixed_precision": self.mixed_precision,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "cpu": self.cpu,
            "num_processes": self.num_processes,
            "num_machines": self.num_machines,
            "machine_rank": self.machine_rank,
            "main_process_ip": self.main_process_ip,
            "main_process_port": self.main_process_port,
            "rdzv_backend": self.rdzv_backend,
            "same_network": self.same_network,
            "fsdp": self.fsdp,
            "fsdp_config": self.fsdp_config,
            "deepspeed": self.deepspeed,
            "dynamo_backend": self.dynamo_backend,
        }


class AccelerateLauncher:
    def __init__(self, config: AccelerateConfig | None = None) -> None:
        self.config = config or AccelerateConfig()

    def generate_config(self, output_path: str) -> str:
        data = self.config.to_dict()
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Wrote accelerate config to %s", output_path)
        return output_path

    def launch(self, training_script: str, script_args: list[str] | None = None) -> int:
        config_path = self.generate_config(".accelerate_config.json")
        cmd = [
            "accelerate", "launch",
            "--config_file", config_path,
        ]
        if self.config.mixed_precision != "no":
            cmd += ["--mixed_precision", self.config.mixed_precision]
        if self.config.gradient_accumulation_steps > 1:
            cmd += ["--gradient_accumulation_steps", str(self.config.gradient_accumulation_steps)]
        if self.config.cpu:
            cmd.append("--cpu")
        if self.config.num_processes:
            cmd += ["--num_processes", str(self.config.num_processes)]
        if self.config.num_machines > 1:
            cmd += ["--num_machines", str(self.config.num_machines)]
            cmd += ["--machine_rank", str(self.config.machine_rank)]
            cmd += ["--main_process_ip", self.config.main_process_ip]
            cmd += ["--main_process_port", str(self.config.main_process_port)]
        if self.config.fsdp:
            cmd += ["--fsdp", self.config.fsdp]
        if self.config.deepspeed:
            cmd += ["--deepspeed", self.config.deepspeed]
        cmd.append(training_script)
        if script_args:
            cmd.extend(script_args)
        logger.info("Launching: %s", " ".join(cmd))
        try:
            import subprocess
            result = subprocess.run(cmd, check=False)
            return result.returncode
        except FileNotFoundError:
            logger.error("accelerate executable not found in PATH")
            return 127

    @staticmethod
    def supports_plugin(name: str) -> bool:
        try:
            import importlib
            importlib.import_module(f"accelerate.utils.imports.{name}")
            return True
        except ImportError:
            return False
