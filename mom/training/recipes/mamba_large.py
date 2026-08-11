"""Large Mamba recipe configuration."""

from __future__ import annotations

from mom.training.recipes import Recipe


class MambaLargeRecipe(Recipe):
    def __init__(self) -> None:
        super().__init__(
            name="mamba_large",
            config={
                "model_type": "mamba",
                "d_model": 1536,
                "n_layers": 48,
                "d_state": 16,
                "d_conv": 4,
                "expand": 2,
                "vocab_size": 32000,
                "max_seq_len": 2048,
                "training": {
                    "epochs": 10,
                    "batch_size": 16,
                    "learning_rate": 5e-5,
                    "weight_decay": 0.01,
                    "warmup_steps": 500,
                    "gradient_accumulation_steps": 8,
                    "max_grad_norm": 1.0,
                    "mixed_precision": True,
                    "distributed": True,
                    "checkpoint_dir": "checkpoints/mamba_large",
                    "log_interval": 10,
                    "save_interval": 2000,
                    "val_interval": 2000,
                },
                "optimization": {
                    "optimizer": "adamw",
                    "scheduler": "cosine",
                    "min_lr_ratio": 0.1,
                },
                "data": {
                    "dataset": "dummy",
                    "num_workers": 4,
                },
            },
        )
