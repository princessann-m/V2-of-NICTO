"""Medium Mamba recipe configuration."""

from __future__ import annotations

from mom.training.recipes import Recipe


class MambaMediumRecipe(Recipe):
    def __init__(self) -> None:
        super().__init__(
            name="mamba_medium",
            config={
                "model_type": "mamba",
                "d_model": 1024,
                "n_layers": 36,
                "d_state": 16,
                "d_conv": 4,
                "expand": 2,
                "vocab_size": 32000,
                "max_seq_len": 2048,
                "training": {
                    "epochs": 5,
                    "batch_size": 8,
                    "learning_rate": 8e-5,
                    "weight_decay": 0.01,
                    "warmup_steps": 200,
                    "gradient_accumulation_steps": 4,
                    "max_grad_norm": 1.0,
                    "mixed_precision": True,
                    "distributed": True,
                    "checkpoint_dir": "checkpoints/mamba_medium",
                    "log_interval": 10,
                    "save_interval": 1000,
                    "val_interval": 1000,
                },
                "optimization": {
                    "optimizer": "adamw",
                    "scheduler": "cosine",
                    "min_lr_ratio": 0.1,
                },
                "data": {
                    "dataset": "dummy",
                    "num_workers": 2,
                },
            },
        )
