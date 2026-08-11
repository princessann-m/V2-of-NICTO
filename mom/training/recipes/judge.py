"""Judge model recipe configuration."""

from __future__ import annotations

from mom.training.recipes import Recipe


class JudgeRecipe(Recipe):
    def __init__(self) -> None:
        super().__init__(
            name="judge",
            config={
                "model_type": "transformer",
                "vocab_size": 10000,
                "hidden_size": 256,
                "num_layers": 4,
                "num_heads": 4,
                "ff_size": 512,
                "dropout": 0.1,
                "max_seq_len": 512,
                "training": {
                    "epochs": 10,
                    "batch_size": 16,
                    "learning_rate": 3e-4,
                    "weight_decay": 0.01,
                    "warmup_steps": 50,
                    "gradient_accumulation_steps": 1,
                    "max_grad_norm": 1.0,
                    "mixed_precision": False,
                    "distributed": False,
                    "checkpoint_dir": "checkpoints/judge",
                    "log_interval": 10,
                    "save_interval": 500,
                    "val_interval": 500,
                },
                "optimization": {
                    "optimizer": "adamw",
                    "scheduler": "cosine",
                    "min_lr_ratio": 0.1,
                },
                "data": {
                    "dataset": "dummy",
                    "num_workers": 0,
                },
            },
        )
