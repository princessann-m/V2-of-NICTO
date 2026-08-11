"""GAN training recipe configuration."""

from __future__ import annotations

from mom.training.recipes import Recipe


class GanRecipe(Recipe):
    def __init__(self) -> None:
        super().__init__(
            name="gan",
            config={
                "model_type": "gan",
                "generator": {
                    "latent_dim": 100,
                    "hidden_size": 256,
                    "num_layers": 3,
                    "output_channels": 3,
                },
                "discriminator": {
                    "hidden_size": 256,
                    "num_layers": 3,
                    "input_channels": 3,
                },
                "training": {
                    "epochs": 100,
                    "batch_size": 32,
                    "learning_rate_g": 2e-4,
                    "learning_rate_d": 2e-4,
                    "weight_decay": 0.0,
                    "warmup_steps": 0,
                    "gradient_accumulation_steps": 1,
                    "max_grad_norm": 1.0,
                    "mixed_precision": True,
                    "distributed": False,
                    "checkpoint_dir": "checkpoints/gan",
                    "log_interval": 10,
                    "save_interval": 1000,
                    "val_interval": 1000,
                },
                "optimization": {
                    "optimizer": "adam",
                    "beta1": 0.5,
                    "beta2": 0.999,
                },
                "data": {
                    "dataset": "dummy",
                    "num_workers": 0,
                    "image_size": 64,
                },
            },
        )
