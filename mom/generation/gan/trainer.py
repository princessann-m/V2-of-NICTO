from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Adam

from .config import GANConfig
from .discriminator import Discriminator
from .generator import Generator
from .losses import bce_loss, lsgan_loss, wgan_gp_loss


class GANTrainer:
    def __init__(self, config: GANConfig) -> None:
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.generator = Generator(config.latent_dim, config.image_channels, config.image_size).to(self.device)
        self.discriminator = Discriminator(config.image_channels, config.image_size).to(self.device)
        self.optimizer_g = Adam(self.generator.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
        self.optimizer_d = Adam(self.discriminator.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
        self.current_epoch = 0
        self.global_step = 0
        self.metrics: list[dict[str, Any]] = []
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        config.sample_dir.mkdir(parents=True, exist_ok=True)

    def _loss_fn(self, real: torch.Tensor, fake: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        loss_type = self.config.loss_type.lower()
        if loss_type == "bce":
            d_real_logits = self.discriminator(real)
            d_fake_logits = self.discriminator(fake.detach())
            d_loss, g_loss = bce_loss(d_real_logits, d_fake_logits)
        elif loss_type == "lsgan":
            d_real_logits = self.discriminator(real)
            d_fake_logits = self.discriminator(fake.detach())
            d_loss, g_loss = lsgan_loss(d_real_logits, d_fake_logits)
        elif loss_type == "wgan":
            d_loss, g_loss = wgan_gp_loss(self.discriminator, real, fake, self.device, self.config.lambda_gp)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        return d_loss, g_loss

    def train_step(self, real_batch: torch.Tensor) -> dict[str, float]:
        real = real_batch.to(self.device)
        batch_size = real.size(0)
        z = torch.randn(batch_size, self.config.latent_dim, device=self.device)
        fake = self.generator(z)

        self.discriminator.zero_grad()
        d_loss, _ = self._loss_fn(real, fake)
        d_loss.backward()
        self.optimizer_d.step()

        if self.config.loss_type.lower() == "wgan" and self.global_step % self.config.critic_iters == 0:
            self.generator.zero_grad()
            z = torch.randn(batch_size, self.config.latent_dim, device=self.device)
            fake = self.generator(z)
            _, g_loss = self._loss_fn(real, fake)
            g_loss.backward()
            self.optimizer_g.step()
        else:
            self.generator.zero_grad()
            z = torch.randn(batch_size, self.config.latent_dim, device=self.device)
            fake = self.generator(z)
            _, g_loss = self._loss_fn(real, fake)
            g_loss.backward()
            self.optimizer_g.step()

        self.global_step += 1
        return {"d_loss": float(d_loss.item()), "g_loss": float(g_loss.item())}

    def fit(self, dataloader: torch.utils.data.DataLoader) -> None:
        self.generator.train()
        self.discriminator.train()
        for epoch in range(self.current_epoch, self.config.num_epochs):
            epoch_metrics = {"d_loss": 0.0, "g_loss": 0.0, "steps": 0}
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    real = batch[0]
                else:
                    real = batch
                metrics = self.train_step(real)
                epoch_metrics["d_loss"] += metrics["d_loss"]
                epoch_metrics["g_loss"] += metrics["g_loss"]
                epoch_metrics["steps"] += 1
            epoch_metrics["d_loss"] /= max(1, epoch_metrics["steps"])
            epoch_metrics["g_loss"] /= max(1, epoch_metrics["steps"])
            self.metrics.append(epoch_metrics)
            self.current_epoch = epoch + 1
            self.save_checkpoint()

    def evaluate(self, num_samples: int = 16) -> dict[str, Any]:
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.config.latent_dim, device=self.device)
            fake = self.generator(z)
            fake = (fake + 1) / 2
        fid_placeholder = None
        return {
            "generated_images": fake.cpu(),
            "fid": fid_placeholder,
            "status": "placeholder_evaluation",
        }

    def save_checkpoint(self) -> None:
        path = self.config.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pt"
        torch.save(
            {
                "generator_state_dict": self.generator.state_dict(),
                "discriminator_state_dict": self.discriminator.state_dict(),
                "optimizer_g_state_dict": self.optimizer_g.state_dict(),
                "optimizer_d_state_dict": self.optimizer_d.state_dict(),
                "epoch": self.current_epoch,
                "global_step": self.global_step,
                "metrics": self.metrics,
                "config": self.config,
            },
            path,
        )

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        self.optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
        self.optimizer_d.load_state_dict(checkpoint["optimizer_d_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint.get("global_step", 0)
        self.metrics = checkpoint.get("metrics", [])
