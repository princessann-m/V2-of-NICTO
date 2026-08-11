from __future__ import annotations

import torch
import torch.nn as nn


def bce_loss(d_real_logits: torch.Tensor, d_fake_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    criterion = nn.BCEWithLogitsLoss()
    real_labels = torch.ones_like(d_real_logits)
    fake_labels = torch.zeros_like(d_fake_logits)
    d_loss = criterion(d_real_logits, real_labels) + criterion(d_fake_logits, fake_labels)
    g_loss = criterion(d_fake_logits, real_labels)
    return d_loss, g_loss


def lsgan_loss(d_real_logits: torch.Tensor, d_fake_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mse = nn.MSELoss()
    real_labels = torch.ones_like(d_real_logits)
    fake_labels = torch.zeros_like(d_fake_logits)
    d_loss = 0.5 * (mse(d_real_logits, real_labels) + mse(d_fake_logits, fake_labels))
    g_loss = 0.5 * mse(d_fake_logits, real_labels)
    return d_loss, g_loss


def compute_gradient_penalty(
    discriminator: nn.Module, real_samples: torch.Tensor, fake_samples: torch.Tensor, device: torch.device
) -> torch.Tensor:
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    alpha = alpha.expand_as(real_samples)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def wgan_gp_loss(
    discriminator: nn.Module, real_samples: torch.Tensor, fake_samples: torch.Tensor, device: torch.device, lambda_gp: float = 10.0
) -> tuple[torch.Tensor, torch.Tensor]:
    d_real = discriminator(real_samples)
    d_fake = discriminator(fake_samples.detach())
    d_loss = -torch.mean(d_real) + torch.mean(d_fake)
    gp = compute_gradient_penalty(discriminator, real_samples, fake_samples, device)
    d_loss = d_loss + lambda_gp * gp
    g_loss = -torch.mean(discriminator(fake_samples))
    return d_loss, g_loss
