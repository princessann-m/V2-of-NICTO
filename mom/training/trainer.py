"""Base trainer with distributed training, AMP, checkpointing, and logging."""

from __future__ import annotations

import os
import time
import random
import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterator

import torch

from .configs import TrainingConfig

logger = logging.getLogger(__name__)


class BaseTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")
        self._setup()

    def _setup(self):
        torch.manual_seed(self.config.seed)
        random.seed(self.config.seed)
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        if self.config.distributed and torch.cuda.is_available() and torch.cuda.device_count() > 1:
            torch.distributed.init_process_group(backend="nccl")
            self.device = torch.device("cuda", torch.distributed.get_rank())
            torch.cuda.set_device(self.device)

    def train(
        self,
        model: torch.nn.Module,
        train_loader: Iterator,
        val_loader: Iterator | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        loss_fn: Callable | None = None,
    ):
        model = model.to(self.device)
        optimizer = optimizer or torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        loss_fn = loss_fn or torch.nn.CrossEntropyLoss()

        scaler = torch.cuda.amp.GradScaler() if self.config.mixed_precision and self.device.type == "cuda" else None

        if self.config.resume_from and os.path.exists(self.config.resume_from):
            self._load_checkpoint(model, optimizer)

        model.train()
        num_batches = 0
        for epoch in range(self.config.epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            num_batches = 0
            start = time.time()
            optimizer.zero_grad()
            for step, batch in enumerate(train_loader):
                loss = self._train_step(model, batch, optimizer, scaler, loss_fn)
                epoch_loss += loss
                num_batches += 1
                self.global_step += 1

                if self.global_step % self.config.log_interval == 0:
                    logger.info(
                        "epoch=%d step=%d loss=%.4f",
                        epoch, self.global_step, loss,
                    )

                if self.global_step % self.config.save_interval == 0:
                    self._save_checkpoint(model, optimizer)

                if val_loader and self.global_step % self.config.val_interval == 0:
                    self._validate(model, val_loader, loss_fn)

            avg = epoch_loss / max(1, num_batches)
            logger.info("epoch=%d avg_loss=%.4f time=%.2fs", epoch, avg, time.time() - start)

        self._save_checkpoint(model, optimizer)

    def _train_step(
        self,
        model: torch.nn.Module,
        batch: Any,
        optimizer: torch.optim.Optimizer,
        scaler: Any,
        loss_fn: Callable,
    ) -> float:
        inputs = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(**inputs)
            logits = outputs.get("logits", outputs)
            labels = inputs.get("labels")
            loss = loss_fn(logits, labels) if labels is not None else logits.mean()

        loss = loss / self.config.gradient_accumulation_steps
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if self.global_step % self.config.gradient_accumulation_steps == 0:
            if scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        return loss.item() * self.config.gradient_accumulation_steps

    @torch.no_grad()
    def _validate(self, model: torch.nn.Module, val_loader: Iterator, loss_fn: Callable):
        model.eval()
        total, count = 0.0, 0
        for batch in val_loader:
            inputs = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
            outputs = model(**inputs)
            logits = outputs.get("logits", outputs)
            labels = inputs.get("labels")
            loss = loss_fn(logits, labels) if labels is not None else logits.mean()
            total += loss.item()
            count += 1
        avg = total / max(1, count)
        logger.info("validation step=%d loss=%.4f", self.global_step, avg)
        if avg < self.best_val_loss:
            self.best_val_loss = avg
            self._save_checkpoint(model, None, name="best")
        model.train()

    def _save_checkpoint(self, model: torch.nn.Module, optimizer: Any, name: str | None = None):
        path = os.path.join(self.config.checkpoint_dir, name or f"checkpoint-{self.global_step}.pt")
        torch.save(
            {
                "epoch": self.epoch,
                "global_step": self.global_step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict() if optimizer else None,
                "config": self.config,
                "best_val_loss": self.best_val_loss,
            },
            path,
        )
        logger.info("saved checkpoint to %s", path)

    def _load_checkpoint(self, model: torch.nn.Module, optimizer: Any):
        ckpt = torch.load(self.config.resume_from, map_location=self.device)
        model.load_state_dict(ckpt["model_state"])
        if optimizer and ckpt.get("optimizer_state"):
            optimizer.load_state_dict(ckpt["optimizer_state"])
        self.epoch = ckpt.get("epoch", 0)
        self.global_step = ckpt.get("global_step", 0)
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        logger.info("resumed from %s at step=%d", self.config.resume_from, self.global_step)
