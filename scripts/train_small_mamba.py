"""Train the smallest Mamba model (7M params) on CPU as proof of concept."""

from __future__ import annotations

import os
import json
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from mom.models.mamba import MambaConfig, MambaModel
from mom.models.tokenizer import BPETokenizer
from mom.training.configs import TrainingConfig


class TextDataset(Dataset):
    """Simple text dataset for training."""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        print(f"Loading data from {data_path}...")
        with open(data_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i > 5000:  # Limit for CPU training
                    break
                sample = json.loads(line)
                text = sample.get("text", "")
                if text and len(text.strip()) > 5:
                    self.samples.append(text)
        
        print(f"Loaded {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        text = self.samples[idx]
        input_ids = self.tokenizer.encode(text, max_length=self.max_length)
        attention_mask = [1 if x != 0 else 0 for x in input_ids]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(input_ids, dtype=torch.long),
        }


def collate_fn(batch):
    """Collate function for DataLoader."""
    input_ids = torch.stack([x["input_ids"] for x in batch])
    attention_mask = torch.stack([x["attention_mask"] for x in batch])
    labels = torch.stack([x["labels"] for x in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def train_epoch(model, dataloader, optimizer, device, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward pass
        logits, loss = model(input_ids, labels=labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % config.log_interval == 0:
            avg_loss = total_loss / num_batches
            print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}, Avg: {avg_loss:.4f}")
    
    return total_loss / max(1, num_batches)


def evaluate(model, dataloader, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            logits, loss = model(input_ids, labels=labels)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / max(1, num_batches)


def main():
    parser = argparse.ArgumentParser(description="Train small Mamba model on CPU")
    parser.add_argument("--output_dir", type=str, default="checkpoints/mamba_xsmall_foundation")
    parser.add_argument("--train_data", type=str, default="data/foundation/train.jsonl")
    parser.add_argument("--val_data", type=str, default="data/foundation/val.jsonl")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("NICTO Foundation Model Training")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Output: {args.output_dir}")
    
    # Config
    config = MambaConfig(
        vocab_size=1000,
        d_model=32,
        n_layers=2,
        d_state=8,
        d_conv=2,
        expand=2,
    )
    
    # Training config
    train_cfg = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        mixed_precision=False,  # CPU doesn't support AMP
        max_grad_norm=1.0,
        log_interval=10,
    )
    
    device = torch.device(args.device)
    
    # Tokenizer
    print("\nTraining tokenizer...")
    tokenizer = BPETokenizer(vocab_size=1000, max_length=args.max_length)
    
    # Load some data for tokenizer training
    tokenizer_texts = []
    with open(args.train_data, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 1000:
                break
            sample = json.loads(line)
            tokenizer_texts.append(sample.get("text", ""))
    
    tokenizer.train("\n".join(tokenizer_texts))
    print(f"Tokenizer trained: vocab_size={tokenizer.vocab_size}")
    
    # Datasets
    print("\nPreparing datasets...")
    train_dataset = TextDataset(args.train_data, tokenizer, args.max_length)
    val_dataset = TextDataset(args.val_data, tokenizer, args.max_length)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # CPU only
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    # Model
    print(f"\nInitializing model with {sum(p.numel() for p in MambaModel(config).parameters()):,} parameters")
    model = MambaModel(config).to(device)
    print(f"Model initialized: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=0.01,
    )
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    best_val_loss = float("inf")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)
        
        train_loss = train_epoch(model, train_loader, optimizer, device, train_cfg)
        print(f"  Train Loss: {train_loss:.4f}")
        
        val_loss = evaluate(model, val_loader, device)
        print(f"  Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": val_loss,
                "config": config.__dict__,
            }, checkpoint_path)
            print(f"  Saved best model to {checkpoint_path}")
        
        # Save tokenizer
        tokenizer_path = os.path.join(args.output_dir, "tokenizer.json")
        tokenizer.save(tokenizer_path)
    
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Training complete in {total_time:.2f}s")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {args.output_dir}")
    print("=" * 60)
    
    # Save final config
    with open(os.path.join(args.output_dir, "training_info.json"), "w") as f:
        json.dump({
            "final_train_loss": train_loss,
            "final_val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "total_time_seconds": total_time,
            "parameters": sum(p.numel() for p in model.parameters()),
            "config": config.__dict__,
            "training_config": train_cfg.__dict__,
        }, f, indent=2)
    
    return best_val_loss


if __name__ == "__main__":
    main()
