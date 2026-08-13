"""Train ~10M parameter Mamba model on CPU as milestone proof of concept."""

from __future__ import annotations

import os
import json
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from mom.models.mamba import MambaConfig, MambaModel
from mom.models.tokenizer import BPETokenizer


class TextDataset(Dataset):
    """Minimal text dataset for 10M model training."""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 32):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        with open(data_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i > 500:
                    break
                sample = json.loads(line)
                text = sample.get("text", "")
                if text and len(text.strip()) > 5:
                    self.samples.append(text)
        
        print(f"Loaded {len(self.samples)} samples for 10M model")
    
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
    input_ids = torch.stack([x["input_ids"] for x in batch])
    attention_mask = torch.stack([x["attention_mask"] for x in batch])
    labels = torch.stack([x["labels"] for x in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def main():
    parser = argparse.ArgumentParser(description="Train ~10M Mamba model on CPU")
    parser.add_argument("--output_dir", type=str, default="checkpoints/mamba_10m_milestone")
    parser.add_argument("--train_data", type=str, default="data/foundation/train.jsonl")
    parser.add_argument("--val_data", type=str, default="data/foundation/val.jsonl")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("NICTO 10M Parameter Model Training")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Output: {args.output_dir}")
    
    # ~10M parameter config
    config = MambaConfig(
        vocab_size=1000,
        d_model=256,
        n_layers=12,
        d_state=16,
        d_conv=4,
        expand=4,
    )
    
    model = MambaModel(config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nTarget model size: {param_count:,} parameters ({param_count/1e6:.2f}M)")
    
    if param_count < 9e6 or param_count > 11e6:
        print(f"WARNING: Model size {param_count/1e6:.2f}M is outside target 10M range")
    
    device = torch.device(args.device)
    model = model.to(device)
    
    # Tokenizer
    print("\nTraining tokenizer...")
    tokenizer = BPETokenizer(vocab_size=1000, max_length=args.max_length)
    
    tokenizer_texts = []
    with open(args.train_data, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 200:
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
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting 10M model training...")
    print("=" * 60)
    
    best_val_loss = float("inf")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)
        
        model.train()
        total_train_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            logits, loss = model(input_ids, labels=labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            total_train_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 5 == 0:
                avg_loss = total_train_loss / max(1, num_batches)
                print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, Avg: {avg_loss:.4f}")
        
        train_loss = total_train_loss / max(1, num_batches)
        print(f"  Train Loss: {train_loss:.4f}")
        
        # Validation
        model.eval()
        total_val_loss = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                logits, loss = model(input_ids, labels=labels)
                total_val_loss += loss.item()
                num_val_batches += 1
        
        val_loss = total_val_loss / max(1, num_val_batches)
        print(f"  Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": val_loss,
                "config": config.__dict__,
                "param_count": param_count,
            }, checkpoint_path)
            print(f"  Saved best model to {checkpoint_path}")
        
        # Save tokenizer
        tokenizer.save(os.path.join(args.output_dir, "tokenizer.json"))
    
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"10M model training complete in {total_time:.2f}s")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Parameters: {param_count:,} ({param_count/1e6:.2f}M)")
    print(f"Model saved to: {args.output_dir}")
    print("=" * 60)
    
    with open(os.path.join(args.output_dir, "training_info.json"), "w") as f:
        json.dump({
            "final_train_loss": train_loss,
            "final_val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "total_time_seconds": total_time,
            "parameters": param_count,
            "param_count_millions": param_count / 1e6,
            "config": config.__dict__,
        }, f, indent=2)
    
    return best_val_loss


if __name__ == "__main__":
    main()
