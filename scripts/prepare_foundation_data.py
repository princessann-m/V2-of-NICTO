"""Download and prepare a small public dataset for initial training."""

from __future__ import annotations

import os
import json
import random
from pathlib import Path

# Small public dataset sources
DATASETS = {
    "openwebtext_small": {
        "url": "https://huggingface.co/datasets/Skylion007/openwebtext/resolve/main/openwebtext/train.jsonl",
        "size_mb": 100,
        "description": "Small subset of OpenWebText for language modeling",
    },
    "tinystories": {
        "url": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.jsonl",
        "size_mb": 50,
        "description": "TinyStories - short stories for children",
    },
}

# Fallback: generate synthetic dataset if download fails
def generate_synthetic_dataset(output_path: str, num_samples: int = 10000, max_length: int = 512):
    """Generate a synthetic text dataset for initial training."""
    print(f"Generating {num_samples} synthetic samples...")
    
    # Create diverse synthetic text
    templates = [
        "The quick brown fox jumps over the lazy dog. " * 10,
        "In a world where artificial intelligence transforms every industry, " + "researchers develop new architectures. " * 20,
        "Mathematics is the language of science. " + "Equations describe physical phenomena. " * 15,
        "Programming requires logic, creativity, and patience. " + "Code is poetry written for machines. " * 15,
        "The universe is vast and mysterious. " + "Stars form in nebulae, planets orbit suns. " * 15,
        "Music combines rhythm, melody, and harmony. " + "Songs evoke emotions and memories. " * 15,
        "History teaches us about human nature. " + "Civilizations rise and fall through time. " * 15,
        "Biology studies life in all its forms. " + "Cells, organisms, ecosystems interact. " * 15,
        "Chemistry explores matter and its transformations. " + "Elements combine into compounds. " * 15,
        "Physics seeks to understand fundamental forces. " + "Gravity, electromagnetism, nuclear forces shape reality. " * 15,
    ]
    
    samples = []
    for i in range(num_samples):
        # Combine multiple templates
        text = ""
        num_sentences = random.randint(3, 8)
        for _ in range(num_sentences):
            text += random.choice(templates)
        
        # Truncate to max_length
        text = text[:max_length]
        samples.append({"text": text, "source": "synthetic", "id": i})
    
    # Save as JSONL
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    
    print(f"Saved {len(samples)} samples to {output_path}")
    return output_path


def download_tinystories(output_path: str, max_samples: int = 5000) -> str:
    """Download TinyStories dataset (small, public, permissive license)."""
    try:
        from datasets import load_dataset
        print("Downloading TinyStories dataset...")
        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for i, sample in enumerate(ds):
                if i >= max_samples:
                    break
                text = sample.get("text", "")
                if text and len(text.strip()) > 10:
                    f.write(json.dumps({"text": text, "source": "tinystories", "id": i}) + "\n")
        
        print(f"Downloaded {min(i+1, max_samples)} samples to {output_path}")
        return output_path
    except Exception as e:
        print(f"Failed to download TinyStories: {e}")
        print("Falling back to synthetic dataset...")
        return generate_synthetic_dataset(output_path, num_samples=max_samples)


def download_openwebtext_small(output_path: str, max_samples: int = 5000) -> str:
    """Download a small subset of OpenWebText."""
    try:
        from datasets import load_dataset
        print("Downloading OpenWebText dataset...")
        ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for i, sample in enumerate(ds):
                if i >= max_samples:
                    break
                text = sample.get("text", "")
                if text and len(text.strip()) > 10:
                    f.write(json.dumps({"text": text, "source": "openwebtext", "id": i}) + "\n")
        
        print(f"Downloaded {min(i+1, max_samples)} samples to {output_path}")
        return output_path
    except Exception as e:
        print(f"Failed to download OpenWebText: {e}")
        print("Falling back to TinyStories...")
        return download_tinystories(output_path, max_samples)


def prepare_training_data(output_dir: str = "data/foundation", max_samples: int = 5000):
    """Prepare training data for foundation model training."""
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "train.jsonl")
    val_path = os.path.join(output_dir, "val.jsonl")
    
    # Download dataset
    data_path = download_openwebtext_small(
        os.path.join(output_dir, "raw_train.jsonl"),
        max_samples=max_samples
    )
    
    # Read all samples
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    
    print(f"Total samples: {len(samples)}")
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(samples)
    
    split_idx = int(0.9 * len(samples))
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]
    
    # Write train split
    with open(train_path, "w", encoding="utf-8") as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + "\n")
    
    # Write val split
    with open(val_path, "w", encoding="utf-8") as f:
        for sample in val_samples:
            f.write(json.dumps(sample) + "\n")
    
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    print(f"Data prepared in {output_dir}")
    
    return train_path, val_path


if __name__ == "__main__":
    prepare_training_data()
