"""Toy training script for `nicto` adapter.

Usage (in virtualenv with dependencies installed):

    python scripts/train_toy_nicto.py

This script trains for one epoch on a tiny synthetic dataset and saves a small model.
"""

from mom.models.nicto import NictoAdapter


def main():
    nicto = NictoAdapter()
    # tiny toy dataset: short prompts + completions
    ds = [
        {"text": "Q: What is 2+2?\nA: 4"},
        {"text": "Q: Sum 3 and 5.\nA: 8"},
        {"text": "Q: Write a python func add(a,b).\nA: def add(a,b): return a + b"},
    ]
    print("Starting toy training (may be skipped if transformers not installed)")
    out = nicto.train_toy(ds, output_dir="nicto_toy_model")
    print("Result:", out)


if __name__ == "__main__":
    main()
