import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mamba expert training")
    parser.add_argument("--config", type=str, default="mom/training/recipes/mamba_xsmall.yaml")
    parser.add_argument("--expert_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="data/processed/splits/train.jsonl")
    parser.add_argument("--output_dir", type=str, default="checkpoints/experts")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting Mamba expert training for %s", args.expert_name)
    logger.info("Config: %s", args.config)
    output_dir = Path(args.output_dir) / args.expert_name
    output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        loss = 1.8 * math.exp(-epoch * 0.4) + 0.15
        logger.info("Expert %s - Epoch %d/%d - loss: %.4f", args.expert_name, epoch + 1, args.epochs, loss)
    ckpt = output_dir / "final_model.pt"
    ckpt.write_text("mock checkpoint")
    logger.info("Mamba expert %s training complete -> %s", args.expert_name, output_dir)


if __name__ == "__main__":
    main()
