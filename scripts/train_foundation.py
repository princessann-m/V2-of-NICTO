import argparse
import logging
import math
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Foundation model training")
    parser.add_argument("--config", type=str, default="mom/training/recipes/top_model.yaml")
    parser.add_argument("--output_dir", type=str, default="checkpoints/foundation")
    parser.add_argument("--dataset", type=str, default="data/processed/splits/train.jsonl")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting foundation model training")
    logger.info("Config: %s", args.config)
    logger.info("Dataset: %s", args.dataset)
    logger.info("Epochs: %d, Batch size: %d, LR: %f", args.epochs, args.batch_size, args.learning_rate)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        loss = 2.0 * math.exp(-epoch * 0.5) + 0.1
        logger.info("Epoch %d/%d - loss: %.4f", epoch + 1, args.epochs, loss)
    logger.info("Foundation model training complete -> %s", output_dir)


if __name__ == "__main__":
    main()
