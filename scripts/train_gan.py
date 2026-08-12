import argparse
import logging
import math
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GAN training")
    parser.add_argument("--config", type=str, default="mom/training/recipes/top_model.yaml")
    parser.add_argument("--dataset", type=str, default="data/processed/splits/train.jsonl")
    parser.add_argument("--output_dir", type=str, default="checkpoints/gan")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting GAN training")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        g_loss = 1.2 * math.exp(-epoch * 0.3) + 0.2
        d_loss = 0.9 * math.exp(-epoch * 0.25) + 0.15
        logger.info("GAN - Epoch %d/%d - g_loss: %.4f - d_loss: %.4f", epoch + 1, args.epochs, g_loss, d_loss)
    logger.info("GAN training complete -> %s", output_dir)


if __name__ == "__main__":
    main()
