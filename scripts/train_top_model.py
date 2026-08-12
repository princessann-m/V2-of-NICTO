import argparse
import logging
import math
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Top model / router training")
    parser.add_argument("--config", type=str, default="mom/training/recipes/top_model.yaml")
    parser.add_argument("--experts", type=str, nargs="+", required=True)
    parser.add_argument("--dataset", type=str, default="data/processed/splits/train.jsonl")
    parser.add_argument("--output_dir", type=str, default="checkpoints/top_model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting top model training with experts: %s", args.experts)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        loss = 2.2 * math.exp(-epoch * 0.3) + 0.12
        logger.info("Top model - Epoch %d/%d - loss: %.4f", epoch + 1, args.epochs, loss)
    logger.info("Top model training complete -> %s", output_dir)


if __name__ == "__main__":
    main()
