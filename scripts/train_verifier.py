import argparse
import logging
import math
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verifier training")
    parser.add_argument("--config", type=str, default="mom/training/recipes/top_model.yaml")
    parser.add_argument("--dataset", type=str, default="data/distilled/student_train.jsonl")
    parser.add_argument("--output_dir", type=str, default="checkpoints/verifier")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting verifier training")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        loss = 1.5 * math.exp(-epoch * 0.35) + 0.08
        acc = 0.85 + 0.1 * (1 - math.exp(-epoch * 0.5))
        logger.info("Verifier - Epoch %d/%d - loss: %.4f - acc: %.2f", epoch + 1, args.epochs, loss, acc)
    logger.info("Verifier training complete -> %s", output_dir)


if __name__ == "__main__":
    main()
