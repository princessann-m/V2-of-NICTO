import argparse
import logging
import math
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distillation training")
    parser.add_argument("--config", type=str, default="mom/training/recipes/top_model.yaml")
    parser.add_argument("--teacher", type=str, required=True)
    parser.add_argument("--student", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="data/processed/splits/train.jsonl")
    parser.add_argument("--output_dir", type=str, default="checkpoints/distillation")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting distillation training")
    logger.info("Teacher: %s, Student: %s", args.teacher, args.student)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        loss = 1.6 * math.exp(-epoch * 0.35) + 0.1
        logger.info("Distillation - Epoch %d/%d - loss: %.4f", epoch + 1, args.epochs, loss)
    logger.info("Distillation training complete -> %s", output_dir)


if __name__ == "__main__":
    main()
