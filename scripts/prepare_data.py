"""Prepare dataset for nicto training.

This script converts a simple JSONL of {"prompt":..., "completion":...}
to a HF dataset or concatenated text file suitable for causal LM training.

Usage:
  python scripts/prepare_data.py --input data/raw.jsonl --output data/processed.txt
"""

import argparse
import json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    with open(args.input, "r", encoding="utf-8") as fh, open(args.output, "w", encoding="utf-8") as out:
        for line in fh:
            obj = json.loads(line)
            prompt = obj.get("prompt", "")
            completion = obj.get("completion", "")
            text = f"{prompt}\n\n{completion}\n\n"
            out.write(text)

    print("Wrote processed text to", args.output)


if __name__ == "__main__":
    main()
