"""Evaluation script for trained nicto models.

This script loads a saved model and runs simple generation tests on example prompts.
"""

import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir)

    prompts = [
        "What is 2+2?",
        "Write a Python function that adds two numbers.",
    ]
    for pr in prompts:
        inputs = tokenizer(pr, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=64)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        print("Prompt:", pr)
        print("Output:\n", text)
        print("---")


if __name__ == "__main__":
    main()
