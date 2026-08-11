"""Training launcher for nicto using Hugging Face Transformers.

This script is a configurable training entrypoint. It is intended as a
template and must be run in an environment with GPUs and sufficient memory
for real training.

DO NOT run this in untrusted environments. Review `docs/SECURITY.md` before use.
"""

import argparse
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_file", required=True, help="Processed text file for training")
    p.add_argument("--output_dir", default="nicto_model_out")
    p.add_argument("--model_name", default="sshleifer/tiny-gpt2")
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    return p.parse_args()


def main():
    args = parse_args()
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
        from datasets import load_dataset
    except Exception as e:
        print("Missing dependencies:", e)
        raise

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    dataset = load_dataset("text", data_files={"train": args.train_file})

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"]) 

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized["train"], data_collator=data_collator)
    trainer.train()
    trainer.save_model(args.output_dir)

    print("Training complete. Model saved to", args.output_dir)


if __name__ == "__main__":
    main()
