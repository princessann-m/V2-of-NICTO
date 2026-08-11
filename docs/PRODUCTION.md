Production training and deployment notes for NICTo (MoM)
=====================================================

This document describes how to train and deploy a real `nicto` model using the
repository scaffolding. The repository provides templates and scripts; you must
execute training on appropriate hardware (GPUs/TPUs) and manage credentials.

Key components provided:
- `docker/` Dockerfiles for CPU and GPU environments
- `scripts/prepare_data.py` to convert JSONL to training text
- `scripts/train_nicto_hf.py` HF Trainer entrypoint (template)
- `scripts/evaluate_nicto.py` generation/eval helper

Hardware recommendations:
- For toy experiments: single GPU with >= 8GB VRAM.
- For production-scale models: multiple A100/RTX 6000 GPUs, distributed training, and high-speed interconnect.

Data requirements and cleaning:
- Provide training data as JSONL with `{"prompt":..., "completion":...}` per line.
- Use `scripts/prepare_data.py` to create a single concatenated text file.

Training example (run inside GPU Docker image):

```bash
docker build -f docker/Dockerfile.gpu -t nicto-train .
docker run --gpus all -it -v $PWD:/workspace nicto-train bash
python scripts/prepare_data.py --input data/raw.jsonl --output data/processed.txt
python scripts/train_nicto_hf.py --train_file data/processed.txt --output_dir models/nicto_out --num_train_epochs 3 --per_device_train_batch_size 2
```

Security and sandbox:
- Never run unreviewed third-party code in the training container with elevated privileges.
- See `docs/SECURITY.md` for sandbox recommendations.

Logging & experiment tracking:
- Integrate with Weights & Biases or similar for experiment tracking. The training scripts are simple templates; adapt to include `accelerate` and distributed launchers for real training.

Limitations:
- The provided scripts are templates and must be adapted for large-scale workloads, mixed precision, gradient accumulation, and proper checkpointing.
