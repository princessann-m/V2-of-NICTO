# Training Guide

## Setup

```bash
pip install -r requirements.txt
```

## Configuration

Training is configured via `mom.training.configs.TrainingConfig`:

```python
from mom.training.configs import TrainingConfig

cfg = TrainingConfig(
    model_name="mamba",
    epochs=10,
    batch_size=8,
    learning_rate=1e-4,
    mixed_precision=True,
    distributed=False,
)
```

## BaseTrainer

```python
from mom.training.trainer import BaseTrainer

trainer = BaseTrainer(cfg)
trainer.train(model, train_loader, val_loader, optimizer, scheduler, loss_fn)
```

### Features

- **DDP**: Set `config.distributed=True` with multiple GPUs
- **AMP**: Automatic mixed precision when CUDA is available
- **Gradient accumulation**: `config.gradient_accumulation_steps`
- **Checkpointing**: Automatic saves at `config.save_interval`
- **Resume**: Set `config.resume_from="checkpoints/checkpoint-500.pt"`
- **Validation**: Run at `config.val_interval`

## Pipelines

```python
from mom.training.pipelines import get_pipeline

pipeline_fn = get_pipeline("mamba")
pipeline = pipeline_fn(model, train_loader, val_loader)
```

Supported: `mamba`, `encoder_decoder`, `gan`, `simulation`, `virtual_lab`
