# NICTO Training Guide

## Overview

This guide explains how to train all NICTO model components, from small 7M-parameter experts up to the full 1-trillion-parameter system.

**Status**: Training infrastructure is IMPLEMENTED. All models are currently UNTRAINED. This guide provides the pathway to training them.

---

## Training Philosophy

NICTO uses a **multi-stage training pipeline**:

1. **Foundation Training** — Pretrain base models on general data
2. **Specialist Training** — Fine-tune experts on domain-specific data
3. **Distillation** — Transfer capabilities from large teachers to efficient students
4. **MME Training** — Train expert routers and aggregation
5. **Verifier/Judge Training** — Train evaluation models
6. **End-to-End Training** — Fine-tune the complete pipeline

---

## Parameter Scaling Plan

### Target: 1 Trillion Total Parameters

| Component | Scale | Parameters | Count | Total |
|-----------|-------|------------|-------|-------|
| **Foundation Models** | | | | |
| Base Mamba | XXL | 13B | 2 | 26B |
| Encoder-Decoder | Large | 1B | 1 | 1B |
| **Expert Pool** | | | | |
| Large Experts | XL | 7B | 50 | 350B |
| Medium Experts | Large | 370M | 100 | 37B |
| Small Experts | Medium | 130M | 160 | 20.8B |
| **Supporting Models** | | | | |
| Verifier | Large | 370M | 1 | 0.37B |
| Judges | Large | 370M | 2 | 0.74B |
| Director | Large | 370M | 1 | 0.37B |
| Router | Small | 30M | 1 | 0.03B |
| **Generation** | | | | |
| GAN | Medium | 130M | 1 | 0.13B |
| **Total** | | | | **~436B** |

**Note**: The 1T target is a scaling goal. The architecture supports up to 310 experts, but full 1T training requires significant GPU resources and data.

---

## Stage 1: Foundation Training

### Purpose

Pretrain base Mamba and encoder-decoder models on general language data.

### Data Requirements

- **Foundation corpus**: 1T+ tokens of high-quality text
- **Sources**: OpenWebText, Common Crawl, Books, Wikipedia
- **Format**: JSONL with `{"text": "..."}` fields

### Training Configuration

```bash
# Base Mamba (13B parameters)
python scripts/train_foundation.py \
    --config mom/training/recipes/mamba_xxl.yaml \
    --dataset_path data/foundation/train.jsonl \
    --output_dir checkpoints/mamba_xxl_foundation \
    --num_nodes 8 \
    --gpus_per_node 8
```

### Expected Compute

| Model | Parameters | GPUs | Time (1T tokens) |
|-------|-----------|------|------------------|
| mamba_xsmall | 7M | 1 | ~1 hour |
| mamba_small | 30M | 1 | ~4 hours |
| mamba_medium | 130M | 4 | ~1 day |
| mamba_large | 370M | 8 | ~3 days |
| mamba_xl | 1B | 16 | ~1 week |
| mamba_xxl | 7B | 64 | ~1 month |
| mamba_xxxl | 13B | 128 | ~2 months |

---

## Stage 2: Specialist Training

### Purpose

Fine-tune foundation models on domain-specific data to create experts.

### Data Requirements

| Domain | Data Needed | Sources |
|--------|-------------|---------|
| Mathematics | Math problems + solutions | MathQA, GSM8K, MATH |
| Coding | Code pairs (problem→solution) | The Stack, GitHub, CodeContests |
| Science | Scientific text + Q&A | ArXiv, PubMed, ScienceQA |
| Reasoning | Logic puzzles, reasoning chains | CommonsenseQA, HotpotQA |
| Vision | Image-text pairs | LAION, COCO |
| Audio | Audio-text pairs | LibriSpeech, Common Voice |

### Training Script

```bash
# Train math expert
python scripts/train_mamba_experts.py \
    --config mom/training/recipes/mamba_large.yaml \
    --dataset_path data/domain/math/train.jsonl \
    --output_dir checkpoints/experts/math_expert \
    --role math_expert \
    --capabilities algebra,calculus,statistics

# Train coding expert
python scripts/train_mamba_experts.py \
    --config mom/training/recipes/mamba_large.yaml \
    --dataset_path data/domain/coding/train.jsonl \
    --output_dir checkpoints/experts/coding_expert \
    --role coding_expert \
    --capabilities python,debug,testing
```

### Expert Training Order

1. Start with **medium-sized experts** (130M params) for common domains
2. Train **large experts** (370M-1B) for complex domains (math, coding, reasoning)
3. Use **distillation** to create smaller experts from larger ones
4. Scale to **7B+ experts** only for critical, high-traffic domains

---

## Stage 3: Distillation

### Purpose

Transfer capabilities from large teacher models to smaller, efficient student experts.

### Process

```
TEACHER (large, capable)
    ↓
Generate candidate outputs on training data
    ↓
VERIFICATION LAYER
    ↓
Filter low-quality outputs
    ↓
TRAINING DATA (verified pairs)
    ↓
STUDENT (small, efficient)
    ↓
Train on verified pairs
    ↓
SPECIALIZED EXPERT
```

### Training Script

```bash
python scripts/train_distillation.py \
    --teacher_config mom/training/recipes/mamba_xl.yaml \
    --student_config mom/training/recipes/mamba_medium.yaml \
    --teacher_checkpoint checkpoints/mamba_xl_foundation \
    --dataset_path data/distillation/teacher_generated.jsonl \
    --output_dir checkpoints/experts/distilled_math_expert \
    --verification_threshold 0.8 \
    --temperature 3.0 \
    --alpha 0.5
```

### Distillation Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `temperature` | Softmax temperature for soft labels | 3.0-5.0 |
| `alpha` | Weight between soft and hard loss | 0.5 |
| `verification_threshold` | Minimum quality for training data | 0.8 |
| `max_distillation_samples` | Max samples per expert | 1M |

---

## Stage 4: Router Training

### Purpose

Train the Master Router and Expert Router to make optimal selection decisions.

### Data Requirements

- Task-expert mapping traces
- Execution logs from MME systems
- Quality scores from judges
- Latency measurements

### Training Script

```bash
python scripts/train_router.py \
    --config mom/training/recipes/router.yaml \
    --routing_data data/routing/traces.jsonl \
    --output_dir checkpoints/router \
    --utility_weights "quality=0.5,latency=0.3,compute=0.2"
```

### Router Training Approach

1. **Supervised pretraining**: Use expert selection traces from rule-based router
2. **Reinforcement learning**: Optimize for utility function (quality - latency - cost)
3. **Online learning**: Continuously improve from execution logs

---

## Stage 5: Verifier & Judge Training

### Purpose

Train models to evaluate candidate outputs and compare alternatives.

### Data Requirements

- Candidate outputs with quality labels
- Human preference judgments
- Automated quality metrics
- Tool verification results

### Training Script

```bash
# Verifier
python scripts/train_verifier.py \
    --config mom/training/recipes/verifier.yaml \
    --dataset_path data/verification/candidates.jsonl \
    --output_dir checkpoints/verifier

# Judges
python scripts/train_judges.py \
    --config mom/training/recipes/judge.yaml \
    --dataset_path data/judging/preferences.jsonl \
    --output_dir checkpoints/judge_alpha \
    --judge_id judge_alpha
```

---

## Stage 6: End-to-End Training

### Purpose

Fine-tune the complete pipeline for optimal coordination.

### Approach

1. **Freeze** all expert weights
2. **Train** only the router and aggregation layers
3. **Gradually unfreeze** top layers of selected experts
4. **End-to-end fine-tuning** with small learning rate

### Training Script

```bash
python scripts/train_end_to_end.py \
    --config mom/training/recipes/end_to_end.yaml \
    --expert_checkpoints checkpoints/experts/ \
    --router_checkpoint checkpoints/router \
    --dataset_path data/e2e/train.jsonl \
    --output_dir checkpoints/nicto_e2e \
    --freeze_experts true \
    --unfreeze_top_layers 2
```

---

## Data Preparation Pipeline

### Step 1: Download

```python
from mom.datasets import DatasetPreparer

preparer = DatasetPreparer()
preparer.download(
    sources=[
        {"name": "openwebtext", "url": "https://huggingface.co/datasets/Skylion007/openwebtext", "type": "huggingface"},
        {"name": "mathqa", "url": "datasets/math_qa", "type": "huggingface"},
        {"name": "the_stack", "url": "datasets/the_stack", "type": "huggingface"},
    ]
)
```

### Step 2: Prepare

```python
preparer.prepare(
    input_dir="data/raw",
    output_dir="data/processed",
    filters=[
        "quality",
        "deduplicate",
        "length:100,2048",
        "language:en",
        "toxicity:0.1",
    ]
)
```

### Step 3: Tokenize

```python
from mom.models.tokenizer import BPETokenizer

tokenizer = BPETokenizer(vocab_size=10000)
preparer.tokenize(
    input_dir="data/processed",
    output_dir="data/tokenized",
    tokenizer=tokenizer,
)
```

### Step 4: Split

```python
preparer.split(
    input_dir="data/tokenized",
    output_dir="data/final",
    splits={"train": 0.9, "validation": 0.05, "test": 0.05},
    seed=42,
)
```

---

## Training Infrastructure

### Hardware Requirements

| Scale | GPUs | VRAM | Nodes | Time |
|-------|------|------|-------|------|
| Small (7M-30M) | 1-4 | 16-32GB | 1 | Hours |
| Medium (130M-370M) | 8-16 | 32GB | 1-2 | Days |
| Large (1B-7B) | 64-128 | 80GB | 4-8 | Weeks |
| XL (13B+) | 128-256 | 80GB | 8-16 | Months |
| 1T Total | 1024+ | 80GB | 64+ | Months |

### Distributed Training

```bash
# Multi-node with torchrun
torchrun \
    --nnodes=8 \
    --nproc_per_node=8 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=29500 \
    scripts/train_foundation.py \
    --config mom/training/recipes/mamba_xxl.yaml \
    --dataset_path data/foundation/train.jsonl \
    --output_dir checkpoints/mamba_xxl_foundation
```

### Mixed Precision

```bash
# FP16/BF16 training
python scripts/train_mamba_experts.py \
    --mixed_precision bf16 \
    --gradient_accumulation_steps 4 \
    ...
```

---

## Monitoring

### TrainingTracker

```python
from mom.training.monitoring import TrainingTracker

tracker = TrainingTracker(log_dir="logs/training")
tracker.log_loss(0.5, step=100)
tracker.log_lr(1e-4, step=100)
tracker.log_gpu_memory(step=100)
tracker.log_throughput(1000, step=100)  # samples/sec
```

### Checkpointing

```python
from mom.training.monitoring import TrainingReporter

reporter = TrainingReporter(
    checkpoint_dir="checkpoints",
    save_interval=500,
    keep_best=3,
    metric="val_loss",
)
```

---

## Scaling to 1T Parameters

### Phase 1: Prototype (Complete)
- [x] Small models (7M-30M params)
- [x] Training infrastructure
- [x] Data pipeline
- [x] Distillation framework

### Phase 2: Medium Scale (In Progress)
- [ ] Medium models (130M-370M params)
- [ ] Domain-specific expert training
- [ ] Router training
- [ ] Verifier/Judge training

### Phase 3: Large Scale (Planned)
- [ ] Large models (1B-7B params)
- [ ] Multi-node distributed training
- [ ] 50-100 trained experts
- [ ] Full verification pipeline

### Phase 4: Full Scale (Goal)
- [ ] XXL models (13B+ params)
- [ ] 310 expert pool
- [ ] 1T total parameters
- [ ] Production-grade inference

---

## Data Sources

### Recommended Datasets

| Domain | Dataset | Size | License |
|--------|---------|------|---------|
| Foundation | OpenWebText | 38GB | Open |
| Foundation | Common Crawl | Petabytes | Open |
| Foundation | The Pile | 825GB | MIT |
| Mathematics | MATH | 7.5K problems | MIT |
| Mathematics | GSM8K | 8.5K problems | CC BY-SA |
| Coding | The Stack | 3TB | MIT |
| Coding | CodeParrot | 180GB | MIT |
| Science | ArXiv | Full text | Various |
| Science | PubMed | Full text | Open |
| Reasoning | CommonsenseQA | 12K | CC BY-SA |
| Vision | LAION-5B | 5B images | LAION |
| Audio | LibriSpeech | 960 hours | CC BY-4.0 |

### Data Quality

- Filter by quality score (>0.7)
- Remove duplicates (MinHash LSH)
- Filter by length (100-2048 tokens)
- Remove toxic content (perspective API)
- Verify language (fastText)
- Deduplicate across sources

---

## Training Recipes

### Small Expert (7M params)

```yaml
model:
  vocab_size: 10000
  d_model: 32
  n_layers: 2
  d_state: 16
  d_conv: 4
  expand: 2

training:
  epochs: 10
  batch_size: 64
  learning_rate: 1e-3
  warmup_steps: 100
  mixed_precision: true
  device: cuda
```

### Medium Expert (130M params)

```yaml
model:
  vocab_size: 50257
  d_model: 256
  n_layers: 4
  d_state: 128
  d_conv: 4
  expand: 2

training:
  epochs: 3
  batch_size: 32
  learning_rate: 5e-4
  warmup_steps: 1000
  gradient_accumulation_steps: 4
  mixed_precision: bf16
  distributed: true
```

### Large Expert (1B params)

```yaml
model:
  vocab_size: 50257
  d_model: 1024
  n_layers: 8
  d_state: 256
  d_conv: 4
  expand: 2

training:
  epochs: 1
  batch_size: 16
  learning_rate: 3e-4
  warmup_steps: 2000
  gradient_accumulation_steps: 16
  mixed_precision: bf16
  distributed: true
  checkpoint_every: 1000
```

---

## Evaluation

### Benchmarks

After training, evaluate each component:

```bash
# Foundation model
python scripts/evaluate_nicto.py --model checkpoints/mamba_xxl_foundation --benchmark lm_eval

# Experts
python scripts/evaluate_nicto.py --model checkpoints/experts/math_expert --benchmark math_qa

# Router
python scripts/evaluate_nicto.py --model checkpoints/router --benchmark routing_accuracy

# Full system
python scripts/evaluate_nicto.py --model checkpoints/nicto_e2e --benchmark mom_full
```

### Metrics

| Component | Metrics |
|-----------|---------|
| Foundation | Perplexity, throughput, memory |
| Experts | Domain accuracy, task completion |
| Router | Expert selection accuracy, latency |
| Verifier | Precision, recall, F1 |
| Judges | Agreement with human judgment |
| End-to-End | Task success, latency, hallucination rate |

---

## Troubleshooting

### Out of Memory
- Reduce batch size
- Increase gradient accumulation
- Enable mixed precision
- Use model parallelism

### Slow Training
- Increase batch size
- Use more GPUs
- Enable mixed precision
- Use faster data loading (more workers, prefetch)

### Poor Quality
- More training data
- Longer training
- Better data filtering
- Adjust learning rate
- Use distillation from better teachers

---

## Next Steps

1. Prepare foundation training data (1T+ tokens)
2. Train small models to validate pipeline
3. Scale to medium models (130M-370M)
4. Train first 10-20 domain experts
5. Implement router training with execution traces
6. Train verifier and judges
7. Scale to large models (1B-7B)
8. Expand expert pool to 100+
9. Implement full 1T parameter plan
10. Continuous evaluation and optimization

---

*This guide will be updated as training progresses. Check docs/TRAINING.md for the latest instructions.*
