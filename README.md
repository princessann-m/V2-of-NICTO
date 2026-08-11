# NICTO V2 — Modular Mixture of Minds

NICTO is an open, modular framework for building and evaluating Mixture of Experts (MoM) systems. It provides intelligent routing, multi-expert orchestration, hallucination verification, judge ensembles, benchmarking, training infrastructure, and an API/CLI interface.

## Features

- **Master Router**: Task analysis, MME system selection, tool activation, modality pipeline selection, latency budget allocation, configurable sparsity
- **Expert Router**: Sparse top-k activation with capability matching
- **MoM Core**: Orchestrator with parse -> route -> MME -> verify -> judge lifecycle
- **Verification**: Hallucination checker with retry logic
- **Judging**: Dual-judge ensemble arbitration
- **Training**: BaseTrainer with DDP, AMP, gradient accumulation, checkpointing, resume, validation, logging
- **Benchmarks**: A/B/C/D/E/F/G/H/I/J comparison modes, accuracy, hallucination rate, latency, compute cost metrics, ablation studies
- **API**: FastAPI server with `/handle`, `/models`, `/benchmark`, `/health`
- **CLI**: Enhanced CLI with `handle`, `models`, `benchmark` commands

## Installation

```bash
git clone https://github.com/your-org/V2-of-NICTO.git
cd V2-of-NICTO
pip install -r requirements.txt
python -m pytest tests/ -q
```

## Quick Start

### CLI

```bash
python -m mom.cli.main handle "Calculate 2 + 2"
python -m mom.cli.main handle "Write a python function to add two numbers"
python -m mom.cli.main models
python -m mom.cli.main benchmark --mode A
```

### API

```bash
uvicorn mom.api.server:app --host 0.0.0.0 --port 8000
```

Then:

```bash
curl -X POST http://localhost:8000/handle -H "Content-Type: application/json" -d '{"input": "Calculate 2+2"}'
curl http://localhost:8000/models
curl -X POST http://localhost:8000/benchmark -H "Content-Type: application/json" -d '{"mode": "A"}'
```

## Project Structure

```
mom/
  core/
    routing/              # MasterRouter, ExpertRouter, LearnedRouter
    orchestrator.py       # Request lifecycle
    router.py             # Legacy router
  models/                 # Model registry, top model, LLM client
  experts/                # Domain experts + MME
  tools/                  # Calculator, coding studio, simulator, simulation, virtual lab
  verification/           # Hallucination checker
  judges/                 # Judge ensemble
  training/               # BaseTrainer, configs, pipelines
  benchmarks/             # Benchmark framework
  api/                    # FastAPI server
  cli/                    # Enhanced CLI
tests/                    # Pytest suite
docs/                     # Architecture, training, deployment
```

## Benchmark Modes

| Mode | Description |
|------|-------------|
| A | Full system |
| B | No routing |
| C | No verification |
| D | No judging |
| E | Single expert |
| F | No experts |
| G | High sparsity |
| H | Low latency budget |
| I | Ablation variant 1 |
| J | Ablation variant 2 |

## Training

```python
from mom.training.configs import TrainingConfig
from mom.training.trainer import BaseTrainer

cfg = TrainingConfig(epochs=10, batch_size=8, device="cuda", mixed_precision=True)
trainer = BaseTrainer(cfg)
trainer.train(model, train_loader, val_loader)
```

## Docker

```bash
docker build -f docker/Dockerfile.cpu -t nicto:cpu .
docker run -p 8000:8000 nicto:cpu
```

## Contributing

PRs welcome. See `docs/ARCHITECTURE.md` for design details.

## License

MIT
