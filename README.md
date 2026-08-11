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
│       ├── mom-creative.svg
│       └── mom-video.svg
│
├── tests/                    # Pytest suite
│   ├── test_pipeline.py
│   ├── test_mamba.py
│   ├── test_gan.py
│   ├── test_simulation.py
│   ├── test_virtual_lab.py
│   ├── test_router.py
│   ├── test_training.py
│   └── ...
│
├── scripts/                  # Training/evaluation scripts
│   ├── train_toy_nicto.py
│   ├── train_nicto_hf.py
│   ├── evaluate_nicto.py
│   └── prepare_data.py
│
├── docker/
│   ├── Dockerfile.cpu
│   └── Dockerfile.gpu
│
├── requirements.txt
├── setup.py
├── run_demo.py
└── README.md
```

---

## Contributing

MoM is a research architecture. Contributions are welcome in:

- Model training and evaluation
- Router optimization
- Tool integration
- Benchmarking
- Documentation

Please see `docs/ARCHITECTURE.md` for design details and `docs/TRAINING.md` for training guidelines.

---

## License

MIT — see `LICENSE` for details.

---

## Citation

If you use MoM in academic research, please cite:

```bibtex
@misc{mom2025,
  title={Mixture of Models (MoM): A Sparse Cooperative AI Architecture},
  author={MoM Contributors},
  year={2025},
  url={https://github.com/princessann-m/V2-of-NICTO}
}
```

---

## Contact

For questions, issues, or collaboration inquiries, please open an issue on GitHub or reach out through the repository.

---

## The Vision

The long-term goal of MoM is to explore whether **intelligence can emerge from cooperation rather than scale**.

Instead of building one model that must do everything, MoM investigates whether:

- **Specialists** can achieve deeper capability in narrow domains
- **Routers** can allocate computation intelligently
- **Tools** can extend reasoning beyond text
- **Simulations** can test hypotheses computationally
- **Verification** can challenge and improve outputs
- **Competition** between independent systems can increase robustness
- **Arbitration** can select the strongest result

The architecture is designed to be:

- **Modular** — components can be replaced or upgraded independently
- **Sparse** — computation is allocated only where needed
- **Adaptive** — the system adjusts to task difficulty and available resources
- **Verifiable** — outputs are checked, not trusted
- **Scalable** — new experts can be added without redesigning the system

This is an experimental direction in AI architecture. The experiments will determine whether the approach holds.

The question is not whether one model can learn everything.

The question is whether **specialized components, working together under intelligent routing and verification, can form a better AI system than any single model alone**.

MoM is built to find out.

---

*Built with PyTorch, Mamba, and the belief that the future of AI may be cooperative, not monolithic.*
