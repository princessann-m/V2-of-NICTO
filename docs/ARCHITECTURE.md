# NICTO Architecture

## Overview

NICTO is a modular Mixture of Minds (MoM) framework built around a multi-expert orchestration layer with intelligent routing, verification, and benchmarking.

## Core Components

```
mom/
  core/
    routing/
      master_router.py    # Task analysis, MME selection, latency budget, sparsity
      expert_router.py    # Sparse top-k expert activation
      learned.py          # Learned router placeholder
    orchestrator.py       # Request lifecycle: parse -> route -> MME -> verify -> judge
    router.py             # Legacy router (heuristic)
  models/
    model_registry.py     # Expert registry
    top_model.py          # Task parser
    llm_client.py         # LLM abstraction
  experts/
    mme.py                # Mixed Mamba Expert System
    math_expert.py        # Domain experts
    coding_expert.py
    reasoning_expert.py
    science_expert.py
    vision_expert.py
  tools/
    calculator.py
    coding_studio.py
    simulator.py
    simulation/           # Physics simulators
    virtual_lab/          # Virtual lab instruments
  verification/
    hallucination.py      # Hallucination checker
  judges/
    judge.py              # Judge ensemble
  training/
    trainer.py            # BaseTrainer (DDP, AMP, checkpointing)
    configs.py            # TrainingConfig
    pipelines.py          # Per-model pipelines
  benchmarks/
    framework.py          # A/B/C/D/E/F/G/H/I/J comparison modes
  api/
    server.py             # FastAPI server
  cli/
    main.py               # Enhanced CLI
```

## Routing

### MasterRouter
1. Analyzes user input into a `TaskAnalysis` (task_type, complexity, tools, modality, latency budget)
2. Selects an MME config via sparse activation
3. Activates required tools
4. Selects modality pipeline
5. Allocates latency budget

### ExpertRouter
- Computes capability overlap scores between task and experts
- Returns top-k experts with configurable k

## Training

- Distributed training via DDP
- Mixed precision (AMP)
- Gradient accumulation
- Checkpointing with resume support
- Validation and logging

## Benchmarks

- Modes A-J cover full vs sparse, single vs multi, with/without judges, ablations
- Metrics: accuracy, hallucination rate, latency, compute cost
- Result storage in JSON

## API

- `POST /handle` — process a request
- `GET /models` — list registered experts
- `POST /benchmark` — run benchmark mode
- `GET /health` — health check

## CLI

- `nicto handle <input>` — process input
- `nicto models` — list models
- `nicto benchmark --mode <A-J>` — run benchmark
