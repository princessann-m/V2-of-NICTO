"""Enhanced CLI for NICTO."""

from __future__ import annotations

import argparse
import json
import sys
import time

from ..core.orchestrator import Orchestrator
from ..config import MoMConfig, LLMConfig


def cmd_handle(args: argparse.Namespace):
    cfg = MoMConfig(llm=LLMConfig(provider="heuristic"), global_deadline=args.deadline)
    orch = Orchestrator(cfg)
    start = time.perf_counter()
    result = orch.handle_request(args.input, request_id=args.request_id)
    latency = (time.perf_counter() - start) * 1000
    print(json.dumps({"input": args.input, "result": result, "latency_ms": latency}, indent=2))


def cmd_models(_args: argparse.Namespace):
    from ..models.model_registry import ModelRegistry
    reg = ModelRegistry()
    print(json.dumps({"experts": reg.experts}, indent=2))


def cmd_benchmark(args: argparse.Namespace):
    from ..benchmarks.framework import BenchmarkFramework
    bf = BenchmarkFramework()
    dataset = [{"input": "Calculate 2+2"}, {"input": "Write a python add function"}]
    res = bf.run_comparison(args.mode, dataset)
    print(json.dumps(res.__dict__, indent=2))


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(prog="nicto")
    sub = parser.add_subparsers(dest="command")

    p_handle = sub.add_parser("handle", help="Handle a task")
    p_handle.add_argument("input", help="User input")
    p_handle.add_argument("--request-id", default=None)
    p_handle.add_argument("--deadline", type=float, default=5.0)
    p_handle.set_defaults(func=cmd_handle)

    p_models = sub.add_parser("models", help="List registered models/experts")
    p_models.set_defaults(func=cmd_models)

    p_bench = sub.add_parser("benchmark", help="Run a benchmark")
    p_bench.add_argument("--mode", default="A")
    p_bench.set_defaults(func=cmd_benchmark)

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.func(args)


app = main
