import logging
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TrainingTracker:
    def __init__(self, window_size: int = 100, log_dir: str = "logs/training"):
        self.window_size = window_size
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.step = 0
        self.epoch = 0
        self.start_time = time.time()
        self.last_step_time = self.start_time
        self.grad_norms: deque = deque(maxlen=window_size)
        self.memory_usage: deque = deque(maxlen=window_size)
        self.samples_processed = 0

    def update(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if step is not None:
            self.step = step
        else:
            self.step += 1
        now = time.time()
        dt = now - self.last_step_time
        self.last_step_time = now
        for key, value in metrics.items():
            self.metrics[key].append(value)
        throughput = self.metrics.get("batch_size", deque([1]))[-1] / max(dt, 1e-6)
        self.metrics["throughput"].append(throughput)
        self.samples_processed += self.metrics.get("batch_size", deque([0]))[-1]

    def log_loss(self, loss: float) -> None:
        self.update({"loss": loss})

    def log_lr(self, lr: float) -> None:
        self.metrics["learning_rate"].append(lr)

    def log_grad_norm(self, norm: float) -> None:
        self.grad_norms.append(norm)

    def log_memory(self, memory_bytes: int) -> None:
        self.memory_usage.append(memory_bytes)

    def get_latest(self, key: str) -> Optional[float]:
        if key in self.metrics and self.metrics[key]:
            return self.metrics[key][-1]
        return None

    def get_average(self, key: str, last_n: Optional[int] = None) -> float:
        if key not in self.metrics or not self.metrics[key]:
            return 0.0
        data = list(self.metrics[key])[-last_n:] if last_n else list(self.metrics[key])
        return sum(data) / len(data)

    def state(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "epoch": self.epoch,
            "elapsed": time.time() - self.start_time,
            "latest_metrics": {k: self.get_latest(k) for k in self.metrics},
            "average_metrics": {k: self.get_average(k) for k in self.metrics},
        }

    def save(self) -> None:
        state = self.state()
        import json
        path = self.log_dir / f"tracker_step{self.step}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
