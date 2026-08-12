import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TrainingReporter:
    def __init__(self, log_dir: str = "logs/training", use_console: bool = True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.use_console = use_console
        self._progress = None
        self._alerts: List[Dict[str, Any]] = []

    def on_epoch_start(self, epoch: int, total_epochs: int) -> None:
        msg = f"Epoch {epoch + 1}/{total_epochs} started"
        self._log(msg)

    def on_epoch_end(self, epoch: int, total_epochs: int, metrics: Dict[str, float]) -> None:
        msg = f"Epoch {epoch + 1}/{total_epochs} ended"
        if metrics:
            parts = [f"{k}: {v:.4f}" for k, v in metrics.items()]
            msg += " - " + ", ".join(parts)
        self._log(msg)

    def on_step(self, step: int, total_steps: int, metrics: Dict[str, float]) -> None:
        if step % 10 != 0 and step != total_steps:
            return
        parts = [f"{k}: {v:.4f}" for k, v in metrics.items()]
        progress = step / max(total_steps, 1)
        msg = f"Step {step}/{total_steps} [{progress:>3.0%}] - " + ", ".join(parts)
        self._log(msg)

    def on_checkpoint(self, path: str, metric: Optional[float] = None) -> None:
        msg = f"Checkpoint saved: {path}"
        if metric is not None:
            msg += f" (metric: {metric:.4f})"
        self._log(msg)

    def on_eval(self, metrics: Dict[str, float]) -> None:
        parts = [f"{k}: {v:.4f}" for k, v in metrics.items()]
        self._log("Eval results - " + ", ".join(parts))

    def alert(self, message: str, level: str = "warning") -> None:
        entry = {"level": level, "message": message}
        self._alerts.append(entry)
        prefix = {"warning": "WARNING", "error": "ERROR", "info": "INFO"}.get(level, "ALERT")
        self._log(f"{prefix}: {message}")

    def summary(self, metrics: Dict[str, float]) -> None:
        self._log("Training complete. Summary:")
        for k, v in metrics.items():
            self._log(f"  {k}: {v:.4f}")

    def _log(self, message: str) -> None:
        if self.use_console:
            print(message, file=sys.stdout, flush=True)
        logger.info(message)
