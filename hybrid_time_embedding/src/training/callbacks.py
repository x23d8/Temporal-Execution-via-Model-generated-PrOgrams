"""
callbacks.py — Training callbacks and SmartCheckpointSaver.
Handles top-k checkpoint management with manifest tracking,
gate value monitoring, and metric logging callbacks.
"""

import json
import logging
import os
import shutil
from typing import Any, Dict, List, Optional

import torch
from safetensors.torch import save_file, load_file

logger = logging.getLogger(__name__)


class SmartCheckpointSaver:
    """
    Saves checkpoints when val_mae improves by threshold.
    Keeps only top-k checkpoints by val_mae.
    Always saves a 'latest' checkpoint every save_every_steps.
    Saves: model state, optimizer state, scheduler state, config, metrics history.

    Args:
        output_dir: Root directory for checkpoint folders.
        top_k: Maximum number of best checkpoints to retain.
        improvement_threshold: Relative improvement needed to save (e.g. 0.01 = 1%).
        save_every_steps: Save 'latest' checkpoint unconditionally every N steps.
    """

    MANIFEST_FILE = "manifest.json"

    def __init__(
        self,
        output_dir: str,
        top_k: int = 3,
        improvement_threshold: float = 0.01,
        save_every_steps: int = 1000,
    ) -> None:
        self.output_dir = output_dir
        self.top_k = top_k
        self.improvement_threshold = improvement_threshold
        self.save_every_steps = save_every_steps
        self.best_mae: Optional[float] = None
        self.checkpoints: List[Dict] = []
        os.makedirs(output_dir, exist_ok=True)
        self._load_manifest()

    def _manifest_path(self) -> str:
        return os.path.join(self.output_dir, self.MANIFEST_FILE)

    def _load_manifest(self) -> None:
        path = self._manifest_path()
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            self.checkpoints = data.get("checkpoints", [])
            if self.checkpoints:
                self.best_mae = min(c["val_mae"] for c in self.checkpoints)

    def _save_manifest(self) -> None:
        manifest = {
            "best_mae": self.best_mae,
            "checkpoints": sorted(self.checkpoints, key=lambda c: c["val_mae"]),
        }
        with open(self._manifest_path(), "w") as f:
            json.dump(manifest, f, indent=2)

    def _save_checkpoint(
        self,
        folder: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[Any],
        config: Any,
        metrics: Dict,
    ) -> None:
        os.makedirs(folder, exist_ok=True)
        # Save model weights
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        save_file(state_dict, os.path.join(folder, "model.safetensors"))
        # Save optimizer and scheduler
        if optimizer is not None:
            torch.save(optimizer.state_dict(), os.path.join(folder, "optimizer.pt"))
        if scheduler is not None:
            torch.save(scheduler.state_dict(), os.path.join(folder, "scheduler.pt"))
        # Save config
        if hasattr(config, "save"):
            config.save(os.path.join(folder, "config.json"))
        # Save metrics
        with open(os.path.join(folder, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    def maybe_save(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[Any],
        metrics: Dict,
        step: int,
        epoch: int,
        phase: str,
        config: Optional[Any] = None,
    ) -> bool:
        """
        Save if val_mae improved by threshold, evict if over top_k.
        Always save 'latest' every save_every_steps.

        Args:
            model: Current model.
            optimizer: Current optimizer (optional).
            scheduler: Current scheduler (optional).
            metrics: Dict containing at least {"val_mae": float}.
            step: Current global training step.
            epoch: Current epoch number.
            phase: Training phase name (e.g. "phase1").
            config: HybridConfig instance for saving.

        Returns:
            True if a named checkpoint was saved.
        """
        val_mae = metrics.get("val_mae", float("inf"))
        saved = False

        # Always save latest
        if step % self.save_every_steps == 0 and step > 0:
            latest_dir = os.path.join(self.output_dir, "latest")
            self._save_checkpoint(latest_dir, model, optimizer, scheduler, config, metrics)
            logger.info(f"Saved latest checkpoint at step {step}.")

        # Save if improved
        improved = (
            self.best_mae is None
            or (self.best_mae - val_mae) / max(self.best_mae, 1e-8) > self.improvement_threshold
        )

        if improved:
            self.best_mae = val_mae
            folder_name = f"{phase}_step{step}_mae{val_mae:.4f}"
            ckpt_dir = os.path.join(self.output_dir, folder_name)
            self._save_checkpoint(ckpt_dir, model, optimizer, scheduler, config, metrics)

            self.checkpoints.append({
                "folder": folder_name,
                "step": step,
                "epoch": epoch,
                "phase": phase,
                "val_mae": val_mae,
                "metrics": metrics,
            })

            # Evict worst if over top_k
            if len(self.checkpoints) > self.top_k:
                self.checkpoints.sort(key=lambda c: c["val_mae"])
                worst = self.checkpoints.pop(-1)
                worst_dir = os.path.join(self.output_dir, worst["folder"])
                if os.path.exists(worst_dir):
                    shutil.rmtree(worst_dir)
                logger.info(f"Evicted checkpoint: {worst['folder']} (mae={worst['val_mae']:.4f})")

            self._save_manifest()
            logger.info(f"Saved checkpoint {folder_name} (mae={val_mae:.4f})")
            saved = True

        return saved

    def load_best(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ) -> Dict:
        """
        Load best checkpoint (lowest val_mae) from manifest.

        Args:
            model: Model to load weights into.
            optimizer: Optional optimizer to restore state.
            scheduler: Optional scheduler to restore state.

        Returns:
            Metrics dict from the best checkpoint.
        """
        self._load_manifest()
        if not self.checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {self.output_dir}")

        best = min(self.checkpoints, key=lambda c: c["val_mae"])
        folder = os.path.join(self.output_dir, best["folder"])
        logger.info(f"Loading best checkpoint: {best['folder']} (mae={best['val_mae']:.4f})")

        # Load model
        state_dict = load_file(os.path.join(folder, "model.safetensors"))
        model.load_state_dict(state_dict, strict=False)

        # Load optimizer
        opt_path = os.path.join(folder, "optimizer.pt")
        if optimizer is not None and os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))

        # Load scheduler
        sched_path = os.path.join(folder, "scheduler.pt")
        if scheduler is not None and os.path.exists(sched_path):
            scheduler.load_state_dict(torch.load(sched_path, map_location="cpu"))

        with open(os.path.join(folder, "metrics.json")) as f:
            return json.load(f)


class GateMonitorCallback:
    """
    Warns when the fusion gate falls below a threshold after warm-up.

    Args:
        threshold: Gate value below which a warning is emitted.
        warmup_steps: Steps to wait before monitoring begins.
        logger: Optional logger. Falls back to module-level logger.
    """

    def __init__(
        self,
        threshold: float = 0.01,
        warmup_steps: int = 200,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.threshold = threshold
        self.warmup_steps = warmup_steps
        self._logger = logger or logging.getLogger(__name__)

    def on_step(self, step: int, gate_value: float) -> None:
        """Check gate and warn if collapsed."""
        if step >= self.warmup_steps and gate_value < self.threshold:
            self._logger.warning(
                f"[GateMonitor] step={step}: gate={gate_value:.5f} < threshold={self.threshold}. "
                "Consider increasing lambda_gate or checking fusion layer."
            )


class MetricCallback:
    """
    Logs metrics to Python logger at each step/epoch.

    Args:
        logger: Logger instance.
        log_every_steps: Step interval for step-level logging.
    """

    def __init__(self, logger: logging.Logger, log_every_steps: int = 50) -> None:
        self._logger = logger
        self.log_every_steps = log_every_steps

    def on_step(self, step: int, metrics: Dict) -> None:
        """Log step metrics if at interval."""
        if step % self.log_every_steps == 0:
            parts = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float))
            self._logger.info(f"Step {step}: {parts}")

    def on_epoch(self, epoch: int, metrics: Dict) -> None:
        """Log epoch-level metrics."""
        parts = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float))
        self._logger.info(f"Epoch {epoch}: {parts}")
