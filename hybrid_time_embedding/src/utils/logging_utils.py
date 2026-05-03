"""
logging_utils.py — Dual logging setup: Python logger + TensorBoard SummaryWriter
+ optional WandB. Used by all training phases to track metrics consistently.
"""

import logging
import os
from datetime import datetime
from typing import Optional, Tuple

from torch.utils.tensorboard import SummaryWriter

# Metrics logged every step
STEP_METRICS = [
    "loss/total", "loss/arithmetic", "loss/duration",
    "loss/consistency", "loss/gate_reg",
    "model/gate_value", "model/grad_norm",
    "train/lr_backbone", "train/lr_embedding", "train/lr_heads",
    "train/gpu_memory_gb",
]

# Metrics logged every epoch
EPOCH_METRICS = [
    "val/mae_arithmetic", "val/mae_duration", "val/mae_overall",
    "val/exact_match_arithmetic", "val/exact_match_duration",
    "val/consistency_rate", "val/within_1yr", "val/within_5yr",
    "model/learned_freq_mean", "model/learned_freq_std",
    "model/gate_value",
]


def setup_logging(
    log_dir: str,
    experiment_name: Optional[str] = None,
) -> Tuple[logging.Logger, SummaryWriter, Optional[object]]:
    """
    Initialize Python logger, TensorBoard writer, and optional WandB run.

    Args:
        log_dir: Base directory for log files and TensorBoard events.
        experiment_name: Optional name tag for the run.

    Returns:
        Tuple of (logger, tb_writer, wandb_run_or_None).
    """
    os.makedirs(log_dir, exist_ok=True)
    tb_dir = os.path.join(log_dir, "tb")
    os.makedirs(tb_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = experiment_name or f"run_{timestamp}"

    # Python logger
    logger = logging.getLogger(run_name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        log_file = os.path.join(log_dir, f"{run_name}.log")
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)

    # TensorBoard
    tb_writer = SummaryWriter(log_dir=os.path.join(tb_dir, run_name))

    # WandB (optional)
    wandb_run = None
    if os.environ.get("WANDB_API_KEY"):
        try:
            import wandb
            wandb_run = wandb.init(project="hybrid-time-embedding", name=run_name, reinit=True)
        except Exception as e:
            logger.warning(f"WandB init failed: {e}")

    logger.info(f"Logging initialized. TensorBoard: {tb_dir}")
    return logger, tb_writer, wandb_run


def log_step_metrics(
    tb_writer: SummaryWriter,
    metrics: dict,
    step: int,
    wandb_run=None,
) -> None:
    """Write scalar metrics to TensorBoard and optionally WandB."""
    for k, v in metrics.items():
        tb_writer.add_scalar(k, v, step)
    if wandb_run is not None:
        try:
            import wandb
            wandb.log(metrics, step=step)
        except Exception:
            pass


def log_epoch_metrics(
    tb_writer: SummaryWriter,
    metrics: dict,
    epoch: int,
    wandb_run=None,
) -> None:
    """Write epoch-level metrics to TensorBoard and optionally WandB."""
    for k, v in metrics.items():
        tb_writer.add_scalar(k, v, epoch)
    if wandb_run is not None:
        try:
            import wandb
            wandb.log(metrics, step=epoch)
        except Exception:
            pass
