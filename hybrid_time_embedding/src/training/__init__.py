"""training — Phase-aware trainer, losses, callbacks, and schedulers."""

from .losses import wrapped_torus_loss, consistency_loss, total_loss
from .trainer import PhaseAwareTrainer, compute_reward
from .callbacks import SmartCheckpointSaver, GateMonitorCallback, MetricCallback
from .scheduler import get_phase_scheduler

__all__ = [
    "wrapped_torus_loss", "consistency_loss", "total_loss",
    "PhaseAwareTrainer", "compute_reward",
    "SmartCheckpointSaver", "GateMonitorCallback", "MetricCallback",
    "get_phase_scheduler",
]
