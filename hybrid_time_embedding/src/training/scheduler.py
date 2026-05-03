"""
scheduler.py — Learning rate schedulers for each training phase.
Provides warmup + cosine decay schedules tailored to each phase's
step count and warmup budget.
"""

import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_phase_scheduler(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """
    Linear warmup followed by cosine decay to min_lr_ratio * peak_lr.

    Args:
        optimizer: The optimizer to schedule.
        num_warmup_steps: Steps for linear warmup from 0 → peak_lr.
        num_training_steps: Total steps including warmup.
        min_lr_ratio: Fraction of peak_lr to decay to (default 0 = full decay).

    Returns:
        LambdaLR scheduler instance.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / max(1, num_warmup_steps)
        progress = float(current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)

    return LambdaLR(optimizer, lr_lambda)
