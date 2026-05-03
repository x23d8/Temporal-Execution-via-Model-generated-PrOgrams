"""
losses.py — Loss functions for the Hybrid Time Embedding training pipeline.
Implements wrapped torus loss, consistency loss, and the combined total loss
used across all training phases.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def wrapped_torus_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """
    Torus-wrapped MSE loss that handles periodic/cyclic differences.

    Computes (pred - true) mod 1 to account for cyclic distance,
    avoiding large gradients from predictions that are off by a full period.

    Args:
        pred: [batch] predicted values.
        true: [batch] ground truth values.

    Returns:
        Scalar mean squared wrapped error.
    """
    diff = pred - true
    diff_wrapped = diff - torch.round(diff)
    return (diff_wrapped ** 2).mean()


def consistency_loss(
    start_times: torch.Tensor,
    dur_pred: torch.Tensor,
    end_pred: torch.Tensor,
) -> torch.Tensor:
    """
    Penalizes inconsistency between predicted start + duration and end.

    Args:
        start_times: [batch] ground truth start times.
        dur_pred:    [batch] predicted durations.
        end_pred:    [batch] predicted end times (from arithmetic head).

    Returns:
        Scalar MSE of (start + duration - end_prediction).
    """
    return F.mse_loss(start_times + dur_pred, end_pred)


def total_loss(
    arith_pred: torch.Tensor,
    arith_true: torch.Tensor,
    dur_pred: torch.Tensor,
    dur_true: torch.Tensor,
    start_times: torch.Tensor,
    end_pred: torch.Tensor,
    gate_reg: torch.Tensor,
    lambda_torus: float = 0.3,
    lambda_consist: float = 0.5,
    lambda_gate: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Combined training loss: arithmetic MSE + duration (MSE + torus) + consistency + gate reg.

    Args:
        arith_pred:    [batch, 1] arithmetic predictions.
        arith_true:    [batch] arithmetic ground truth.
        dur_pred:      [batch, 1] duration predictions.
        dur_true:      [batch] duration ground truth.
        start_times:   [batch] start times for consistency term.
        end_pred:      [batch, 1] end time predictions (arithmetic head output for end).
        gate_reg:      scalar gate regularization loss from OptimalFusion.
        lambda_torus:  Weight for wrapped torus loss on duration.
        lambda_consist: Weight for start + duration = end consistency.
        lambda_gate:   Weight for gate regularization.

    Returns:
        Tuple of (total_loss_scalar, dict of component losses).
    """
    arith_pred_sq = arith_pred.squeeze()
    dur_pred_sq = dur_pred.squeeze()
    end_pred_sq = end_pred.squeeze()

    L_arith = F.mse_loss(arith_pred_sq, arith_true)
    L_dur = F.mse_loss(dur_pred_sq, dur_true) + lambda_torus * wrapped_torus_loss(dur_pred_sq, dur_true)
    L_consist = consistency_loss(start_times, dur_pred_sq, end_pred_sq)
    L_gate = lambda_gate * gate_reg

    total = L_arith + L_dur + lambda_consist * L_consist + L_gate

    return total, {
        "arith": L_arith.item(),
        "dur": L_dur.item(),
        "consist": L_consist.item(),
        "gate": L_gate.item(),
    }
