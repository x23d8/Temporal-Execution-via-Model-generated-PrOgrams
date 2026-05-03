"""
metrics.py — Evaluation metrics for temporal QA tasks.
Computes MAE, exact match, within-N-year accuracy, consistency rate,
median AE, and 90th percentile AE.
"""

from typing import Dict, List, Optional
import numpy as np


def compute_metrics(
    predictions: List[float],
    ground_truths: List[float],
    start_times: Optional[List[float]] = None,
    subtask: str = "arithmetic",
) -> Dict[str, float]:
    """
    Compute all evaluation metrics for temporal QA predictions.

    Args:
        predictions: Model predicted values (years or durations).
        ground_truths: Ground truth values.
        start_times: Optional start times for consistency metric (duration subtask).
        subtask: One of "arithmetic" or "duration".

    Returns:
        Dict with keys: mae, exact_match, within_1yr, within_5yr,
        consistency_rate, median_ae, p90_ae.
    """
    preds = np.array(predictions, dtype=np.float32)
    truths = np.array(ground_truths, dtype=np.float32)
    abs_errors = np.abs(preds - truths)

    mae = float(np.mean(abs_errors))
    exact_match = float(np.mean(np.round(preds) == np.round(truths)))
    within_1yr = float(np.mean(abs_errors <= 1.0))
    within_5yr = float(np.mean(abs_errors <= 5.0))
    median_ae = float(np.median(abs_errors))
    p90_ae = float(np.percentile(abs_errors, 90))

    consistency_rate = 0.0
    if start_times is not None and subtask == "duration":
        starts = np.array(start_times, dtype=np.float32)
        end_pred = starts + preds
        end_true = starts + truths
        consist_errors = np.abs(end_pred - end_true)
        consistency_rate = float(np.mean(consist_errors <= 1.0))

    return {
        "mae": mae,
        "exact_match": exact_match,
        "within_1yr": within_1yr,
        "within_5yr": within_5yr,
        "consistency_rate": consistency_rate,
        "median_ae": median_ae,
        "p90_ae": p90_ae,
    }


def compute_mae(predictions: List[float], ground_truths: List[float]) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(np.array(predictions) - np.array(ground_truths))))


def compute_exact_match(predictions: List[float], ground_truths: List[float]) -> float:
    """Fraction where rounded prediction equals rounded ground truth."""
    p = np.round(np.array(predictions))
    t = np.round(np.array(ground_truths))
    return float(np.mean(p == t))


def compute_consistency_rate(
    start_times: List[float],
    durations_pred: List[float],
    end_times: List[float],
    tolerance: float = 1.0,
) -> float:
    """Fraction where start + duration ≈ end within tolerance."""
    starts = np.array(start_times)
    durs = np.array(durations_pred)
    ends = np.array(end_times)
    return float(np.mean(np.abs(starts + durs - ends) <= tolerance))
