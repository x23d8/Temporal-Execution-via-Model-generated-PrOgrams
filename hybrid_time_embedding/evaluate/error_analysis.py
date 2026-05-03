"""
error_analysis.py — Buckets prediction errors by magnitude, time period,
and subtask. Identifies worst predictions and systematic biases.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


# Error magnitude buckets (in years)
ERROR_BUCKETS = [
    ("exact", 0, 0),
    ("0-1yr", 0, 1),
    ("1-5yr", 1, 5),
    ("5-10yr", 5, 10),
    (">10yr", 10, float("inf")),
]

# Historical period buckets
PERIOD_BUCKETS = [
    ("ancient", float("-inf"), 1000),
    ("medieval", 1000, 1800),
    ("modern", 1800, 2000),
    ("contemporary", 2000, float("inf")),
]


def bucket_by_magnitude(
    predictions: List[float],
    ground_truths: List[float],
    year_scale: float = 2100.0,
) -> Dict[str, Dict]:
    """
    Group errors into magnitude buckets.

    Args:
        predictions: Model predictions (normalized [0,1]).
        ground_truths: Ground truth values (normalized [0,1]).
        year_scale: Scale factor to convert normalized values to years.

    Returns:
        Dict mapping bucket name to {count, fraction, mean_error}.
    """
    preds = np.array(predictions) * year_scale
    truths = np.array(ground_truths) * year_scale
    abs_errors = np.abs(preds - truths)
    n = len(abs_errors)

    results = {}
    for name, lo, hi in ERROR_BUCKETS:
        if hi == 0:
            mask = abs_errors == 0
        else:
            mask = (abs_errors > lo) & (abs_errors <= hi)
        count = int(mask.sum())
        results[name] = {
            "count": count,
            "fraction": count / max(n, 1),
            "mean_error": float(abs_errors[mask].mean()) if count > 0 else 0.0,
        }
    return results


def bucket_by_time_period(
    predictions: List[float],
    ground_truths: List[float],
    year_scale: float = 2100.0,
) -> Dict[str, Dict]:
    """
    Group errors by the historical period of the ground truth.

    Args:
        predictions: Model predictions (normalized [0,1]).
        ground_truths: Ground truth values (normalized [0,1]).
        year_scale: Scale factor for converting to years.

    Returns:
        Dict mapping period name to {count, mae, exact_match}.
    """
    preds = np.array(predictions) * year_scale
    truths = np.array(ground_truths) * year_scale
    abs_errors = np.abs(preds - truths)

    results = {}
    for name, lo, hi in PERIOD_BUCKETS:
        mask = (truths > lo) & (truths <= hi)
        count = int(mask.sum())
        results[name] = {
            "count": count,
            "mae": float(abs_errors[mask].mean()) if count > 0 else 0.0,
            "exact_match": float((abs_errors[mask] < 0.5).mean()) if count > 0 else 0.0,
        }
    return results


def worst_predictions(
    predictions: List[float],
    ground_truths: List[float],
    item_ids: Optional[List[str]] = None,
    n: int = 10,
    year_scale: float = 2100.0,
) -> List[Dict]:
    """
    Return the N worst predictions by absolute error.

    Args:
        predictions: Model predictions (normalized).
        ground_truths: Ground truth values (normalized).
        item_ids: Optional list of item IDs for identification.
        n: Number of worst examples to return.
        year_scale: Scale for converting to years.

    Returns:
        List of dicts with pred, truth, error, and optional id.
    """
    preds = np.array(predictions) * year_scale
    truths = np.array(ground_truths) * year_scale
    abs_errors = np.abs(preds - truths)
    worst_idx = np.argsort(abs_errors)[::-1][:n]

    results = []
    for i in worst_idx:
        entry = {
            "pred": float(preds[i]),
            "truth": float(truths[i]),
            "error": float(abs_errors[i]),
        }
        if item_ids:
            entry["id"] = item_ids[i]
        results.append(entry)
    return results


def full_error_analysis(
    results: Dict,
    year_scale: float = 2100.0,
) -> Dict:
    """
    Run all error analyses on TemporalEvaluator results.

    Args:
        results: Output dict from TemporalEvaluator.evaluate().
        year_scale: Scale for year conversion.

    Returns:
        Dict with magnitude_buckets, period_buckets, worst_predictions keys.
    """
    raw = results.get("raw", {})
    preds = raw.get("arith_preds", [])
    truths = raw.get("arith_truths", [])

    if not preds:
        return {}

    return {
        "magnitude_buckets": bucket_by_magnitude(preds, truths, year_scale),
        "period_buckets": bucket_by_time_period(preds, truths, year_scale),
        "worst_predictions": worst_predictions(preds, truths, n=10, year_scale=year_scale),
    }
