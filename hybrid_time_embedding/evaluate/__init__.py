"""evaluate — Evaluation suite, metrics reporting, and error analysis."""

from .evaluator import TemporalEvaluator
from .metrics_report import generate_report
from .error_analysis import full_error_analysis, bucket_by_magnitude, bucket_by_time_period, worst_predictions

__all__ = [
    "TemporalEvaluator",
    "generate_report",
    "full_error_analysis",
    "bucket_by_magnitude",
    "bucket_by_time_period",
    "worst_predictions",
]
