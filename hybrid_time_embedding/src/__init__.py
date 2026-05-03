"""
hybrid_time_embedding.src — Public API for the Hybrid Time Embedding system.
Exposes the key classes needed to train, evaluate, and run inference.
"""

from .utils.config import HybridConfig
from .models.full_model import HybridTemporalModel
from .data.dataset import TemporalQADataset

__all__ = [
    "HybridConfig",
    "HybridTemporalModel",
    "TemporalQADataset",
]

# Lazy imports to avoid circular dependencies at import time
def get_evaluator():
    from ..evaluate.evaluator import TemporalEvaluator
    return TemporalEvaluator

def get_pipeline():
    from ..inference.pipeline import TemporalQAPipeline
    return TemporalQAPipeline
