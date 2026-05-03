"""
predictor.py — Batch and single-item prediction utilities.
Wraps TemporalQAPipeline for high-throughput batch inference and
single-query prediction with confidence estimation.
"""

import time
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from .pipeline import TemporalQAPipeline
from ..src.data.dataset import TemporalQADataset
from ..src.data.collator import DataCollatorWithTimestamps


def single_predict(
    pipeline: TemporalQAPipeline,
    query: str,
    context: str = "",
    timestamps: Optional[List[float]] = None,
) -> Dict:
    """
    Run inference on a single query with timing.

    Args:
        pipeline: Initialized TemporalQAPipeline.
        query: Question text.
        context: Optional context string.
        timestamps: Optional year timestamps.

    Returns:
        Prediction dict with 'latency_ms' appended.
    """
    t0 = time.perf_counter()
    result = pipeline.predict(query, context, timestamps)
    result["latency_ms"] = (time.perf_counter() - t0) * 1000
    return result


@torch.no_grad()
def batch_predict(
    pipeline: TemporalQAPipeline,
    queries: List[str],
    contexts: Optional[List[str]] = None,
    timestamps_list: Optional[List[List[float]]] = None,
    batch_size: int = 16,
) -> List[Dict]:
    """
    Run inference on a list of queries in batches.

    Args:
        pipeline: Initialized TemporalQAPipeline.
        queries: List of question strings.
        contexts: Optional list of context strings (same length as queries).
        timestamps_list: Optional list of timestamp lists per query.
        batch_size: Number of queries per forward pass.

    Returns:
        List of prediction dicts, one per query.
    """
    contexts = contexts or [""] * len(queries)
    timestamps_list = timestamps_list or [None] * len(queries)
    results = []

    t0 = time.perf_counter()
    for i in range(0, len(queries), batch_size):
        batch_q = queries[i: i + batch_size]
        batch_c = contexts[i: i + batch_size]
        batch_t = timestamps_list[i: i + batch_size]
        for q, c, ts in zip(batch_q, batch_c, batch_t):
            results.append(pipeline.predict(q, c, ts))

    total_time = time.perf_counter() - t0
    qps = len(queries) / max(total_time, 1e-6)
    print(f"Batch inference: {len(queries)} queries in {total_time:.2f}s ({qps:.1f} queries/sec)")
    return results
