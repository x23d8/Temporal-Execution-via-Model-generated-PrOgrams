"""
evaluator.py — TemporalEvaluator: full evaluation suite for the Hybrid
Temporal Model. Runs inference on a DataLoader and returns per-subtask
and overall metrics.
"""

from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..src.models.full_model import HybridTemporalModel
from ..src.utils.metrics import compute_metrics


class TemporalEvaluator:
    """
    Full evaluation suite for HybridTemporalModel.

    Runs the model on a DataLoader and computes metrics separately for
    date_arithmetic and date_duration subtasks, plus combined overall metrics.

    Args:
        model: Trained HybridTemporalModel.
        device: Torch device string.
    """

    def __init__(self, model: HybridTemporalModel, device: str = "cuda") -> None:
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, Dict]:
        """
        Run evaluation on a full DataLoader.

        Args:
            dataloader: DataLoader yielding batches from TemporalQADataset.

        Returns:
            Dict with keys "arithmetic", "duration", "overall", each containing
            a metrics dict from compute_metrics().
        """
        self.model.eval()
        arith_preds, arith_truths = [], []
        dur_preds, dur_truths = [], []
        start_times_all = []
        subtask_flags = []

        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            arith_pred, dur_pred, _ = self.model(
                batch["input_ids"], batch["attention_mask"], batch["timestamps"]
            )
            arith_preds.extend(arith_pred.squeeze().cpu().tolist())
            arith_truths.extend(batch["arith_labels"].cpu().tolist())
            dur_preds.extend(dur_pred.squeeze().cpu().tolist())
            dur_truths.extend(batch["dur_labels"].cpu().tolist())
            start_times_all.extend(batch["start_times"].cpu().tolist())
            subtask_flags.extend(batch["subtask_mask"].cpu().tolist())

        # Split by subtask
        arith_idx = [i for i, m in enumerate(subtask_flags) if m == 1.0]
        dur_idx = [i for i, m in enumerate(subtask_flags) if m == 0.0]

        def pick(lst, idx):
            return [lst[i] for i in idx]

        arith_metrics = compute_metrics(
            pick(arith_preds, arith_idx), pick(arith_truths, arith_idx), subtask="arithmetic"
        ) if arith_idx else {}

        dur_metrics = compute_metrics(
            pick(dur_preds, dur_idx), pick(dur_truths, dur_idx),
            pick(start_times_all, dur_idx), subtask="duration"
        ) if dur_idx else {}

        overall_metrics = compute_metrics(
            arith_preds, arith_truths, start_times_all, subtask="arithmetic"
        )

        return {
            "arithmetic": arith_metrics,
            "duration": dur_metrics,
            "overall": overall_metrics,
            "raw": {
                "arith_preds": arith_preds,
                "arith_truths": arith_truths,
                "dur_preds": dur_preds,
                "dur_truths": dur_truths,
                "start_times": start_times_all,
            },
        }
