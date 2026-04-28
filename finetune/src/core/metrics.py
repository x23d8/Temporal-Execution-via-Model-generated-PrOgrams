"""Per-epoch metrics tracking.

Tracked per epoch:
  train_loss            – mean CE loss over training batches
  eval_loss             – mean CE loss over eval batches
  perplexity            – exp(eval_loss)
  token_accuracy        – next-token prediction accuracy (non-pad tokens)
  token_f1              – weighted F1 over non-pad tokens
  avg_inference_ms      – mean per-sample eval inference time (ms)
  learning_rate
  status                – "running" | "early_stopped" | "completed"
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float = 0.0
    eval_loss: float = 0.0
    perplexity: float = 0.0
    token_accuracy: float = 0.0
    token_f1: float = 0.0
    avg_inference_ms: float = 0.0
    learning_rate: float = 0.0
    status: str = "running"


class MetricsTracker:
    def __init__(self) -> None:
        self.history: List[EpochMetrics] = []
        self._reset()

    def _reset(self) -> None:
        self._train_losses: List[float] = []
        self._eval_losses: List[float] = []
        self._inf_times: List[float] = []
        self._preds: List[np.ndarray] = []
        self._labels: List[np.ndarray] = []

    def reset_epoch(self) -> None:
        self._reset()

    # ── Accumulate ────────────────────────────────────────────────────────

    def add_train_loss(self, loss: float) -> None:
        self._train_losses.append(loss)

    def add_eval_batch(
        self,
        loss: float,
        logits_np: np.ndarray,   # [B, T, V]
        labels_np: np.ndarray,   # [B, T]
        inference_ms_per_sample: float,
    ) -> None:
        self._eval_losses.append(loss)
        self._inf_times.append(inference_ms_per_sample)
        preds = logits_np.argmax(axis=-1).flatten()          # [B*T]
        labs  = labels_np.flatten()                           # [B*T]
        mask  = labs != -100
        if mask.any():
            self._preds.append(preds[mask])
            self._labels.append(labs[mask])

    # ── Compute ───────────────────────────────────────────────────────────

    def compute(self, epoch: int, lr: float, status: str = "running") -> EpochMetrics:
        train_loss = float(np.mean(self._train_losses)) if self._train_losses else 0.0
        eval_loss  = float(np.mean(self._eval_losses))  if self._eval_losses  else 0.0
        ppl        = math.exp(min(eval_loss, 20.0))     # cap to avoid overflow
        inf_ms     = float(np.mean(self._inf_times))    if self._inf_times     else 0.0

        acc, f1 = 0.0, 0.0
        if self._preds:
            from sklearn.metrics import accuracy_score, f1_score
            all_p = np.concatenate(self._preds)
            all_l = np.concatenate(self._labels)
            # Subsample to 100k tokens max for speed
            if len(all_p) > 100_000:
                idx = np.random.choice(len(all_p), 100_000, replace=False)
                all_p, all_l = all_p[idx], all_l[idx]
            acc = float(accuracy_score(all_l, all_p))
            f1  = float(f1_score(all_l, all_p, average="weighted", zero_division=0))

        m = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            eval_loss=eval_loss,
            perplexity=ppl,
            token_accuracy=acc,
            token_f1=f1,
            avg_inference_ms=inf_ms,
            learning_rate=lr,
            status=status,
        )
        self.history.append(m)
        return m

    # ── Helpers for early stopping ────────────────────────────────────────

    def last(self, key: str) -> float:
        if not self.history:
            return float("inf")
        return float(getattr(self.history[-1], key))

    def best(self, key: str) -> float:
        if not self.history:
            return float("inf")
        vals = [getattr(m, key) for m in self.history]
        return min(vals) if "loss" in key else max(vals)


class EarlyStopper:
    def __init__(self, patience: int, min_delta: float, monitor: str) -> None:
        self.patience   = patience
        self.min_delta  = min_delta
        self.monitor    = monitor
        self._higher_is_better = monitor not in ("eval_loss", "train_loss")
        self.best_val   = float("-inf") if self._higher_is_better else float("inf")
        self.counter    = 0
        self.triggered  = False

    def step(self, value: float) -> bool:
        if self._higher_is_better:
            improved = value > self.best_val + self.min_delta
        else:
            improved = value < self.best_val - self.min_delta

        if improved:
            self.best_val = value
            self.counter  = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
                return True
        return False

    def state(self) -> dict:
        return {"counter": self.counter, "best_val": self.best_val}

    def load_state(self, d: dict) -> None:
        self.counter  = d["counter"]
        self.best_val = d["best_val"]
