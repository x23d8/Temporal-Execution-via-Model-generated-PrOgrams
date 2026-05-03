"""
collator.py — DataCollatorWithTimestamps for batching TemporalQADataset items.
Tokenizes text, pads sequences, and stacks timestamp/label tensors into a
unified batch dict consumed by HybridTemporalModel.
"""

from typing import Dict, List

import torch
from transformers import PreTrainedTokenizer


class DataCollatorWithTimestamps:
    """
    Collates a list of TemporalQADataset items into a model-ready batch.

    Inserts [TIME_START] and [TIME_END] special tokens during tokenization
    and returns a unified batch dict with:
        input_ids, attention_mask, timestamps, start_times,
        end_times, arith_labels, dur_labels, subtask_mask.

    Args:
        tokenizer: Qwen2.5 tokenizer with [TIME_START]/[TIME_END] added.
        max_length: Maximum token sequence length.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 512) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: List of dicts from TemporalQADataset.__getitem__.

        Returns:
            Batch dict with tensor values on CPU. Move to device in training loop.
        """
        texts = [f["text"] for f in features]
        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        timestamps = torch.tensor([f["timestamp"] for f in features], dtype=torch.float32)
        start_times = torch.tensor([f["start_time"] for f in features], dtype=torch.float32)
        end_times = torch.tensor([f["end_time"] for f in features], dtype=torch.float32)
        arith_labels = torch.tensor([f["arith_label"] for f in features], dtype=torch.float32)
        dur_labels = torch.tensor([f["dur_label"] for f in features], dtype=torch.float32)

        # subtask_mask: 1 for date_arithmetic, 0 for date_duration
        subtask_mask = torch.tensor(
            [1.0 if f["subtask"] == "date_arithmetic" else 0.0 for f in features],
            dtype=torch.float32,
        )

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "timestamps": timestamps,
            "start_times": start_times,
            "end_times": end_times,
            "arith_labels": arith_labels,
            "dur_labels": dur_labels,
            "subtask_mask": subtask_mask,
        }
