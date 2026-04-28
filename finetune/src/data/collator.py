"""DualPromptCollator — pad gen batch và cls batch riêng biệt.

HuggingFace tokenizer.pad() pad input_ids bằng pad_token_id và attention_mask bằng 0.
Labels cần pad bằng -100 (không tính loss) — xử lý thủ công.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def _pad_sequence(seqs: list[list[int]], pad_value: int) -> list[list[int]]:
    max_len = max(len(s) for s in seqs)
    return [s + [pad_value] * (max_len - len(s)) for s in seqs]


@dataclass
class DualPromptCollator:
    """DataCollator cho MultiTaskDataset.

    Trả về batch dict với các key flat (không lồng nhau) để MultiTaskTrainer
    có thể gọi model(**gen_inputs) và model(**cls_inputs) độc lập.
    """

    pad_token_id: int
    max_seq_len: int = 512

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        def _collect(key: str) -> list[list[int]]:
            return [f[key] for f in features]

        def _to_tensor(seqs: list[list[int]], pad: int) -> torch.Tensor:
            padded = _pad_sequence(seqs, pad)
            # Truncate nếu vượt max_seq_len
            padded = [s[:self.max_seq_len] for s in padded]
            return torch.tensor(padded, dtype=torch.long)

        batch: dict[str, Any] = {}

        # ── Generation batch ──────────────────────────────────────────────
        batch["gen_input_ids"]      = _to_tensor(_collect("gen_input_ids"), self.pad_token_id)
        batch["gen_attention_mask"] = _to_tensor(_collect("gen_attention_mask"), 0)
        batch["gen_labels"]         = _to_tensor(_collect("gen_labels"), -100)

        # ── Classification batch (optional) ───────────────────────────────
        if "cls_input_ids" in features[0]:
            batch["cls_input_ids"]      = _to_tensor(_collect("cls_input_ids"), self.pad_token_id)
            batch["cls_attention_mask"] = _to_tensor(_collect("cls_attention_mask"), 0)
            batch["cls_labels"]         = _to_tensor(_collect("cls_labels"), -100)

        # ── Metadata ──────────────────────────────────────────────────────
        batch["task"]        = [f["task"] for f in features]
        batch["task_weight"] = torch.tensor(
            [f["task_weight"] for f in features], dtype=torch.float32
        )

        return batch
