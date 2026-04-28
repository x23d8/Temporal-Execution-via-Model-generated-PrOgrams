"""MultiTaskTrainer — HuggingFace Trainer mở rộng cho dual-prompt + task weights.

compute_loss():
    loss = λ_gen * L_gen + λ_cls * L_cls   (nếu có cls batch)

Task weights nhân vào loss để cân bằng giữa date_arith và duration.
Sử dụng flat batch keys từ DualPromptCollator (không lồng nhau):
    gen_input_ids, gen_attention_mask, gen_labels
    cls_input_ids, cls_attention_mask, cls_labels  (optional)
    task, task_weight
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import Trainer


class MultiTaskTrainer(Trainer):
    """Trainer hỗ trợ dual-prompt loss và task-level loss weighting."""

    def __init__(
        self,
        *args: Any,
        lambda_gen: float = 1.0,
        lambda_cls: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.lambda_gen = lambda_gen
        self.lambda_cls = lambda_cls

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        **kwargs: Any,
    ):
        task_weight = inputs.get("task_weight")  # [batch]

        # ── Generation loss ───────────────────────────────────────────────
        gen_out = model(
            input_ids      = inputs["gen_input_ids"],
            attention_mask = inputs["gen_attention_mask"],
            labels         = inputs["gen_labels"],
        )
        loss_gen = gen_out.loss

        # Task-level weighting: scale bằng mean weight của batch
        if task_weight is not None:
            loss_gen = loss_gen * task_weight.mean().to(loss_gen.device)

        loss = self.lambda_gen * loss_gen

        # ── Classification loss (dual-prompt) ─────────────────────────────
        if "cls_input_ids" in inputs:
            cls_out = model(
                input_ids      = inputs["cls_input_ids"],
                attention_mask = inputs["cls_attention_mask"],
                labels         = inputs["cls_labels"],
            )
            loss_cls = cls_out.loss
            if task_weight is not None:
                loss_cls = loss_cls * task_weight.mean().to(loss_cls.device)
            loss = loss + self.lambda_cls * loss_cls

        return (loss, gen_out) if return_outputs else loss

    def _remove_unused_columns(
        self,
        dataset: Any,
        description: str | None = None,
    ) -> Any:
        # Override để không xoá các custom keys (gen_*, cls_*, task, task_weight)
        return dataset
