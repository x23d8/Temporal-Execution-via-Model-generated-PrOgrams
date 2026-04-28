"""MultiTaskDataset — kết hợp nhiều dataset, áp dụng DualPromptBuilder.

Luồng:
  load_dataset(name) → list[Sample]  (từ src/data/registry)
    → train/eval split ngẫu nhiên
    → MultiTaskDataset.__getitem__ gọi DualPromptBuilder.build_gen / build_cls
    → Trả về dict để DualPromptCollator xử lý tiếp
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

# Thêm repo root vào sys.path để import src/data/registry từ thư mục cha
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.registry import load_dataset as _load_dataset  # noqa: E402
from finetune.src.prompts.dual_prompt import DualPromptBuilder  # noqa: E402
from finetune.src.config import FinetuneConfig  # noqa: E402


class MultiTaskDataset(Dataset):
    """Dataset multi-task cho fine-tuning.

    Mỗi item trả về:
      {
        "gen_input_ids":      list[int],
        "gen_attention_mask": list[int],
        "gen_labels":         list[int],
        "cls_input_ids":      list[int],   # chỉ khi prompt_mode="dual"
        "cls_attention_mask": list[int],
        "cls_labels":         list[int],
        "task":               str,
        "task_weight":        float,
      }

    Với date_arith dual-prompt, mỗi sample gốc tạo ra 2 cls examples (pos+neg).
    Dataset expand các sample thành flat list để __len__ / __getitem__ đơn giản.
    """

    def __init__(
        self,
        samples: list[dict],
        prompt_builder: DualPromptBuilder,
        task_weights: dict[str, float],
        prompt_mode: str = "dual",
    ) -> None:
        self._items: list[dict] = []
        for sample in samples:
            task   = sample["task"]
            weight = task_weights.get(task, 1.0)
            gen    = prompt_builder.build_gen(sample)

            if prompt_mode == "dual":
                cls_list = prompt_builder.build_cls(sample)
                for cls_enc in cls_list:
                    self._items.append({
                        "gen_input_ids":      gen["input_ids"],
                        "gen_attention_mask": gen["attention_mask"],
                        "gen_labels":         gen["labels"],
                        "cls_input_ids":      cls_enc["input_ids"],
                        "cls_attention_mask": cls_enc["attention_mask"],
                        "cls_labels":         cls_enc["labels"],
                        "task":               task,
                        "task_weight":        weight,
                    })
            else:
                self._items.append({
                    "gen_input_ids":      gen["input_ids"],
                    "gen_attention_mask": gen["attention_mask"],
                    "gen_labels":         gen["labels"],
                    "task":               task,
                    "task_weight":        weight,
                })

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict:
        return self._items[idx]


def build_datasets(
    cfg: FinetuneConfig,
    prompt_builder: DualPromptBuilder,
) -> tuple["MultiTaskDataset", "MultiTaskDataset"]:
    """Load tất cả datasets, split train/eval, trả về (train_ds, eval_ds)."""
    rng = random.Random(cfg.seed)
    train_samples: list[dict] = []
    eval_samples:  list[dict] = []

    for ds_name in cfg.datasets:
        kwargs: dict[str, Any] = {}
        if cfg.max_samples_per_dataset is not None:
            kwargs["max_samples"] = cfg.max_samples_per_dataset

        samples = _load_dataset(ds_name, **kwargs)
        rng.shuffle(samples)

        n_eval  = max(1, int(len(samples) * cfg.eval_split))
        eval_samples.extend(samples[:n_eval])
        train_samples.extend(samples[n_eval:])

    print(
        f"[dataset] train={len(train_samples)} eval={len(eval_samples)} "
        f"across {len(cfg.datasets)} datasets"
    )

    train_ds = MultiTaskDataset(
        train_samples, prompt_builder, cfg.task_weights, cfg.prompt_mode
    )
    eval_ds = MultiTaskDataset(
        eval_samples, prompt_builder, cfg.task_weights, cfg.prompt_mode
    )
    print(
        f"[dataset] expanded: train={len(train_ds)} items  "
        f"eval={len(eval_ds)} items"
    )
    return train_ds, eval_ds
