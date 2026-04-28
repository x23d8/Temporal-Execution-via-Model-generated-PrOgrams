"""FinetuneConfig — cấu hình toàn bộ pipeline fine-tuning.

Load từ YAML: load_finetune_config(path) -> FinetuneConfig
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FinetuneConfig:
    # ── Model ──────────────────────────────────────────────────────────────
    model_name: str = "Qwen/Qwen3.5-9B"
    dtype: str = "bfloat16"          # "float16" | "bfloat16" | "float32"

    # ── Datasets (tên trong src/data/registry.DATASET_LOADERS) ────────────
    datasets: list[str] = field(default_factory=lambda: [
        "bigbench_date",
        "udst_duration",
        "vlsp_date",
        "vlsp_duration",
    ])
    max_samples_per_dataset: int | None = None   # None = toàn bộ dataset
    eval_split: float = 0.1                      # tỉ lệ tách val set

    # ── Task weights (scale loss theo task) ───────────────────────────────
    task_weights: dict[str, float] = field(default_factory=lambda: {
        "date_arith": 1.0,
        "duration": 1.0,
    })

    # ── Dual-Prompt ────────────────────────────────────────────────────────
    prompt_mode: str = "dual"    # "single" | "dual"
    #   "single": chỉ P_gen (generative CE loss)
    #   "dual"  : P_gen + P_cls (thêm classification CE loss trên label token)
    lambda_gen: float = 1.0      # trọng số L_gen
    lambda_cls: float = 0.5      # trọng số L_cls (chỉ dùng khi prompt_mode=dual)

    # ── Log-prob ──────────────────────────────────────────────────────────
    logprob_eval: bool = True
    # Dùng log P("yes") vs log P("no") thay vì greedy generation
    # cho evaluation của duration tasks → không có parse error

    # ── Training ──────────────────────────────────────────────────────────
    output_dir: str = "finetune/outputs"
    max_seq_len: int = 512
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    num_train_epochs: int = 3
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    eval_steps: int = 200
    save_steps: int = 200
    logging_steps: int = 20
    seed: int = 42
    dataloader_num_workers: int = 0    # 0 = safe trên Windows/Colab

    # ── LoRA (PEFT) ────────────────────────────────────────────────────────
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
    ])
    lora_bias: str = "none"            # "none" | "all" | "lora_only"


def load_finetune_config(path: str | Path) -> FinetuneConfig:
    with open(path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    return FinetuneConfig(**{k: v for k, v in raw.items() if hasattr(FinetuneConfig, k)})
