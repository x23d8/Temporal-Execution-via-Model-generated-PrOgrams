"""Typed YAML config for the general-purpose Kaggle finetuning pipeline.

Load a config:
    cfg = load_config("configs/gemma_2b_qlora.yaml")
"""
from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ── PEFT ──────────────────────────────────────────────────────────────────────

@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    bias: str = "none"          # "none" | "all" | "lora_only"


@dataclass
class QLoRAConfig:
    bits: int = 4
    double_quant: bool = True
    quant_type: str = "nf4"     # "nf4" | "fp4"
    compute_dtype: str = "float16"  # "float16" | "bfloat16"


@dataclass
class PEFTConfig:
    method: str = "qlora"       # "lora" | "qlora" | "none" (full finetune)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    qlora: QLoRAConfig = field(default_factory=QLoRAConfig)


# ── Model ─────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    name: str = "google/gemma-2b"
    tokenizer: Optional[str] = None   # defaults to `name`
    max_length: int = 512
    trust_remote_code: bool = False
    attn_implementation: Optional[str] = None  # "flash_attention_2" | None

    def __post_init__(self) -> None:
        if self.tokenizer is None:
            self.tokenizer = self.name


# ── Training ──────────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    epochs: int = 3
    learning_rate: float = 2e-4
    batch_size: int = 4
    eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    fp16: bool = True           # T4 on Kaggle supports fp16; set bf16=True for A100
    bf16: bool = False
    optim: str = "paged_adamw_32bit"   # "paged_adamw_32bit" | "adamw_torch"
    lr_scheduler_type: str = "cosine"
    dataloader_num_workers: int = 2
    seed: int = 42


# ── Early stopping ────────────────────────────────────────────────────────────

@dataclass
class EarlyStoppingConfig:
    enabled: bool = True
    patience: int = 3
    min_delta: float = 0.001
    monitor: str = "eval_loss"  # "eval_loss" | "accuracy" | "f1"


# ── Data ──────────────────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    train_path: str = "data/train.jsonl"
    eval_path: Optional[str] = None
    format: str = "text"        # "text" | "instruction" | "chat"
    # column names
    text_column: str = "text"
    instruction_column: str = "instruction"
    input_column: str = "input"
    output_column: str = "output"
    messages_column: str = "messages"
    # alpaca-style template (only for format="instruction")
    prompt_template: str = (
        "### Instruction:\n{instruction}\n\n"
        "### Input:\n{input}\n\n"
        "### Response:\n{output}"
    )
    validation_split: float = 0.1
    max_samples: Optional[int] = None


# ── Logging ───────────────────────────────────────────────────────────────────

@dataclass
class LoggingConfig:
    output_dir: str = "finetune/outputs/run"


# ── Checkpointing ─────────────────────────────────────────────────────────────

@dataclass
class CheckpointConfig:
    enabled: bool = True
    save_total_limit: int = 3   # max epoch checkpoints to keep on disk
    resume_from: Optional[str] = None  # path or "auto" (latest)


# ── Top-level ─────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    peft: PEFTConfig = field(default_factory=PEFTConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)


# ── Loader ────────────────────────────────────────────────────────────────────

def _build(cls, raw: dict):
    import dataclasses
    known = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in raw.items() if k in known})


def load_config(path: str) -> PipelineConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = PipelineConfig()

    if "model" in raw:
        cfg.model = _build(ModelConfig, raw["model"])

    if "peft" in raw:
        p = raw["peft"]
        cfg.peft = PEFTConfig(
            method=p.get("method", "qlora"),
            lora=_build(LoRAConfig, p.get("lora", {})),
            qlora=_build(QLoRAConfig, p.get("qlora", {})),
        )

    if "training" in raw:
        cfg.training = _build(TrainingConfig, raw["training"])

    if "early_stopping" in raw:
        cfg.early_stopping = _build(EarlyStoppingConfig, raw["early_stopping"])

    if "data" in raw:
        cfg.data = _build(DataConfig, raw["data"])

    if "logging" in raw:
        cfg.logging = _build(LoggingConfig, raw["logging"])

    if "checkpoint" in raw:
        cfg.checkpoint = _build(CheckpointConfig, raw["checkpoint"])

    return cfg
