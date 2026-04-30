"""Supervised fine-tuning (LoRA / QLoRA) for Gemma-4-E4B-it on temporal-reasoning tasks.

Adapted from duc_temporal/src/training/sft.py for Gemma-4-E4B-it:
  - Model: google/gemma-4-E4B-it (~4B params, MatFormer efficient architecture)
  - System prompt merged into first user turn (Gemma chat template convention)
  - Response template: '<start_of_turn>model\\n' for completion-only loss masking
  - No enable_thinking (Gemma does not support thinking mode)
  - Optional QLoRA (load_in_4bit=True) for T4/P100 (16 GB VRAM budget)
  - Default: LoRA bf16 for A100 (40 GB) — needs ~10 GB VRAM for 4B params

Notebook usage:
  import yaml
  from src.training.sft import SFTRunConfig, train_sft
  raw = yaml.safe_load(open("configs/sft_gemma4_e4b_lora.yaml"))
  cfg = SFTRunConfig(**raw)
  adapter_path = train_sft(cfg)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..data.registry import load_dataset
from ..data.schema import Sample
from ..utils.seed import set_seed
from .data import (
    CompletionOnlyCollator,
    resolve_assistant_response_template,
    samples_to_chat_dataset,
    split_train_val,
)


@dataclass
class SFTRunConfig:
    # ── Model ────────────────────────────────────────────────────────────────
    model_name: str = "google/gemma-4-E4B"
    dtype: str = "bfloat16"           # A100: bfloat16 | T4: float16
    load_in_4bit: bool = False         # True → QLoRA (saves ~4× VRAM)

    # ── Data ─────────────────────────────────────────────────────────────────
    dataset: str = "vlsp_date"
    dataset_path: str | None = None
    train_pool_start: int = 1500       # rows 0..1499 = Phase-1 eval, off-limits
    train_pool_size: int = 1500        # rows 1500..2999
    val_ratio: float = 0.1
    shuffle_seed: int = 42

    # ── Output ───────────────────────────────────────────────────────────────
    output_dir: str = "checkpoints/gemma4_e4b_lora_vlsp_date"

    # ── LoRA ─────────────────────────────────────────────────────────────────
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )

    # ── Training hyperparameters ──────────────────────────────────────────────
    num_epochs: int = 3
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8   # effective batch = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"
    weight_decay: float = 0.0
    max_seq_length: int = 256

    # Precision — mutually exclusive; bf16 for A100, fp16 for T4
    bf16: bool = True
    fp16: bool = False
    tf32: bool = True   # A100 TF32 tensor cores; ignored on T4 (no-op)

    # ── Logging / checkpointing ───────────────────────────────────────────────
    logging_steps: int = 10
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    report_to: str = "none"

    seed: int = 42


def _load_train_pool(cfg: SFTRunConfig) -> list[Sample]:
    """Load enough rows to slice the train pool, return rows [start:start+size]."""
    needed = cfg.train_pool_start + cfg.train_pool_size
    kwargs: dict[str, Any] = {"max_samples": needed}
    if cfg.dataset_path:
        kwargs["path"] = cfg.dataset_path
    samples = load_dataset(cfg.dataset, **kwargs)
    pool = samples[cfg.train_pool_start : cfg.train_pool_start + cfg.train_pool_size]
    if not pool:
        raise RuntimeError(
            f"Train pool empty: dataset={cfg.dataset!r} has only {len(samples)} rows, "
            f"cannot slice [{cfg.train_pool_start}:{cfg.train_pool_start + cfg.train_pool_size}]"
        )
    return pool


def _tokenize_text_dataset(ds, tokenizer, max_length: int):
    def _fn(example):
        enc = tokenizer(
            example["text"],
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,  # chat_template already inserted special tokens
        )
        return {"input_ids": enc["input_ids"]}

    return ds.map(_fn, batched=False, remove_columns=ds.column_names)


def train_sft(cfg: SFTRunConfig) -> str:
    """Run SFT, save LoRA adapter to cfg.output_dir, return that path."""
    set_seed(cfg.seed)

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    # ── 1. Data ───────────────────────────────────────────────────────────────
    pool = _load_train_pool(cfg)
    train_samples, val_samples = split_train_val(
        pool, val_ratio=cfg.val_ratio, seed=cfg.shuffle_seed
    )
    print(f"[sft] pool={len(pool)} → train={len(train_samples)}  val={len(val_samples)}")

    # ── 2. Tokenizer ──────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"  # Gemma standard

    # Base model tokenizers (non -it) ship without a chat_template.
    # Borrow it from the instruction-tuned variant so apply_chat_template works.
    if not getattr(tokenizer, "chat_template", None):
        it_name = cfg.model_name if cfg.model_name.endswith("-it") else cfg.model_name + "-it"
        print(f"[sft] no chat_template on tokenizer — borrowing from {it_name}")
        _it_tok = AutoTokenizer.from_pretrained(it_name)
        tokenizer.chat_template = _it_tok.chat_template

    # ── 3. Base model ─────────────────────────────────────────────────────────
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(cfg.dtype, torch.bfloat16)
    model_kwargs: dict[str, Any] = {"device_map": "auto"}

    if cfg.load_in_4bit:
        from peft import prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )
        print(f"[sft] loading {cfg.model_name} in 4-bit QLoRA mode...")
        model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_kwargs)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model_kwargs["torch_dtype"] = torch_dtype
        print(f"[sft] loading {cfg.model_name} dtype={cfg.dtype}...")
        model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_kwargs)

    model.config.use_cache = False  # required for gradient_checkpointing

    # ── 4. Apply LoRA ─────────────────────────────────────────────────────────
    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    if hasattr(model, "enable_input_require_grads"):
        # Required so gradients flow through input embeddings with gradient_checkpointing.
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    # ── 5. Build chat-format datasets + tokenize ──────────────────────────────
    train_text_ds = samples_to_chat_dataset(train_samples, tokenizer)
    val_text_ds   = samples_to_chat_dataset(val_samples,   tokenizer)
    train_ds = _tokenize_text_dataset(train_text_ds, tokenizer, cfg.max_seq_length)
    val_ds   = _tokenize_text_dataset(val_text_ds,   tokenizer, cfg.max_seq_length)

    # ── 6. Completion-only collator (mask prompt tokens) ──────────────────────
    response_template = resolve_assistant_response_template(tokenizer)
    print(f"[sft] response_template (loss-mask boundary) = {response_template!r}")
    collator = CompletionOnlyCollator(
        tokenizer=tokenizer,
        response_template=response_template,
    )

    # ── 7. TrainingArguments — dynamic kwargs for transformers version compat ──
    import inspect

    targs_kwargs: dict[str, Any] = dict(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.per_device_batch_size,
        per_device_eval_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler,
        weight_decay=cfg.weight_decay,
        bf16=cfg.bf16,
        fp16=cfg.fp16,
        tf32=cfg.tf32,
        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        report_to=cfg.report_to,
        seed=cfg.seed,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
    )
    ta_fields = set(inspect.signature(TrainingArguments).parameters)
    if "eval_strategy" in ta_fields:
        targs_kwargs["eval_strategy"] = cfg.eval_strategy
    else:
        targs_kwargs["evaluation_strategy"] = cfg.eval_strategy
    targs = TrainingArguments(**targs_kwargs)

    # ── 8. Trainer — compat with tokenizer= vs processing_class= ─────────────
    trainer_kwargs: dict[str, Any] = dict(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )
    trainer_params = set(inspect.signature(Trainer.__init__).parameters)
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    # ── 9. Train + save adapter only (not merged) ─────────────────────────────
    trainer.train()
    out = Path(cfg.output_dir)
    trainer.model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"[sft] adapter saved → {out.resolve()}")
    return str(out)
