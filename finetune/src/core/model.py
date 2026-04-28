"""Model + tokenizer + PEFT setup.

Supports:
  - "lora"  : full-precision model (fp16/bf16) with LoRA adapters
  - "qlora" : 4-bit quantized model (BitsAndBytes) with LoRA adapters
  - "none"  : full finetuning — all weights trained, no adapters
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from .config import PipelineConfig


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def load_tokenizer(cfg: PipelineConfig) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(
        cfg.model.tokenizer,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "right"
    return tok


def _base_dtype(cfg: PipelineConfig) -> torch.dtype:
    if cfg.training.bf16:
        return torch.bfloat16
    if cfg.training.fp16:
        return torch.float16
    return torch.float32


def load_model(cfg: PipelineConfig) -> AutoModelForCausalLM:
    method = cfg.peft.method.lower()
    extra_kwargs: dict = {}

    if cfg.model.attn_implementation:
        extra_kwargs["attn_implementation"] = cfg.model.attn_implementation

    # ── QLoRA: 4-bit quantized base + LoRA ───────────────────────────────
    if method == "qlora":
        qcfg = cfg.peft.qlora
        compute_dtype = _DTYPE_MAP.get(qcfg.compute_dtype, torch.float16)
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=qcfg.double_quant,
            bnb_4bit_quant_type=qcfg.quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.name,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=cfg.model.trust_remote_code,
            **extra_kwargs,
        )
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )
        model = _attach_lora(model, cfg)

    # ── LoRA: full-precision base + LoRA ─────────────────────────────────
    elif method == "lora":
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.name,
            torch_dtype=_base_dtype(cfg),
            device_map="auto",
            trust_remote_code=cfg.model.trust_remote_code,
            **extra_kwargs,
        )
        model = _attach_lora(model, cfg)

    # ── Full finetuning: no adapters, all weights trainable ───────────────
    elif method == "none":
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.name,
            torch_dtype=_base_dtype(cfg),
            device_map="auto",
            trust_remote_code=cfg.model.trust_remote_code,
            **extra_kwargs,
        )
        # Gradient checkpointing trades recomputation for activation memory —
        # essential for fitting a 4B model's backward pass on T4 (16 GB).
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"trainable params: {trainable:,} || "
            f"all params: {total:,} || "
            f"trainable%: {100 * trainable / total:.2f}"
        )

    else:
        raise ValueError(
            f"peft.method must be 'lora', 'qlora', or 'none'; got {method!r}"
        )

    return model


def _attach_lora(model, cfg: PipelineConfig):
    lora = cfg.peft.lora
    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora.r,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        target_modules=lora.target_modules,
        bias=lora.bias,
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    return model


def primary_device(model) -> torch.device:
    """Return the device of the first model parameter (input tensors go here)."""
    return next(iter(model.parameters())).device
