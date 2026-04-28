"""CLI entry-point cho fine-tuning.

Usage:
    python finetune/scripts/run_finetune.py --config finetune/configs/lora_multitask.yaml

Output:
    finetune/outputs/<experiment>/
        ├── adapter_model/   (LoRA weights nếu use_lora=True)
        ├── trainer_state.json
        └── eval_logprob.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Repo root vào sys.path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from finetune.src.config import FinetuneConfig, load_finetune_config
from finetune.src.data.collator import DualPromptCollator
from finetune.src.data.dataset import build_datasets
from finetune.src.prompts.dual_prompt import DualPromptBuilder
from finetune.src.trainer.logprob import logprob_evaluate
from finetune.src.trainer.multitask_trainer import MultiTaskTrainer


def _load_model_and_tokenizer(cfg: FinetuneConfig):
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map.get(cfg.dtype, torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    if cfg.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            bias=cfg.lora_bias,
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    return model, tokenizer


def main(cfg: FinetuneConfig) -> None:
    from src.utils.seed import set_seed
    set_seed(cfg.seed)

    print(f"[finetune] model={cfg.model_name}  prompt_mode={cfg.prompt_mode}  "
          f"logprob_eval={cfg.logprob_eval}  use_lora={cfg.use_lora}")

    model, tokenizer = _load_model_and_tokenizer(cfg)

    prompt_builder = DualPromptBuilder(
        tokenizer=tokenizer,
        max_seq_len=cfg.max_seq_len,
        seed=cfg.seed,
    )
    train_ds, eval_ds = build_datasets(cfg, prompt_builder)

    collator = DualPromptCollator(
        pad_token_id=tokenizer.pad_token_id,
        max_seq_len=cfg.max_seq_len,
    )

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_batch_size,
        per_device_eval_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        logging_steps=cfg.logging_steps,
        load_best_model_at_end=True,
        bf16=(cfg.dtype == "bfloat16"),
        fp16=(cfg.dtype == "float16"),
        dataloader_num_workers=cfg.dataloader_num_workers,
        seed=cfg.seed,
        report_to="none",
    )

    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        lambda_gen=cfg.lambda_gen,
        lambda_cls=cfg.lambda_cls,
    )

    trainer.train()

    # ── Lưu model / adapter ───────────────────────────────────────────────
    out_path = Path(cfg.output_dir)
    if cfg.use_lora:
        model.save_pretrained(out_path / "adapter_model")
        tokenizer.save_pretrained(out_path / "adapter_model")
        print(f"[finetune] LoRA adapter saved → {out_path / 'adapter_model'}")
    else:
        model.save_pretrained(out_path / "full_model")
        tokenizer.save_pretrained(out_path / "full_model")

    # ── Log-prob evaluation ───────────────────────────────────────────────
    if cfg.logprob_eval:
        device = next(model.parameters()).device
        # Lấy raw samples từ eval_ds để pass vào logprob_evaluate
        print("[finetune] running log-prob evaluation on eval set...")
        from finetune.src.data.dataset import build_datasets as _bd, MultiTaskDataset
        import random as _rand
        # Rebuild raw eval samples (tránh overhead của MultiTaskDataset đã expand)
        _rng = _rand.Random(cfg.seed)
        from src.data.registry import load_dataset as _ld
        eval_samples_raw = []
        for ds_name in cfg.datasets:
            kw = {}
            if cfg.max_samples_per_dataset is not None:
                kw["max_samples"] = cfg.max_samples_per_dataset
            ss = _ld(ds_name, **kw)
            _rng.shuffle(ss)
            n_eval = max(1, int(len(ss) * cfg.eval_split))
            eval_samples_raw.extend(ss[:n_eval])

        metrics = logprob_evaluate(model, tokenizer, eval_samples_raw, prompt_builder, device)
        print(f"[finetune] log-prob metrics: {metrics}")
        with open(out_path / "eval_logprob.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to finetune YAML config")
    args = ap.parse_args()
    cfg = load_finetune_config(args.config)
    main(cfg)
