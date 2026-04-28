#!/usr/bin/env python
"""General-purpose LLM finetuning pipeline entry point.

Usage:
    python finetune/train.py --config finetune/configs/gemma_2b_qlora.yaml
    python finetune/train.py --config finetune/configs/mistral_7b_qlora.yaml --resume auto
    python finetune/train.py --config finetune/configs/gemma_2b_lora.yaml \
        --resume finetune/outputs/run/checkpoints/ckpt_epoch_2

    # Override model(s) without editing the YAML:
    python finetune/train.py --config finetune/configs/gemma4_full.yaml \
        --models google/gemma-2-2b-it
    python finetune/train.py --config finetune/configs/gemma4_full.yaml \
        --models google/gemma-2-2b-it,mistralai/Mistral-7B-v0.1

Kaggle (notebook cell):
    !python finetune/train.py --config finetune/configs/gemma_2b_qlora.yaml
"""
from __future__ import annotations

import os
os.environ.setdefault("HF_HOME", "D:/cache")

import argparse
import copy
import re
import sys
from pathlib import Path

# Make repo root importable regardless of cwd
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Modular LLM Finetuning Pipeline")
    p.add_argument("--config", required=True, help="Path to YAML config file")
    p.add_argument(
        "--models",
        default=None,
        help=(
            "Comma-separated model IDs to finetune (overrides model.name in config). "
            "E.g. --models google/gemma-2-2b-it,mistralai/Mistral-7B-v0.1"
        ),
    )
    p.add_argument(
        "--resume",
        default=None,
        help='Checkpoint to resume from. Use "auto" for latest, or a path.',
    )
    p.add_argument(
        "--output_dir",
        default=None,
        help="Override logging.output_dir in config (auto-suffixed per model when --models is used)",
    )
    return p.parse_args()


def _model_slug(model_id: str) -> str:
    """Convert a HuggingFace model ID to a filesystem-safe slug."""
    return re.sub(r"[^a-zA-Z0-9._-]", "_", model_id)


def _run_single(cfg, model_id: str | None, base_output_dir: str, multi: bool) -> None:
    from finetune.src.core.trainer import Trainer

    run_cfg = copy.deepcopy(cfg)

    if model_id is not None:
        run_cfg.model.name = model_id
        run_cfg.model.tokenizer = model_id

    if multi:
        slug = _model_slug(run_cfg.model.name)
        run_cfg.logging.output_dir = str(Path(base_output_dir) / slug)

    print(f"\n{'='*60}")
    print(f"[pipeline] model       : {run_cfg.model.name}")
    print(f"[pipeline] peft method : {run_cfg.peft.method}")
    print(f"[pipeline] output_dir  : {run_cfg.logging.output_dir}")
    print(f"{'='*60}")

    Trainer(run_cfg).train()


def main() -> None:
    args = parse_args()

    from finetune.src.core.config import load_config

    cfg = load_config(args.config)

    if args.resume:
        cfg.checkpoint.resume_from = args.resume
    if args.output_dir:
        cfg.logging.output_dir = args.output_dir

    print(f"[pipeline] config      : {args.config}")

    model_ids: list[str | None]
    if args.models:
        model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        model_ids = [None]  # None → use whatever is in the config

    multi = len(model_ids) > 1
    base_output_dir = cfg.logging.output_dir

    for model_id in model_ids:
        _run_single(cfg, model_id, base_output_dir, multi)


if __name__ == "__main__":
    main()
