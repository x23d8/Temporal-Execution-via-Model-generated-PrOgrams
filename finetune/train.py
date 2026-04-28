#!/usr/bin/env python
"""General-purpose LLM finetuning pipeline entry point.

Usage:
    python finetune/train.py --config finetune/configs/gemma_2b_qlora.yaml
    python finetune/train.py --config finetune/configs/mistral_7b_qlora.yaml --resume auto
    python finetune/train.py --config finetune/configs/gemma_2b_lora.yaml \
        --resume finetune/outputs/run/checkpoints/ckpt_epoch_2

Kaggle (notebook cell):
    !python finetune/train.py --config finetune/configs/gemma_2b_qlora.yaml
"""
from __future__ import annotations

import argparse
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
        "--resume",
        default=None,
        help='Checkpoint to resume from. Use "auto" for latest, or a path.',
    )
    p.add_argument(
        "--output_dir",
        default=None,
        help="Override logging.output_dir in config",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from finetune.src.core.config import load_config
    from finetune.src.core.trainer import Trainer

    cfg = load_config(args.config)

    if args.resume:
        cfg.checkpoint.resume_from = args.resume
    if args.output_dir:
        cfg.logging.output_dir = args.output_dir

    print(f"[pipeline] config      : {args.config}")
    print(f"[pipeline] model       : {cfg.model.name}")
    print(f"[pipeline] peft method : {cfg.peft.method}")
    print(f"[pipeline] output_dir  : {cfg.logging.output_dir}")

    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
