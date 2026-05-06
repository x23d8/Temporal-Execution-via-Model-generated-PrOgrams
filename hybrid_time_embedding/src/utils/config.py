"""
config.py — HybridConfig dataclass containing all hyperparameters for the
Hybrid Time Embedding system. Single source of truth for model architecture,
training phases, logging, and path configuration.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import List


@dataclass
class HybridConfig:
    # ── Model architecture ──────────────────────────
    base_model_name: str = "Qwen/Qwen2.5-7B"
    d_model: int = 3584
    n_learned_freq: int = 8
    n_random_freq: int = 16
    gate_init: float = 0.1
    gate_threshold: float = 0.05

    # ── LoRA ────────────────────────────────────────
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"]
    )
    frozen_layers: int = 4  # freeze first N transformer layers

    # ── Loss weights ────────────────────────────────
    lambda_torus: float = 0.3
    lambda_consist: float = 0.5
    lambda_gate: float = 1.0

    # ── Phase 1 ─────────────────────────────────────
    phase1_epochs: int = 2
    phase1_lr_emb: float = 1e-3
    phase1_batch_size: int = 16
    phase1_warmup_steps: int = 100

    # ── Phase 2 ─────────────────────────────────────
    phase2_epochs: int = 3
    phase2_lr_backbone: float = 2e-5
    phase2_lr_emb: float = 1e-4
    phase2_lr_heads: float = 1e-3
    phase2_batch_size: int = 8
    phase2_grad_accum: int = 8
    phase2_warmup_steps: int = 200

    # ── Phase 3 GRPO ────────────────────────────────
    phase3_lr: float = 5e-7
    phase3_n_generations: int = 8
    phase3_beta: float = 0.04
    phase3_freeze_emb_steps: int = 500
    phase3_kl_threshold: float = 2.0

    # ── Checkpointing ───────────────────────────────
    checkpoint_top_k: int = 3
    checkpoint_improvement_threshold: float = 0.01
    checkpoint_save_every_steps: int = 100

    # ── Logging ─────────────────────────────────────
    log_every_steps: int = 50
    eval_every_steps: int = 500
    gate_monitor_steps: int = 100

    # ── Paths ───────────────────────────────────────
    data_dir: str = "./data"
    output_dir: str = "./models"
    experiment_dir: str = "./experiments"
    log_dir: str = "./experiments/logs"

    # ── Hardware ────────────────────────────────────
    device: str = "cuda"
    bf16: bool = True
    tf32: bool = True
    dataloader_num_workers: int = 4
    seed: int = 42

    def save(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "HybridConfig":
        """Load config from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)
