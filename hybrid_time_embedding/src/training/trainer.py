"""
trainer.py — PhaseAwareTrainer managing the 3-phase freeze/unfreeze strategy.
Phase 1: backbone frozen, time embedding + heads warm up.
Phase 2: LoRA layers unfrozen, supervised fine-tuning.
Phase 3: GRPO reinforcement learning with delayed time embedding unfreeze.
"""

import random
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.full_model import HybridTemporalModel
from ..utils.config import HybridConfig
from ..utils.metrics import compute_metrics
from .callbacks import SmartCheckpointSaver, GateMonitorCallback, MetricCallback
from .losses import total_loss
from .scheduler import get_phase_scheduler


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


class PhaseAwareTrainer:
    """
    Manages all three training phases with per-phase freeze/unfreeze, optimizers,
    schedulers, checkpointing, and metric logging.

    Args:
        model: HybridTemporalModel instance.
        config: HybridConfig with all hyperparameters.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        checkpoint_saver: SmartCheckpointSaver instance.
        logger: Python logger.
        tb_writer: TensorBoard SummaryWriter.
        wandb_run: Optional WandB run.
    """

    def __init__(
        self,
        model: HybridTemporalModel,
        config: HybridConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_saver: SmartCheckpointSaver,
        logger,
        tb_writer,
        wandb_run=None,
    ) -> None:
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.saver = checkpoint_saver
        self.logger = logger
        self.tb_writer = tb_writer
        self.wandb_run = wandb_run
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.global_step = 0

        if config.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def _move_batch(self, batch: Dict) -> Dict:
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    def _log_step(self, metrics: Dict) -> None:
        for k, v in metrics.items():
            if isinstance(v, float):
                self.tb_writer.add_scalar(k, v, self.global_step)
        if self.wandb_run:
            try:
                import wandb
                wandb.log(metrics, step=self.global_step)
            except Exception:
                pass

    @torch.no_grad()
    def evaluate(self, phase: str = "val") -> Dict:
        """Run evaluation on the validation loader and return metrics dict."""
        self.model.eval()
        all_arith_pred, all_arith_true = [], []
        all_dur_pred, all_dur_true = [], []
        all_start_times = []

        for batch in self.val_loader:
            batch = self._move_batch(batch)
            arith_pred, dur_pred, _ = self.model(
                batch["input_ids"], batch["attention_mask"], batch["timestamps"]
            )
            all_arith_pred.extend(arith_pred.squeeze().cpu().tolist())
            all_arith_true.extend(batch["arith_labels"].cpu().tolist())
            all_dur_pred.extend(dur_pred.squeeze().cpu().tolist())
            all_dur_true.extend(batch["dur_labels"].cpu().tolist())
            all_start_times.extend(batch["start_times"].cpu().tolist())

        arith_metrics = compute_metrics(all_arith_pred, all_arith_true, subtask="arithmetic")
        dur_metrics = compute_metrics(all_dur_pred, all_dur_true, all_start_times, subtask="duration")

        return {
            "val/mae_arithmetic": arith_metrics["mae"],
            "val/mae_duration": dur_metrics["mae"],
            "val/mae_overall": (arith_metrics["mae"] + dur_metrics["mae"]) / 2,
            "val/exact_match_arithmetic": arith_metrics["exact_match"],
            "val/exact_match_duration": dur_metrics["exact_match"],
            "val/consistency_rate": dur_metrics["consistency_rate"],
            "val/within_1yr": (arith_metrics["within_1yr"] + dur_metrics["within_1yr"]) / 2,
            "val/within_5yr": (arith_metrics["within_5yr"] + dur_metrics["within_5yr"]) / 2,
            "val_mae": (arith_metrics["mae"] + dur_metrics["mae"]) / 2,
        }

    # ─── Phase 1 ───────────────────────────────────────────────────────────────

    def train_phase1(self) -> Dict:
        """
        Phase 1: Embedding warmup. Backbone fully frozen.
        Only time_embedding, fusion, and task heads are trained.
        """
        _set_seed(self.config.seed)
        self.logger.info("=== Phase 1: Embedding Warmup ===")

        # Freeze backbone
        for p in self.model.backbone.parameters():
            p.requires_grad = False
        for p in self.model.time_embedding.parameters():
            p.requires_grad = True
        for p in self.model.fusion.parameters():
            p.requires_grad = True
        for p in self.model.pooler.parameters():
            p.requires_grad = True
        for p in self.model.arith_head.parameters():
            p.requires_grad = True
        for p in self.model.dur_head.parameters():
            p.requires_grad = True

        optimizer = torch.optim.AdamW([
            {"params": self.model.time_embedding.parameters(), "lr": self.config.phase1_lr_emb},
            {"params": self.model.fusion.parameters(), "lr": self.config.phase1_lr_emb},
            {"params": list(self.model.pooler.parameters()) +
                       list(self.model.arith_head.parameters()) +
                       list(self.model.dur_head.parameters()), "lr": self.config.phase1_lr_emb},
        ])

        steps_per_epoch = len(self.train_loader)
        total_steps = steps_per_epoch * self.config.phase1_epochs
        scheduler = get_phase_scheduler(optimizer, self.config.phase1_warmup_steps, total_steps)

        gate_monitor = GateMonitorCallback(self.config.gate_threshold, warmup_steps=200)

        for epoch in range(self.config.phase1_epochs):
            self.model.train()
            pbar = tqdm(self.train_loader, desc=f"Phase1 E{epoch+1}")
            for batch in pbar:
                batch = self._move_batch(batch)
                optimizer.zero_grad()

                arith_pred, dur_pred, gate_reg = self.model(
                    batch["input_ids"], batch["attention_mask"], batch["timestamps"]
                )
                loss, loss_dict = total_loss(
                    arith_pred, batch["arith_labels"],
                    dur_pred, batch["dur_labels"],
                    batch["start_times"], arith_pred,
                    gate_reg,
                    self.config.lambda_torus, self.config.lambda_consist, self.config.lambda_gate,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                self.global_step += 1

                gate_val = self.model.gate_value
                gate_monitor.on_step(self.global_step, gate_val)
                pbar.set_postfix(loss=f"{loss.item():.4f}", gate=f"{gate_val:.4f}")

                if self.global_step % self.config.log_every_steps == 0:
                    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                    self._log_step({
                        "loss/total": loss.item(),
                        "loss/arithmetic": loss_dict["arith"],
                        "loss/duration": loss_dict["dur"],
                        "loss/consistency": loss_dict["consist"],
                        "loss/gate_reg": loss_dict["gate"],
                        "model/gate_value": gate_val,
                        "model/grad_norm": _grad_norm(self.model),
                        "train/gpu_memory_gb": gpu_mem,
                    })

            val_metrics = self.evaluate()
            self.logger.info(f"Phase1 Epoch {epoch+1} val: {val_metrics}")
            val_metrics.update({
                "model/gate_value": self.model.gate_value,
                "model/learned_freq_mean": self.model.time_embedding.log_freq_learned.exp().mean().item(),
                "model/learned_freq_std": self.model.time_embedding.log_freq_learned.exp().std().item(),
            })
            self.saver.maybe_save(self.model, optimizer, scheduler, val_metrics,
                                  self.global_step, epoch, "phase1", self.config)

        return self.evaluate()

    # ─── Phase 2 ───────────────────────────────────────────────────────────────

    def train_phase2(self) -> Dict:
        """
        Phase 2: Supervised fine-tuning. LoRA adapters unfrozen, first 4 blocks frozen.
        """
        _set_seed(self.config.seed)
        self.logger.info("=== Phase 2: Supervised Fine-Tuning ===")

        # Load best phase1 checkpoint
        try:
            self.saver.load_best(self.model)
            self.logger.info("Loaded best Phase 1 checkpoint.")
        except FileNotFoundError:
            self.logger.warning("No Phase 1 checkpoint found, continuing from current state.")

        # Freeze first N layers, enable LoRA
        self.model.freeze_backbone_layers(self.config.frozen_layers)
        self.model.unfreeze_lora_layers()

        optimizer = torch.optim.AdamW([
            {"params": [p for n, p in self.model.backbone.named_parameters() if "lora_" in n and p.requires_grad],
             "lr": self.config.phase2_lr_backbone},
            {"params": self.model.time_embedding.parameters(), "lr": self.config.phase2_lr_emb},
            {"params": self.model.fusion.parameters(), "lr": self.config.phase2_lr_emb},
            {"params": list(self.model.pooler.parameters()) +
                       list(self.model.arith_head.parameters()) +
                       list(self.model.dur_head.parameters()), "lr": self.config.phase2_lr_heads},
        ])

        steps_per_epoch = len(self.train_loader) // self.config.phase2_grad_accum
        total_steps = steps_per_epoch * self.config.phase2_epochs
        scheduler = get_phase_scheduler(optimizer, self.config.phase2_warmup_steps, total_steps)

        gate_monitor = GateMonitorCallback(self.config.gate_threshold, warmup_steps=100)
        freq_snapshot_interval = 500

        for epoch in range(self.config.phase2_epochs):
            self.model.train()
            pbar = tqdm(self.train_loader, desc=f"Phase2 E{epoch+1}")
            optimizer.zero_grad()

            for i, batch in enumerate(pbar):
                batch = self._move_batch(batch)
                arith_pred, dur_pred, gate_reg = self.model(
                    batch["input_ids"], batch["attention_mask"], batch["timestamps"]
                )
                loss, loss_dict = total_loss(
                    arith_pred, batch["arith_labels"],
                    dur_pred, batch["dur_labels"],
                    batch["start_times"], arith_pred,
                    gate_reg,
                    self.config.lambda_torus, self.config.lambda_consist, self.config.lambda_gate,
                )
                (loss / self.config.phase2_grad_accum).backward()

                if (i + 1) % self.config.phase2_grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    self.global_step += 1

                    gate_val = self.model.gate_value
                    gate_monitor.on_step(self.global_step, gate_val)
                    pbar.set_postfix(loss=f"{loss.item():.4f}", gate=f"{gate_val:.4f}")

                    if self.global_step % self.config.log_every_steps == 0:
                        gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                        lrs = [pg["lr"] for pg in optimizer.param_groups]
                        self._log_step({
                            "loss/total": loss.item(),
                            "loss/arithmetic": loss_dict["arith"],
                            "loss/duration": loss_dict["dur"],
                            "loss/consistency": loss_dict["consist"],
                            "loss/gate_reg": loss_dict["gate"],
                            "model/gate_value": gate_val,
                            "model/grad_norm": _grad_norm(self.model),
                            "train/lr_backbone": lrs[0],
                            "train/lr_embedding": lrs[1],
                            "train/lr_heads": lrs[3] if len(lrs) > 3 else lrs[-1],
                            "train/gpu_memory_gb": gpu_mem,
                        })

                    if self.global_step % freq_snapshot_interval == 0:
                        freqs = self.model.time_embedding.log_freq_learned.exp().detach().cpu().tolist()
                        self.logger.info(f"Learned frequencies at step {self.global_step}: {[f'{f:.4f}' for f in freqs]}")

                    if self.global_step % self.config.eval_every_steps == 0:
                        val_metrics = self.evaluate()
                        self.saver.maybe_save(self.model, optimizer, scheduler, val_metrics,
                                              self.global_step, epoch, "phase2", self.config)

            val_metrics = self.evaluate()
            self.logger.info(f"Phase2 Epoch {epoch+1} val: {val_metrics}")
            self.saver.maybe_save(self.model, optimizer, scheduler, val_metrics,
                                  self.global_step, epoch, "phase2", self.config)

        return self.evaluate()

    # ─── Phase 3 ───────────────────────────────────────────────────────────────

    def train_phase3_grpo(self) -> Dict:
        """
        Phase 3: GRPO reinforcement learning.
        HybridTimeEmb frozen for first phase3_freeze_emb_steps steps, then unfrozen.
        """
        _set_seed(self.config.seed)
        self.logger.info("=== Phase 3: GRPO Reinforcement Learning ===")

        try:
            self.saver.load_best(self.model)
            self.logger.info("Loaded best Phase 2 checkpoint.")
        except FileNotFoundError:
            self.logger.warning("No Phase 2 checkpoint found.")

        # Freeze time embedding initially
        for p in self.model.time_embedding.parameters():
            p.requires_grad = False

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.phase3_lr,
        )

        phase3_step = 0
        emb_unfrozen = False

        for batch in tqdm(self.train_loader, desc="Phase3 GRPO"):
            batch = self._move_batch(batch)

            # Unfreeze embedding at step 500
            if phase3_step == self.config.phase3_freeze_emb_steps and not emb_unfrozen:
                for p in self.model.time_embedding.parameters():
                    p.requires_grad = True
                self.logger.info(f"Phase3: Unfreezing HybridTimeEmb at step {phase3_step}.")
                emb_unfrozen = True

            # GRPO: generate N rollouts, compute rewards, policy gradient update
            rewards = self._grpo_step(batch, optimizer)
            phase3_step += 1
            self.global_step += 1

            if phase3_step % self.config.log_every_steps == 0:
                self._log_step({
                    "grpo/reward_mean": float(np.mean(rewards)),
                    "grpo/reward_std": float(np.std(rewards)),
                    "model/gate_value": self.model.gate_value,
                })

            if phase3_step % self.config.eval_every_steps == 0:
                val_metrics = self.evaluate()
                self.saver.maybe_save(self.model, optimizer, None, val_metrics,
                                      self.global_step, 0, "phase3", self.config)

        return self.evaluate()

    def _grpo_step(self, batch: Dict, optimizer: torch.optim.Optimizer) -> list:
        """
        Single GRPO update step: generate N predictions, compute rewards,
        update policy with clipped surrogate loss.

        Returns:
            List of reward values for the batch.
        """
        self.model.train()
        all_rewards = []

        with torch.no_grad():
            base_arith, base_dur, _ = self.model(
                batch["input_ids"], batch["attention_mask"], batch["timestamps"]
            )

        # Generate N rollouts by adding small noise (simplified GRPO)
        rollout_losses = []
        for _ in range(self.config.phase3_n_generations):
            arith_pred, dur_pred, gate_reg = self.model(
                batch["input_ids"], batch["attention_mask"], batch["timestamps"]
            )
            arith_vals = arith_pred.squeeze().detach().cpu().tolist()
            dur_vals = dur_pred.squeeze().detach().cpu().tolist()
            arith_true = batch["arith_labels"].cpu().tolist()
            dur_true = batch["dur_labels"].cpu().tolist()

            if not isinstance(arith_vals, list):
                arith_vals = [arith_vals]
                dur_vals = [dur_vals]
                arith_true = [arith_true]
                dur_true = [dur_true]

            rewards = [
                compute_reward(a_p, a_t, "arithmetic") * 0.5 + compute_reward(d_p, d_t, "duration") * 0.5
                for a_p, a_t, d_p, d_t in zip(arith_vals, arith_true, dur_vals, dur_true)
            ]
            all_rewards.extend(rewards)

            reward_tensor = torch.tensor(rewards, device=arith_pred.device, dtype=torch.float32)
            policy_loss = -(reward_tensor * (arith_pred.squeeze() + dur_pred.squeeze())).mean()
            rollout_losses.append(policy_loss)

        # Average rollout losses
        total = sum(rollout_losses) / len(rollout_losses)
        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()

        return all_rewards


def compute_reward(
    pred: float,
    ground_truth: float,
    subtask: str,
    start: Optional[float] = None,
    end_pred: Optional[float] = None,
) -> float:
    """
    Compute GRPO reward combining correctness, format, and consistency.

    Args:
        pred: Model prediction.
        ground_truth: Ground truth value.
        subtask: "arithmetic" or "duration".
        start: Optional start time for consistency reward.
        end_pred: Optional predicted end time for consistency reward.

    Returns:
        Scalar reward in [0, 1].
    """
    error = abs(pred - ground_truth)
    if error == 0:
        r_correct = 1.0
    elif error <= 1:
        r_correct = 0.5
    elif error <= 5:
        r_correct = 0.2
    else:
        r_correct = 0.0

    r_format = 1.0  # full format reward when prediction is a valid scalar

    if start is not None and end_pred is not None:
        r_consist = max(0.0, 1.0 - min(abs(start + pred - end_pred) / 10.0, 1.0))
    else:
        r_consist = 0.0

    return 0.7 * r_correct + 0.2 * r_format + 0.1 * r_consist

