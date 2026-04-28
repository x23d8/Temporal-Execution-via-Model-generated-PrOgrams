"""Main training loop.

Features:
  - Custom PyTorch loop (no HF Trainer dependency) for full display control
  - Per-epoch Rich table: loss, perplexity, accuracy, F1, inference time, LR, status
  - End-of-run summary table across all epochs
  - Early stopping (configurable monitor, patience, min_delta)
  - Gradient accumulation + optional mixed-precision (fp16 / bf16)
  - Atomic checkpoint saves (write tmp → rename) after every epoch
  - Auto-resume from latest checkpoint or explicit path
  - SIGINT / SIGTERM handler: saves interrupt checkpoint before exit
"""
from __future__ import annotations

import json
import math
import os
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler

from .config import PipelineConfig
from .data import build_datasets
from .metrics import EarlyStopper, EpochMetrics, MetricsTracker
from .model import load_model, load_tokenizer, primary_device

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from tqdm.rich import tqdm

    _RICH = True
except ImportError:
    from tqdm import tqdm  # type: ignore

    _RICH = False


# ── Display helpers ───────────────────────────────────────────────────────────

def _plain(msg: str) -> str:
    """Strip Rich markup for plain-text fallback."""
    import re
    return re.sub(r"\[/?[^\]]*\]", "", msg)


class _Display:
    def __init__(self) -> None:
        self._con = Console() if _RICH else None

    def print(self, msg: str) -> None:
        if _RICH:
            self._con.print(msg)  # type: ignore[union-attr]
        else:
            print(_plain(msg))

    def epoch_table(self, m: EpochMetrics) -> None:
        status_style = {
            "running": "green",
            "early_stopped": "yellow",
            "completed": "bold green",
        }.get(m.status, "white")

        if _RICH:
            t = Table(title=f"Epoch {m.epoch}", show_header=False, min_width=42)
            t.add_column("Metric", style="cyan", no_wrap=True)
            t.add_column("Value", style="white")
            t.add_row("Status",           f"[{status_style}]{m.status}[/{status_style}]")
            t.add_row("Train Loss",       f"{m.train_loss:.4f}")
            t.add_row("Eval Loss",        f"{m.eval_loss:.4f}")
            t.add_row("Perplexity",       f"{m.perplexity:.2f}")
            t.add_row("Token Accuracy",   f"{m.token_accuracy:.4f}")
            t.add_row("Token F1",         f"{m.token_f1:.4f}")
            t.add_row("Avg Infer. (ms)",  f"{m.avg_inference_ms:.2f}")
            t.add_row("Learning Rate",    f"{m.learning_rate:.3e}")
            self._con.print(t)  # type: ignore[union-attr]
        else:
            sep = "=" * 50
            print(f"\n{sep}\nEpoch {m.epoch} | Status: {m.status}\n{sep}")
            print(f"  Train Loss      : {m.train_loss:.4f}")
            print(f"  Eval  Loss      : {m.eval_loss:.4f}")
            print(f"  Perplexity      : {m.perplexity:.2f}")
            print(f"  Token Accuracy  : {m.token_accuracy:.4f}")
            print(f"  Token F1        : {m.token_f1:.4f}")
            print(f"  Avg Infer. (ms) : {m.avg_inference_ms:.2f}")
            print(f"  Learning Rate   : {m.learning_rate:.3e}")
            print(sep)

    def summary_table(self, history: list[EpochMetrics]) -> None:
        if not history:
            return
        if _RICH:
            t = Table(title="Training Summary", show_header=True)
            for col, just in [
                ("Ep", "center"), ("TrLoss", "right"), ("EvLoss", "right"),
                ("PPL", "right"), ("Acc", "right"), ("F1", "right"),
                ("Infer(ms)", "right"), ("Status", "center"),
            ]:
                t.add_column(col, justify=just)
            for m in history:
                ss = {"running": "green", "early_stopped": "yellow", "completed": "bold green"}.get(m.status, "white")
                t.add_row(
                    str(m.epoch),
                    f"{m.train_loss:.4f}", f"{m.eval_loss:.4f}",
                    f"{m.perplexity:.1f}", f"{m.token_accuracy:.4f}", f"{m.token_f1:.4f}",
                    f"{m.avg_inference_ms:.1f}",
                    f"[{ss}]{m.status}[/{ss}]",
                )
            self._con.print(t)  # type: ignore[union-attr]
        else:
            hdr = f"{'Ep':>3} {'TrLoss':>8} {'EvLoss':>8} {'PPL':>7} {'Acc':>7} {'F1':>7} {'InfMs':>8} {'Status':>14}"
            print("\n=== Training Summary ===\n" + hdr)
            for m in history:
                print(
                    f"{m.epoch:>3} {m.train_loss:>8.4f} {m.eval_loss:>8.4f} "
                    f"{m.perplexity:>7.1f} {m.token_accuracy:>7.4f} {m.token_f1:>7.4f} "
                    f"{m.avg_inference_ms:>8.1f} {m.status:>14}"
                )


# ── Trainer ───────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self.out_dir  = Path(cfg.logging.output_dir)
        self.ckpt_dir = self.out_dir / "checkpoints"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self._display  = _Display()
        self._tracker  = MetricsTracker()
        self._cur_epoch = 0
        self._interrupted = False

        signal.signal(signal.SIGINT, self._on_interrupt)
        try:
            signal.signal(signal.SIGTERM, self._on_interrupt)
        except (OSError, AttributeError):
            pass

        self._setup()

    # ── Setup ─────────────────────────────────────────────────────────────

    def _setup(self) -> None:
        self._display.print("[bold cyan]Setting up pipeline…[/bold cyan]")

        self._display.print("Loading tokenizer…")
        self.tokenizer = load_tokenizer(self.cfg)

        self._display.print("Loading model…")
        self.model = load_model(self.cfg)
        self._device = primary_device(self.model)

        self._display.print("Loading datasets…")
        train_ds, eval_ds = build_datasets(
            self.cfg.data, self.cfg.model, self.tokenizer, self.cfg.training.seed
        )

        t = self.cfg.training
        self._train_loader = DataLoader(
            train_ds, batch_size=t.batch_size, shuffle=True,
            num_workers=t.dataloader_num_workers, pin_memory=True,
        )
        self._eval_loader = DataLoader(
            eval_ds, batch_size=t.eval_batch_size, shuffle=False,
            num_workers=t.dataloader_num_workers, pin_memory=True,
        )

        self._optimizer = self._build_optimizer()
        self._total_steps = (
            math.ceil(len(self._train_loader) / t.gradient_accumulation_steps)
            * t.epochs
        )
        self._scheduler = get_scheduler(
            t.lr_scheduler_type,
            optimizer=self._optimizer,
            num_warmup_steps=t.warmup_steps,
            num_training_steps=self._total_steps,
        )
        self._scaler = (
            torch.cuda.amp.GradScaler()  # type: ignore[attr-defined]
            if t.fp16 and torch.cuda.is_available()
            else None
        )
        self._early_stopper: Optional[EarlyStopper] = (
            EarlyStopper(
                self.cfg.early_stopping.patience,
                self.cfg.early_stopping.min_delta,
                self.cfg.early_stopping.monitor,
            )
            if self.cfg.early_stopping.enabled
            else None
        )

        self._start_epoch = 0
        ccfg = self.cfg.checkpoint
        if ccfg.resume_from and ccfg.resume_from != "auto":
            self._start_epoch = self._load_ckpt(Path(ccfg.resume_from))
        elif ccfg.enabled and (ccfg.resume_from == "auto" or not ccfg.resume_from):
            latest = self._latest_ckpt()
            if latest:
                self._display.print(f"[yellow]Auto-resuming from {latest.name}[/yellow]")
                self._start_epoch = self._load_ckpt(latest)

    def _build_optimizer(self) -> AdamW:
        no_decay = {"bias", "LayerNorm.weight"}
        params = [
            {
                "params": [p for n, p in self.model.named_parameters()
                           if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": self.cfg.training.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                           if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
        lr = self.cfg.training.learning_rate
        optim_name = self.cfg.training.optim.lower()
        if "paged_adamw" in optim_name:
            try:
                from bitsandbytes.optim import PagedAdamW32bit
                return PagedAdamW32bit(params, lr=lr)
            except ImportError:
                self._display.print("[yellow]bitsandbytes not found; using AdamW[/yellow]")
        return AdamW(params, lr=lr)

    # ── Signal handler ────────────────────────────────────────────────────

    def _on_interrupt(self, signum, frame) -> None:
        self._display.print("\n[yellow]Interrupt! Saving checkpoint…[/yellow]")
        self._interrupted = True
        self._save_ckpt(self._cur_epoch, suffix="interrupt")
        sys.exit(0)

    # ── Checkpoint ────────────────────────────────────────────────────────

    def _save_ckpt(
        self,
        epoch: int,
        metrics: Optional[EpochMetrics] = None,
        suffix: Optional[str] = None,
    ) -> None:
        name = suffix or f"epoch_{epoch}"
        target = self.ckpt_dir / f"ckpt_{name}"
        tmp    = self.ckpt_dir / f"ckpt_{name}_tmp"
        tmp.mkdir(parents=True, exist_ok=True)

        try:
            self.model.save_pretrained(tmp)
            self.tokenizer.save_pretrained(tmp)

            state: dict = {
                "epoch": epoch,
                "optimizer": self.optimizer_state(),
                "scheduler": self._scheduler.state_dict(),
                "metrics": [
                    {
                        k: getattr(m, k)
                        for k in ("epoch", "train_loss", "eval_loss", "perplexity",
                                  "token_accuracy", "token_f1", "avg_inference_ms",
                                  "learning_rate", "status")
                    }
                    for m in self._tracker.history
                ],
                "early_stopper": (
                    self._early_stopper.state() if self._early_stopper else None
                ),
            }
            if self._scaler:
                state["scaler"] = self._scaler.state_dict()

            torch.save(state, tmp / "training_state.pt")

            # Atomic rename
            if target.exists():
                shutil.rmtree(target)
            tmp.rename(target)
            self._display.print(f"[green]Checkpoint → {target.name}[/green]")

            self._prune_ckpts()

        except Exception as exc:
            self._display.print(f"[red]Checkpoint save failed: {exc}[/red]")
            if tmp.exists():
                shutil.rmtree(tmp)

    def optimizer_state(self) -> dict:
        return self._optimizer.state_dict()

    def _load_ckpt(self, path: Path) -> int:
        state_file = path / "training_state.pt"
        if not state_file.exists():
            self._display.print(f"[yellow]No training_state.pt in {path}; starting fresh[/yellow]")
            return 0

        state = torch.load(state_file, map_location="cpu")
        self._optimizer.load_state_dict(state["optimizer"])
        self._scheduler.load_state_dict(state["scheduler"])
        if self._scaler and "scaler" in state:
            self._scaler.load_state_dict(state["scaler"])
        if self._early_stopper and state.get("early_stopper"):
            self._early_stopper.load_state(state["early_stopper"])

        epoch = state.get("epoch", 0)
        self._display.print(f"[green]Resumed at epoch {epoch}[/green]")
        return epoch + 1

    def _latest_ckpt(self) -> Optional[Path]:
        ckpts = [
            d for d in self.ckpt_dir.iterdir()
            if d.is_dir() and d.name.startswith("ckpt_epoch_")
        ]
        if not ckpts:
            return None

        def _num(p: Path) -> int:
            try:
                return int(p.name.split("_")[-1])
            except ValueError:
                return -1

        return max(ckpts, key=_num)

    def _prune_ckpts(self) -> None:
        limit = self.cfg.checkpoint.save_total_limit
        ckpts = sorted(
            [d for d in self.ckpt_dir.iterdir()
             if d.is_dir() and d.name.startswith("ckpt_epoch_")],
            key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else 0,
        )
        while len(ckpts) > limit:
            old = ckpts.pop(0)
            shutil.rmtree(old)
            self._display.print(f"[dim]Pruned {old.name}[/dim]")

    # ── Epoch loops ───────────────────────────────────────────────────────

    def _amp_ctx(self):
        t = self.cfg.training
        if not torch.cuda.is_available():
            return torch.no_grad.__class__  # dummy context manager
        if t.bf16:
            return torch.autocast("cuda", dtype=torch.bfloat16)
        if t.fp16:
            return torch.autocast("cuda", dtype=torch.float16)
        # Return a no-op context
        import contextlib
        return contextlib.nullcontext()

    def _train_epoch(self, epoch: int) -> None:
        self.model.train()
        t = self.cfg.training
        acc_steps = t.gradient_accumulation_steps
        self._optimizer.zero_grad()

        pbar = tqdm(
            self._train_loader,
            desc=f"Ep {epoch}/{t.epochs} [train]",
            leave=False,
        )
        for step, batch in enumerate(pbar):
            batch = {k: v.to(self._device) for k, v in batch.items()}

            with self._amp_ctx():
                out  = self.model(**batch)
                loss = out.loss / acc_steps

            if self._scaler:
                self._scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation flush
            if (step + 1) % acc_steps == 0:
                if self._scaler:
                    self._scaler.unscale_(self._optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), t.max_grad_norm
                    )
                    self._scaler.step(self._optimizer)
                    self._scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), t.max_grad_norm
                    )
                    self._optimizer.step()

                self._scheduler.step()
                self._optimizer.zero_grad()

            raw_loss = loss.item() * acc_steps
            self._tracker.add_train_loss(raw_loss)
            pbar.set_postfix(
                loss=f"{raw_loss:.4f}",
                lr=f"{self._scheduler.get_last_lr()[0]:.2e}",
            )

    @torch.no_grad()
    def _eval_epoch(self, epoch: int) -> None:
        self.model.eval()
        t = self.cfg.training
        pbar = tqdm(
            self._eval_loader,
            desc=f"Ep {epoch}/{t.epochs} [eval]",
            leave=False,
        )
        for batch in pbar:
            batch = {k: v.to(self._device) for k, v in batch.items()}

            t0 = time.perf_counter()
            with self._amp_ctx():
                out = self.model(**batch)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            bs = batch["input_ids"].shape[0]
            self._tracker.add_eval_batch(
                loss=out.loss.item(),
                logits_np=out.logits.cpu().float().numpy(),
                labels_np=batch["labels"].cpu().numpy(),
                inference_ms_per_sample=elapsed_ms / bs,
            )

    # ── Save metrics log ──────────────────────────────────────────────────

    def _flush_metrics(self) -> None:
        log = [
            {k: getattr(m, k) for k in (
                "epoch", "train_loss", "eval_loss", "perplexity",
                "token_accuracy", "token_f1", "avg_inference_ms",
                "learning_rate", "status"
            )}
            for m in self._tracker.history
        ]
        with open(self.out_dir / "metrics_history.json", "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)

    # ── Main entry ────────────────────────────────────────────────────────

    def train(self) -> None:
        t = self.cfg.training
        header = (
            f"[bold]{self.cfg.model.name}[/bold]  "
            f"PEFT=[cyan]{self.cfg.peft.method.upper()}[/cyan]  "
            f"Epochs={t.epochs}  BS={t.batch_size}  "
            f"GradAcc={t.gradient_accumulation_steps}  LR={t.learning_rate}"
        )
        if _RICH:
            self._display.print(Panel.fit(header, title="[bold]Kaggle Finetune Pipeline[/bold]"))  # type: ignore[arg-type]
        else:
            self._display.print(f"\n=== Kaggle Finetune Pipeline ===\n{_plain(header)}\n")

        final_status = "completed"

        for epoch in range(self._start_epoch + 1, t.epochs + 1):
            self._cur_epoch = epoch
            self._tracker.reset_epoch()

            self._train_epoch(epoch)
            self._eval_epoch(epoch)

            cur_lr = self._scheduler.get_last_lr()[0]
            status = "running"

            # Early stopping check
            stop_now = False
            if self._early_stopper:
                monitor_val = self._tracker.last(self.cfg.early_stopping.monitor)
                if self._early_stopper.step(monitor_val):
                    stop_now = True
                    status = "early_stopped"

            if epoch == t.epochs and not stop_now:
                status = "completed"

            metrics = self._tracker.compute(epoch, cur_lr, status)
            self._display.epoch_table(metrics)
            self._flush_metrics()

            if self.cfg.checkpoint.enabled:
                self._save_ckpt(epoch, metrics)

            if stop_now:
                self._display.print(
                    f"[yellow]Early stopping at epoch {epoch} "
                    f"(patience={self.cfg.early_stopping.patience})[/yellow]"
                )
                final_status = "early_stopped"
                break

        # Save final model
        final_path = self.out_dir / "final_model"
        self.model.save_pretrained(final_path)
        self.tokenizer.save_pretrained(final_path)
        self._display.print(
            f"[bold green]Training {final_status}. "
            f"Model saved → {final_path}[/bold green]"
        )
        self._display.summary_table(self._tracker.history)
