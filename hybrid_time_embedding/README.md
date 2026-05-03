# Hybrid Time Embedding

Production-ready PyTorch system for temporal QA using Qwen2.5-7B with a 6-layer
hybrid architecture. Supports Date Arithmetic and Date Duration subtasks.

---

## 1. Architecture Overview (6-Layer Pipeline)

```
┌─────────────────────────────────────────────────────────────────────┐
│  INPUT: query text + timestamp floats                               │
└────────────────────┬────────────────────────────────────────────────┘
                     │
           ┌─────────▼──────────┐
           │  Layer 1           │  Token Embedding
           │  embed_tokens      │  [B, S] → [B, S, 3584]   FROZEN
           └─────────┬──────────┘
                     │            timestamp floats (separate path)
           ┌─────────▼──────────┐      ┌──────────────────────┐
           │  Layer 2           │      │  Linear Branch       │
           │  OptimalHybrid     │◄─────│  + Toroidal Branch   │
           │  TimeEmbedding     │      │  (P1 learned + P2    │
           └─────────┬──────────┘      │   random Fourier)    │
                     │                 └──────────────────────┘
           ┌─────────▼──────────┐      [B] → [B, 3584]  TRAINABLE
           │  Layer 3           │
           │  OptimalFusion     │  Gated add: token + gate * time
           │  (gate=0.1 init)   │  [B,S,D] + [B,D] → [B,S,D]
           └─────────┬──────────┘
                     │
           ┌─────────▼──────────┐
           │  Layer 4           │  Qwen2.5-7B Transformer
           │  Transformer       │  Layers 0-3: FROZEN
           │  (LoRA adapters)   │  Layers 4-27: LoRA (r=16)
           └─────────┬──────────┘
                     │
           ┌─────────▼──────────┐
           │  Layer 5           │  AttentionPooling → dual heads
           │  Task Heads        │  arith_head: ℝ (year)
           │                    │  dur_head: ℝ⁺ (duration, Softplus)
           └─────────┬──────────┘
                     │
           ┌─────────▼──────────┐
           │  Layer 6           │  GRPO RL (Phase 3)
           │  Reinforcement     │  N=8 generations, beta=0.04
           │  Learning          │  EMB frozen → unfrozen at step 500
           └────────────────────┘
```

---

## 2. Folder Map

```
hybrid_time_embedding/
├── src/
│   ├── __init__.py              Public API: HybridConfig, HybridTemporalModel, TemporalQADataset
│   ├── data/
│   │   ├── dataset.py           TemporalQADataset — loads JSON splits for both subtasks
│   │   ├── collator.py          DataCollatorWithTimestamps — pads text + stacks timestamps
│   │   └── preprocessing.py    extract_timestamps, normalize_timestamp, add_time_tokens
│   ├── models/
│   │   ├── hybrid_time_emb.py  OptimalHybridTimeEmbedding (linear + toroidal branches)
│   │   ├── fusion.py            OptimalFusion (gated add + LayerNorm + gate_reg loss)
│   │   ├── task_heads.py        AttentionPooling, ArithmeticHead, DurationHead
│   │   └── full_model.py        HybridTemporalModel — wires all 6 layers
│   ├── training/
│   │   ├── losses.py            wrapped_torus_loss, consistency_loss, total_loss
│   │   ├── trainer.py           PhaseAwareTrainer (Phase 1/2/3 loops)
│   │   ├── callbacks.py         SmartCheckpointSaver, GateMonitorCallback, MetricCallback
│   │   └── scheduler.py         get_phase_scheduler (warmup + cosine)
│   └── utils/
│       ├── config.py            HybridConfig dataclass (all hyperparams)
│       ├── logging_utils.py     setup_logging (Python + TensorBoard + WandB)
│       └── metrics.py           compute_metrics, MAE, exact_match, consistency_rate
├── models/                      Saved checkpoints (manifest.json + per-phase folders)
├── evaluate/
│   ├── evaluator.py             TemporalEvaluator — per-subtask evaluation
│   ├── metrics_report.py        generate_report() — prints + saves JSON + CSV
│   └── error_analysis.py        bucket_by_magnitude, bucket_by_time_period, worst_predictions
├── inference/
│   ├── pipeline.py              TemporalQAPipeline — load checkpoint + run inference
│   ├── predictor.py             single_predict(), batch_predict() with timing
│   └── run_inference.py         CLI: python run_inference.py --checkpoint ... --query ...
├── experiments/                 Training logs and results
├── finetune.ipynb               Main training notebook (7 sections)
└── README.md
```

---

## 3. Quick Start

**Step 1 — Install dependencies:**
```bash
pip install torch transformers peft trl safetensors bitsandbytes tqdm tabulate
```

**Step 2 — Prepare data:**
Place your JSON files in `data/train.json`, `data/val.json`, `data/test.json`.
Each file is a JSON array following the schema in [Dataset Format](#dataset-format).

**Step 3 — Run the notebook:**
```bash
cd hybrid_time_embedding
jupyter notebook finetune.ipynb
```
Set `config.data_dir = "./data"` in Cell 0.2, then run all cells top-to-bottom.

---

## 4. Config Reference

| Field | Default | Description |
|---|---|---|
| `base_model_name` | `Qwen/Qwen2.5-7B` | HuggingFace model ID |
| `d_model` | `3584` | Hidden dimension |
| `n_learned_freq` | `8` | Learnable toroidal frequencies |
| `n_random_freq` | `16` | Fixed random Fourier features |
| `gate_init` | `0.1` | Initial fusion gate value |
| `gate_threshold` | `0.05` | Gate collapse warning threshold |
| `lora_r` | `16` | LoRA rank |
| `lora_alpha` | `32` | LoRA alpha |
| `frozen_layers` | `4` | Transformer blocks to freeze |
| `lambda_torus` | `0.3` | Torus loss weight |
| `lambda_consist` | `0.5` | Consistency loss weight |
| `lambda_gate` | `1.0` | Gate regularization weight |
| `phase1_epochs` | `2` | Embedding warmup epochs |
| `phase1_lr_emb` | `1e-3` | Phase 1 learning rate |
| `phase2_epochs` | `3` | SFT epochs |
| `phase2_lr_backbone` | `2e-5` | LoRA backbone LR |
| `phase2_lr_emb` | `1e-4` | Embedding LR in Phase 2 |
| `phase2_lr_heads` | `1e-3` | Task head LR in Phase 2 |
| `phase2_grad_accum` | `8` | Gradient accumulation steps |
| `phase3_lr` | `5e-7` | GRPO learning rate |
| `phase3_n_generations` | `8` | GRPO rollouts per step |
| `phase3_beta` | `0.04` | GRPO KL penalty |
| `phase3_freeze_emb_steps` | `500` | Steps before unfreezing TimeEmb in Phase 3 |
| `checkpoint_top_k` | `3` | Max checkpoints to retain |
| `seed` | `42` | Random seed |

---

## 5. Checkpoint Format

`models/manifest.json` tracks all saved checkpoints:

```json
{
  "best_mae": 1.87,
  "checkpoints": [
    {
      "folder": "phase2_step3000_mae1.8700",
      "step": 3000,
      "epoch": 2,
      "phase": "phase2",
      "val_mae": 1.87,
      "metrics": { "val/mae_overall": 1.87, "val/exact_match_arithmetic": 0.62 }
    }
  ]
}
```

Each checkpoint folder contains:
- `model.safetensors` — model weights
- `optimizer.pt` — optimizer state
- `scheduler.pt` — scheduler state
- `config.json` — HybridConfig snapshot
- `metrics.json` — validation metrics at save time

---

## 6. Expected Metrics

| Phase | MAE (arith) | MAE (dur) | Exact Match |
|---|---|---|---|
| After Phase 1 (warmup) | ~15–25 yr | ~10–20 yr | ~5–15% |
| After Phase 2 (SFT) | ~2–5 yr | ~3–8 yr | ~40–65% |
| After Phase 3 (GRPO) | ~1–3 yr | ~2–6 yr | ~55–75% |

*(Depends heavily on dataset quality and size)*

---

## 7. Troubleshooting

**Gate collapse** (`gate_value < 0.01`):
- Increase `lambda_gate` (try 2.0–5.0)
- Reduce `phase1_lr_emb` — learning rate too high destabilizes the gate
- Check `gate_threshold` and `gate_init` are consistent

**OOM on A100 80GB**:
- Reduce `phase2_batch_size` to 4 and increase `phase2_grad_accum` to 16
- Enable `load_in_4bit=True` for 4-bit quantization
- Reduce `max_length` in the collator (default 512)

**NaN loss**:
- Check for missing timestamps (empty `timestamps` list in data)
- Ensure `normalize_timestamp` is not dividing by zero (year_min != year_max)
- Reduce learning rates by 10× and re-run Phase 1 from scratch
- Enable `torch.autograd.set_detect_anomaly(True)` to locate the NaN source

**Slow convergence**:
- Verify `[TIME_START]` and `[TIME_END]` tokens are in the tokenizer vocabulary
- Check that `time_embedding` is actually receiving gradients (Cell 2.3 test)
- Inspect learned frequencies — if they collapse to a single value, increase `n_random_freq`
