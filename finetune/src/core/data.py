"""Generic data loading for the Kaggle finetuning pipeline.

Supported formats (configured via data.format in YAML):
  "text"        – {"text": "..."}
  "instruction" – {"instruction": "...", "input": "...", "output": "..."}
  "chat"        – {"messages": [{"role": "user", "content": "..."}, ...]}
"""
from __future__ import annotations

import json
import random
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from .config import DataConfig, ModelConfig


# ── JSONL loader ──────────────────────────────────────────────────────────────

def _load_jsonl(path: str, max_samples: Optional[int] = None) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if max_samples is not None:
        rows = rows[:max_samples]
    return rows


# ── Text extraction ───────────────────────────────────────────────────────────

def _extract_text(row: dict, dcfg: DataConfig) -> str:
    fmt = dcfg.format

    if fmt == "text":
        return row[dcfg.text_column]

    if fmt == "instruction":
        return dcfg.prompt_template.format(
            instruction=row.get(dcfg.instruction_column, ""),
            input=row.get(dcfg.input_column, ""),
            output=row.get(dcfg.output_column, ""),
        )

    if fmt == "chat":
        msgs = row[dcfg.messages_column]
        parts = []
        for m in msgs:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"<|{role}|>\n{content}")
        return "\n".join(parts)

    raise ValueError(f"Unknown data.format: {fmt!r}")


# ── Dataset ───────────────────────────────────────────────────────────────────

class FinetuneDataset(Dataset):
    def __init__(
        self,
        rows: List[dict],
        tokenizer: PreTrainedTokenizerBase,
        dcfg: DataConfig,
        max_length: int,
    ) -> None:
        self._rows = rows
        self._tok = tokenizer
        self._dcfg = dcfg
        self._max_length = max_length

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict:
        text = _extract_text(self._rows[idx], self._dcfg)
        enc = self._tok(
            text,
            max_length=self._max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # Causal LM: labels = input_ids, pad tokens masked with -100
        labels = input_ids.clone()
        labels[labels == self._tok.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ── Builder ───────────────────────────────────────────────────────────────────

def build_datasets(
    dcfg: DataConfig,
    mcfg: ModelConfig,
    tokenizer: PreTrainedTokenizerBase,
    seed: int = 42,
) -> Tuple[FinetuneDataset, FinetuneDataset]:
    train_rows = _load_jsonl(dcfg.train_path, dcfg.max_samples)

    if dcfg.eval_path:
        eval_rows = _load_jsonl(dcfg.eval_path)
    else:
        rng = random.Random(seed)
        rng.shuffle(train_rows)
        split = int(len(train_rows) * (1 - dcfg.validation_split))
        train_rows, eval_rows = train_rows[:split], train_rows[split:]

    print(
        f"[data] train={len(train_rows)}  eval={len(eval_rows)}  "
        f"format={dcfg.format}"
    )

    kw = dict(tokenizer=tokenizer, dcfg=dcfg, max_length=mcfg.max_length)
    return FinetuneDataset(train_rows, **kw), FinetuneDataset(eval_rows, **kw)
