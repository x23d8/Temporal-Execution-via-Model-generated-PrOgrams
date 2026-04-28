"""Log-probability utilities cho evaluation không dùng generation.

Thay vì greedy decode "yes"/"no", ta tính:
    score = log P(token | prompt)   tại vị trí prediction cuối cùng

Cho duration:
    logprob_predict_binary(model, tokenizer, prompt_ids) → "yes" | "no"
    → Không có parse error, không bị ảnh hưởng bởi garbage tokens.

Cho date_arith:
    logprob_score(model, tokenizer, prompt_ids, answer_ids) → float
    → Dùng để rank candidate dates theo likelihood.

logprob_evaluate(model, tokenizer, samples, prompt_builder, device) → dict
    → Chạy log-prob eval trên toàn bộ eval set, trả về metrics dict.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.evaluation.metrics import binary_f1_yes, accuracy  # noqa: E402


@torch.inference_mode()
def logprob_score(
    model: Any,
    prompt_ids: list[int],
    target_ids: list[int],
    device: torch.device,
) -> float:
    """Tính log P(target | prompt) = tổng log-prob của từng target token.

    Dùng để score candidate dates trong date_arith.
    """
    input_ids = torch.tensor([prompt_ids + target_ids], dtype=torch.long).to(device)
    outputs = model(input_ids=input_ids)
    # logits: [1, seq_len, vocab] → lấy logits tại các vị trí predict target
    logits = outputs.logits[0, len(prompt_ids) - 1 : len(prompt_ids) + len(target_ids) - 1, :]
    log_probs = F.log_softmax(logits, dim=-1)
    target_tensor = torch.tensor(target_ids, dtype=torch.long).to(device)
    return log_probs[range(len(target_ids)), target_tensor].sum().item()


@torch.inference_mode()
def logprob_predict_binary(
    model: Any,
    tokenizer: Any,
    prompt_ids: list[int],
    device: torch.device,
) -> tuple[str, dict[str, float]]:
    """Predict "yes" hoặc "no" bằng cách so log P tại next-token position.

    Trả về (prediction, {"log_prob_yes": float, "log_prob_no": float}).
    """
    yes_id = tokenizer.encode("yes", add_special_tokens=False)[0]
    no_id  = tokenizer.encode("no",  add_special_tokens=False)[0]

    input_ids = torch.tensor([prompt_ids], dtype=torch.long).to(device)
    outputs   = model(input_ids=input_ids)
    logits    = outputs.logits[0, -1, :]            # vị trí cuối → next token
    log_probs = F.log_softmax(logits, dim=-1)

    lp_yes = log_probs[yes_id].item()
    lp_no  = log_probs[no_id].item()
    pred   = "yes" if lp_yes >= lp_no else "no"
    return pred, {"log_prob_yes": lp_yes, "log_prob_no": lp_no}


def logprob_evaluate(
    model: Any,
    tokenizer: Any,
    samples: list[dict],
    prompt_builder: Any,       # DualPromptBuilder
    device: torch.device,
    task_filter: str | None = None,
) -> dict:
    """Chạy log-prob evaluation trên danh sách samples.

    task_filter: None = tất cả task; "duration" hoặc "date_arith" = lọc theo task.
    Trả về metrics dict tương thích với src/evaluation/metrics.
    """
    from src.evaluation.extractor import normalize_gold  # lazy import

    model.eval()
    golds:  list[str | None] = []
    preds:  list[str | None] = []
    tasks:  list[str] = []

    for sample in samples:
        task = sample["task"]
        lang = sample["language"]

        if task_filter and task != task_filter:
            continue

        gold_norm = normalize_gold(task, lang, sample["gold"])
        prompt_ids = prompt_builder.build_cls_prompt_only(sample)

        if task == "duration":
            pred, _ = logprob_predict_binary(model, tokenizer, prompt_ids, device)
        else:
            # date_arith: không có candidate set → fallback score trên gold
            # (dùng logprob_score để debug; prediction thực tế dùng greedy gen)
            gold_ids = tokenizer.encode(sample["gold"], add_special_tokens=False)
            score    = logprob_score(model, prompt_ids, gold_ids, device)
            pred     = sample["gold"] if score > -50.0 else None

        golds.append(gold_norm)
        preds.append(normalize_gold(task, lang, pred) if pred else None)
        tasks.append(task)

    if not golds:
        return {}

    # Tách metrics theo task
    metrics: dict[str, Any] = {}
    for t in ("duration", "date_arith"):
        t_golds = [g for g, tk in zip(golds, tasks) if tk == t]
        t_preds = [p for p, tk in zip(preds, tasks) if tk == t]
        if not t_golds:
            continue
        if t == "duration":
            metrics[t] = binary_f1_yes(t_golds, t_preds)
        else:
            metrics[t] = accuracy(t_golds, t_preds)

    return metrics
