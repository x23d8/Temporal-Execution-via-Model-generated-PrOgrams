"""Dual-Prompt Builder cho multi-task fine-tuning.

Mỗi Sample sinh ra HAI training examples:

  P_gen (Generative Prompt)
  ─────────────────────────
  [system: task-specific generation instruction]
  [user:   question / context+question+candidate]
  [assistant: gold_answer]          ← CE loss chỉ trên phần này

  P_cls (Classification Prompt — chỉ khi prompt_mode="dual")
  ────────────────────────────────────────────────────────────
  [system: verification instruction]
  [user:   question + proposed answer]
  [assistant: "yes" / "no"]         ← CE loss trên token duy nhất này

  duration  → P_cls dùng candidate_answer + gold label (yes/no tự nhiên)
  date_arith → P_cls dùng gold date (→ "yes") + corrupted date (→ "no") xen kẽ

Log-prob evaluation (duration):
  Không generate; forward pass → logit tại vị trí cuối → so P("yes") vs P("no").

Tokenisation helper:
  _tokenize_with_labels(prompt_text, answer_text, tokenizer, max_seq_len)
  Trả về dict {input_ids, attention_mask, labels} với labels=-100 ở phần prompt.
"""

from __future__ import annotations

import random
import datetime
from typing import Any

# System prompts ─────────────────────────────────────────────────────────────

_GEN_SYS: dict[tuple[str, str], str] = {
    ("date_arith", "en"): (
        "You are a date arithmetic solver. "
        "Respond with ONLY the target date in MM/DD/YYYY format. "
        "No reasoning, no words, no punctuation other than the slashes."
    ),
    ("date_arith", "vi"): (
        "Bạn là bộ giải các phép tính thời gian. "
        "Chỉ trả lời duy nhất theo đúng mẫu 'Tháng M, YYYY' (ví dụ: 'Tháng 4, 1321'). "
        "Không giải thích, không thêm ký tự hay văn bản nào khác."
    ),
    ("duration", "en"): (
        "You judge whether a candidate duration is plausible for the event in the context. "
        "Respond with only the single word 'yes' or 'no'. "
        "No reasoning, no punctuation, no extra tokens."
    ),
    ("duration", "vi"): (
        "Bạn đánh giá một khoảng thời gian có hợp lý cho sự kiện trong ngữ cảnh hay không. "
        "Chỉ trả lời duy nhất 'yes' hoặc 'no'. "
        "Không giải thích, không thêm bất kỳ ký tự nào khác."
    ),
}

_CLS_SYS: dict[tuple[str, str], str] = {
    ("date_arith", "en"): (
        "You verify date arithmetic answers. "
        "Given a question and a proposed date, reply with 'yes' if the date is correct, "
        "or 'no' if it is wrong. No other output."
    ),
    ("date_arith", "vi"): (
        "Bạn xác minh các đáp án phép tính ngày tháng. "
        "Cho câu hỏi và một ngày được đề xuất, trả lời 'yes' nếu ngày đúng, "
        "'no' nếu sai. Không thêm nội dung nào khác."
    ),
    ("duration", "en"): (
        "You judge whether a candidate duration is plausible for the event in the context. "
        "Respond with only the single word 'yes' or 'no'. "
        "No reasoning, no punctuation, no extra tokens."
    ),
    ("duration", "vi"): (
        "Bạn đánh giá một khoảng thời gian có hợp lý cho sự kiện trong ngữ cảnh hay không. "
        "Chỉ trả lời duy nhất 'yes' hoặc 'no'. "
        "Không giải thích, không thêm bất kỳ ký tự nào khác."
    ),
}


# User message renderers ──────────────────────────────────────────────────────

def _user_gen(sample: dict) -> str:
    task, lang = sample["task"], sample["language"]
    q = sample["question"]
    ctx = (sample.get("context") or "").strip()
    cand = (sample.get("meta") or {}).get("candidate_answer", "")

    if task == "date_arith":
        if lang == "vi" and ctx:
            return f"Ngữ cảnh: {ctx}\nCâu hỏi: {q}"
        return q

    # duration
    lines = []
    if ctx:
        lines.append(f"Context: {ctx}" if lang == "en" else f"Ngữ cảnh: {ctx}")
    lines.append(f"Question: {q}" if lang == "en" else f"Câu hỏi: {q}")
    if cand:
        lines.append(
            f"Candidate duration: {cand}" if lang == "en"
            else f"Khoảng thời gian đề xuất: {cand}"
        )
    if lang == "en":
        lines.append("Is this a plausible duration? Answer 'yes' or 'no'.")
    else:
        lines.append("Khoảng thời gian này có hợp lý không? Trả lời 'yes' hoặc 'no'.")
    return "\n".join(lines)


def _user_cls(sample: dict, proposed_answer: str) -> str:
    task, lang = sample["task"], sample["language"]
    q = sample["question"]
    ctx = (sample.get("context") or "").strip()
    cand = (sample.get("meta") or {}).get("candidate_answer", "")

    if task == "date_arith":
        if lang == "en":
            return f"Question: {q}\nProposed date: {proposed_answer}"
        ctx_part = f"Ngữ cảnh: {ctx}\n" if ctx else ""
        return f"{ctx_part}Câu hỏi: {q}\nNgày đề xuất: {proposed_answer}"

    # duration — same as gen (candidate already specified)
    lines = []
    if ctx:
        lines.append(f"Context: {ctx}" if lang == "en" else f"Ngữ cảnh: {ctx}")
    lines.append(f"Question: {q}" if lang == "en" else f"Câu hỏi: {q}")
    if cand:
        lines.append(
            f"Candidate duration: {cand}" if lang == "en"
            else f"Khoảng thời gian đề xuất: {cand}"
        )
    if lang == "en":
        lines.append("Is this a plausible duration? Answer 'yes' or 'no'.")
    else:
        lines.append("Khoảng thời gian này có hợp lý không? Trả lời 'yes' hoặc 'no'.")
    return "\n".join(lines)


# Tokenisation helper ─────────────────────────────────────────────────────────

def _tokenize_with_labels(
    prompt_text: str,
    answer_text: str,
    tokenizer: Any,
    max_seq_len: int,
) -> dict:
    """Tokenise prompt+answer; labels=-100 trên phần prompt (không tính loss ở đó)."""
    p_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    a_ids = tokenizer.encode(answer_text, add_special_tokens=False)

    # Giữ tối đa 25% cho answer; còn lại dành cho prompt (lấy phần cuối)
    max_a = max(1, max_seq_len // 4)
    if len(a_ids) > max_a:
        a_ids = a_ids[:max_a]

    max_p = max_seq_len - len(a_ids) - 1   # -1 cho eos
    if len(p_ids) > max_p:
        p_ids = p_ids[-max_p:]             # cắt đầu prompt, giữ phần cuối gần question

    eos = tokenizer.eos_token_id
    input_ids = p_ids + a_ids + [eos]
    labels    = [-100] * len(p_ids) + a_ids + [eos]
    attn_mask = [1] * len(input_ids)

    return {
        "input_ids":      input_ids,
        "attention_mask": attn_mask,
        "labels":         labels,
    }


def _build_prompt_text(system: str, user: str, tokenizer: Any) -> str:
    """Dùng chat template của tokenizer để tạo prompt text (không có assistant turn)."""
    chat = [
        {"role": "system",    "content": system},
        {"role": "user",      "content": user},
    ]
    return tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


# Corrupt date helper (sinh negative samples cho date_arith cls) ──────────────

def _corrupt_date_en(gold: str, rng: random.Random) -> str:
    """Dịch chuyển ngày ±1–30 days để tạo negative sample cho P_cls."""
    try:
        mm, dd, yyyy = (int(x) for x in gold.split("/"))
        d = datetime.date(yyyy, mm, dd)
        delta = rng.choice([-1, 1]) * rng.randint(1, 30)
        d2 = d + datetime.timedelta(days=delta)
        return f"{d2.month:02d}/{d2.day:02d}/{d2.year:04d}"
    except Exception:
        return gold   # nếu parse fail, dùng gold (label="yes")


def _corrupt_date_vi(gold: str, rng: random.Random) -> str:
    """Shift month ±1–6 để tạo negative sample cho VI date."""
    import re
    m = re.match(r"Tháng (\d+), (\d+)", gold)
    if not m:
        return gold
    month, year = int(m.group(1)), int(m.group(2))
    delta = rng.choice([-1, 1]) * rng.randint(1, 6)
    month2 = month + delta
    if month2 < 1:
        month2 += 12; year -= 1
    elif month2 > 12:
        month2 -= 12; year += 1
    return f"Tháng {month2}, {year}"


# Main builder ────────────────────────────────────────────────────────────────

class DualPromptBuilder:
    """Xây dựng P_gen và P_cls cho một Sample.

    Mỗi Sample có thể cho ra 1 hoặc 2 P_cls (positive + negative cho date_arith).
    """

    def __init__(self, tokenizer: Any, max_seq_len: int = 512, seed: int = 42):
        self.tokenizer   = tokenizer
        self.max_seq_len = max_seq_len
        self._rng        = random.Random(seed)

    def build_gen(self, sample: dict) -> dict:
        """Trả về dict {input_ids, attention_mask, labels} cho P_gen."""
        task, lang = sample["task"], sample["language"]
        system      = _GEN_SYS[(task, lang)]
        user_text   = _user_gen(sample)
        prompt_text = _build_prompt_text(system, user_text, self.tokenizer)
        answer_text = sample["gold"]
        return _tokenize_with_labels(
            prompt_text, answer_text, self.tokenizer, self.max_seq_len
        )

    def build_cls(self, sample: dict) -> list[dict]:
        """Trả về list[dict] P_cls examples cho sample.

        duration  → 1 example (candidate + gold label "yes"/"no")
        date_arith → 2 examples: positive (gold → "yes") + negative (corrupt → "no")
        """
        task, lang  = sample["task"], sample["language"]
        system      = _CLS_SYS[(task, lang)]
        gold        = sample["gold"]

        if task == "duration":
            user_text   = _user_cls(sample, proposed_answer=gold)  # candidate in meta
            prompt_text = _build_prompt_text(system, user_text, self.tokenizer)
            # gold label: "yes" hoặc "no" — đây chính là nhãn cần học
            return [_tokenize_with_labels(
                prompt_text, gold, self.tokenizer, self.max_seq_len
            )]

        # date_arith — positive + negative
        examples = []

        # Positive: gold date → "yes"
        user_pos    = _user_cls(sample, proposed_answer=gold)
        prompt_pos  = _build_prompt_text(system, user_pos, self.tokenizer)
        examples.append(_tokenize_with_labels(
            prompt_pos, "yes", self.tokenizer, self.max_seq_len
        ))

        # Negative: corrupted date → "no"
        corrupt = (
            _corrupt_date_en(gold, self._rng) if lang == "en"
            else _corrupt_date_vi(gold, self._rng)
        )
        if corrupt != gold:   # chỉ thêm nếu corrupt thực sự khác gold
            user_neg   = _user_cls(sample, proposed_answer=corrupt)
            prompt_neg = _build_prompt_text(system, user_neg, self.tokenizer)
            examples.append(_tokenize_with_labels(
                prompt_neg, "no", self.tokenizer, self.max_seq_len
            ))

        return examples

    def build_cls_prompt_only(self, sample: dict) -> list[int]:
        """Trả về prompt_ids (không có answer) để dùng trong log-prob evaluation."""
        task, lang = sample["task"], sample["language"]
        system     = _CLS_SYS[(task, lang)]
        user_text  = _user_cls(
            sample,
            proposed_answer=(sample.get("meta") or {}).get("candidate_answer", sample["gold"]),
        )
        prompt_text = _build_prompt_text(system, user_text, self.tokenizer)
        return self.tokenizer.encode(prompt_text, add_special_tokens=False)
