"""Prompt templates for the arithmetic MCQ task.

Two evaluation strategies are provided:

1. **Direct MCQ** (build_mcq_messages)
   Single inference: model sees Q + 4 options → outputs A/B/C/D.
   Fast, but relies on the model doing both computation and selection.

2. **Compute-then-Match** (build_compute_messages + build_match_messages)
   Stage 1 — Compute: model solves the question and outputs a raw answer string.
   Stage 2 — Match : model sees the raw answer + 4 options → outputs A/B/C/D.
   Separates arithmetic from option selection; better for small (4B) models
   that struggle to simultaneously compute and format as a letter.

Usage
-----
    # Direct MCQ
    from src.prompts.mcq_templates import build_mcq_messages
    msgs = build_mcq_messages(sample, shots=shots_list)
    raw  = model.generate(msgs, max_new_tokens=10, ...)

    # Compute-then-Match
    from src.prompts.mcq_templates import build_compute_messages, build_match_messages
    compute_msgs = build_compute_messages(sample, shots=shots_list)
    computed_raw = model.generate(compute_msgs, max_new_tokens=200, ...)
    match_msgs   = build_match_messages(computed_raw, sample)
    letter_raw   = model.generate(match_msgs, max_new_tokens=10, ...)
"""

from __future__ import annotations

import re
from typing import Sequence

from ..data.schema import McqSample
from ..models.base import ChatMessage

# ─────────────────────────────────────────────────────────────────────────────
# Strategy 1 — Direct MCQ
# ─────────────────────────────────────────────────────────────────────────────

MCQ_SYSTEM = (
    "You are a temporal reasoning assistant. "
    "Read the question and the four answer options carefully. "
    "Output ONLY the letter of the correct answer: A, B, C, or D. "
    "No explanation, no punctuation — just the single letter."
)


def render_user(s: McqSample) -> str:
    return (
        f"Question: {s['question']}\n"
        f"A) {s['option_a']}\n"
        f"B) {s['option_b']}\n"
        f"C) {s['option_c']}\n"
        f"D) {s['option_d']}"
    )


def render_assistant(s: McqSample) -> str:
    return s["gold"]


def build_mcq_messages(
    sample: McqSample,
    shots: Sequence[McqSample] = (),
) -> list[ChatMessage]:
    """Build messages for direct A/B/C/D prediction."""
    msgs: list[ChatMessage] = [ChatMessage(role="system", content=MCQ_SYSTEM)]
    for shot in shots:
        msgs.append(ChatMessage(role="user",      content=render_user(shot)))
        msgs.append(ChatMessage(role="assistant", content=render_assistant(shot)))
    msgs.append(ChatMessage(role="user", content=render_user(sample)))
    return msgs


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 2 — Compute-then-Match
# ─────────────────────────────────────────────────────────────────────────────

# Stage 1: compute the raw answer from the question text alone.
# The model must NOT see the options — we want an independent computation.
COMPUTE_SYSTEM = (
    "You are a temporal arithmetic solver. "
    "Solve the question step by step, then write your final answer on the last line "
    "starting with 'Answer:'. "
    "Example last line: Answer: 08:57"
)

# Stage 2: given the computed answer, pick the matching option letter.
MATCH_SYSTEM = (
    "You are given a computed answer and four answer options. "
    "Decide which option best matches the computed answer. "
    "Output ONLY the single letter: A, B, C, or D. "
    "No explanation."
)


def _option_text(s: McqSample, letter: str) -> str:
    """Return the text of the option corresponding to the given letter."""
    return {
        "A": s["option_a"], "B": s["option_b"],
        "C": s["option_c"], "D": s["option_d"],
    }[letter.upper()]


def render_compute_user(s: McqSample) -> str:
    """Question only — no options shown during computation stage."""
    return f"Question: {s['question']}"


def render_compute_assistant(s: McqSample) -> str:
    """Gold answer text (not the letter) used as the shot assistant turn."""
    return f"Answer: {_option_text(s, s['gold'])}"


def render_match_user(computed: str, s: McqSample) -> str:
    """Format the match prompt: computed answer + 4 options."""
    # Strip 'Answer:' prefix if the model included it
    clean = re.sub(r"(?i)^answer\s*:\s*", "", computed.strip()).strip()
    return (
        f"Computed answer: {clean}\n\n"
        f"Options:\n"
        f"A) {s['option_a']}\n"
        f"B) {s['option_b']}\n"
        f"C) {s['option_c']}\n"
        f"D) {s['option_d']}\n\n"
        f"Which letter (A/B/C/D) matches the computed answer?"
    )


def build_compute_messages(
    sample: McqSample,
    shots: Sequence[McqSample] = (),
) -> list[ChatMessage]:
    """Stage 1: messages for computing the raw answer (no options shown)."""
    msgs: list[ChatMessage] = [ChatMessage(role="system", content=COMPUTE_SYSTEM)]
    for shot in shots:
        msgs.append(ChatMessage(role="user",      content=render_compute_user(shot)))
        msgs.append(ChatMessage(role="assistant", content=render_compute_assistant(shot)))
    msgs.append(ChatMessage(role="user", content=render_compute_user(sample)))
    return msgs


def build_match_messages(computed: str, sample: McqSample) -> list[ChatMessage]:
    """Stage 2: messages for matching computed answer to A/B/C/D.

    No few-shot examples here — the task is trivial enough for zero-shot.
    """
    return [
        ChatMessage(role="system", content=MATCH_SYSTEM),
        ChatMessage(role="user",   content=render_match_user(computed, sample)),
    ]


def extract_computed_answer(raw: str) -> str | None:
    """Pull the answer value out of a Stage-1 model response.

    Looks for the last 'Answer: ...' line. Returns None if not found so the
    caller knows the response was truncated and should retry with more tokens.
    """
    matches = re.findall(r"(?i)answer\s*:\s*(.+?)(?:\n|$)", raw)
    if matches:
        return matches[-1].strip()
    return None


def extract_letter(raw: str) -> str | None:
    """Extract the first A/B/C/D letter from a model response."""
    m = re.search(r"\b([A-D])\b", raw.strip().upper())
    if m:
        return m.group(1)
    first = raw.strip().upper()[:1]
    return first if first in "ABCD" else None
