"""Extract-Normalize method: deterministic rule-based solver with LLM fallback.

Pipeline per sample:
  1. Extract  — regex pull all temporal expressions (no LLM)
  2. Normalize — resolve relative → absolute date / yes/no
  3. Format   — task-correct string
  4. Fallback — zero-shot LLM only when steps 1-3 return None

Advantages over pure LLM:
  - Deterministic and instant for simple cases (no GPU needed)
  - No token generation = no parse errors on trivial questions
  - LLM budget preserved for genuinely hard / ambiguous cases

Interface: predict(sample) -> str   (identical to all other methods)
"""

from __future__ import annotations

from typing import Any

from ..data.schema import Sample
from ..models.base import ChatLM, ChatMessage
from ..prompts.templates import build_messages
from ..utils.temporal_extractor import solve_date_arith, solve_duration
from .base import gen_kwargs_for


class ExtractNormalizeMethod:
    """Rule-based Extract→Normalize→Format with LLM fallback on ambiguity."""

    name = "extract_normalize"

    def __init__(
        self,
        model: ChatLM,
        enable_thinking: bool = False,
    ) -> None:
        self.model = model
        self.enable_thinking = enable_thinking
        self._rule_hits = 0
        self._llm_hits = 0

    # ── Public interface ──────────────────────────────────────────────────────

    def predict(self, sample: Sample) -> str:
        task = sample["task"]
        result: str | None = None

        if task == "date_arith":
            result = solve_date_arith(sample)
        elif task == "duration":
            result = solve_duration(sample)

        if result is not None:
            self._rule_hits += 1
            return result

        self._llm_hits += 1
        return self._llm_fallback(sample)

    @property
    def rule_ratio(self) -> float:
        total = self._rule_hits + self._llm_hits
        return self._rule_hits / total if total else 0.0

    # ── Fallback ──────────────────────────────────────────────────────────────

    def _llm_fallback(self, sample: Sample) -> str:
        msgs = build_messages(sample, shots=())  # zero-shot
        kwargs = gen_kwargs_for(sample["task"])
        kwargs["enable_thinking"] = self.enable_thinking
        return self.model.generate(msgs, **kwargs)
