"""Hybrid SymbolicCoT+ method.

Architecture:
  Layer 0 – Rule-based Fast Path  : temporal_extractor (0 LLM calls)
  Layer 1 – Planner (CoT)         : LLM decomposes question into steps
  Layer 2A – Guided Synthesis     : LLM reasons + writes code (date_arith)
  Layer 2B – CoT + KB Path        : LLM reasons with KB context (duration)
  Layer 3 – Symbolic Execution    : execute_program() sandboxed
  Layer 4 – Self-Correction       : LLM fixes runtime errors
  Layer 5 – Retrospective Verify  : LLM checks reasoning faithfulness
  Layer 6 – Vote + Fallback       : majority vote → zero-shot fallback

Smart routing:
  rule-based succeeds → return immediately (Layer 0, 0 LLM calls)
  duration task       → Layer 1 → 2B (CoT + KB, no code synthesis)
  date_arith task     → Layer 1 → 2A → 3 → 4 → 5
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Sequence

from ..data.schema import Sample
from ..models.base import ChatLM, ChatMessage
from ..prompts.templates import build_messages
from ..utils.temporal_executor import execute_program, verify_answer
from ..utils.temporal_extractor import (
    _match_activity,
    solve_date_arith,
    solve_duration,
)
from .base import gen_kwargs_for

# ---------------------------------------------------------------------------
# Temperatures per hypothesis (index 0 = greedy, 1+ = sampling)
# ---------------------------------------------------------------------------
_HYPOTHESIS_TEMPS = [0.0, 0.3, 0.6, 0.9]
_HYPOTHESIS_DO_SAMPLE = [False, True, True, True]

# ---------------------------------------------------------------------------
# Layer 1 – Planner prompts
# ---------------------------------------------------------------------------
_PLANNER_SYSTEM: dict[str, str] = {
    "en": (
        "You are a temporal reasoning planner. "
        "Given a question, output a numbered step-by-step plan to solve it. "
        "Be concise. Do NOT compute the answer — only outline the steps. "
        "Maximum 5 steps."
    ),
    "vi": (
        "Bạn là bộ lập kế hoạch suy luận thời gian. "
        "Cho câu hỏi, hãy xuất kế hoạch từng bước có đánh số để giải quyết. "
        "Ngắn gọn. KHÔNG tính toán kết quả — chỉ phác thảo các bước. "
        "Tối đa 5 bước."
    ),
}

# ---------------------------------------------------------------------------
# Layer 2A – Guided Synthesis prompts (date_arith)
# Combines CoT reasoning steps + Python code in one LLM call
# ---------------------------------------------------------------------------
_GUIDED_SYNTHESIS_SYSTEM: dict[tuple[str, str], str] = {
    ("date_arith", "en"): (
        "You are a temporal computation engine.\n"
        "Given a question and an optional plan, first write brief reasoning steps "
        "following the plan, then write a Python program that computes the answer.\n\n"
        "Output format:\n"
        "Reasoning:\n"
        "1. <step>\n"
        "2. <step>\n"
        "Code:\n"
        "```python\n"
        "<code>\n"
        "```\n\n"
        "Code rules:\n"
        "- Use only: date, datetime, timedelta, relativedelta (already in scope)\n"
        "- MONDAY=0 … SUNDAY=6 available\n"
        "- Assign final date to `answer` in MM/DD/YYYY format\n"
        "- No imports, no print(), max 15 lines"
    ),
    ("date_arith", "vi"): (
        "Bạn là engine tính toán thời gian.\n"
        "Cho câu hỏi và kế hoạch tùy chọn, hãy viết bước suy luận ngắn gọn "
        "theo kế hoạch, sau đó viết chương trình Python tính kết quả.\n\n"
        "Định dạng đầu ra:\n"
        "Reasoning:\n"
        "1. <bước>\n"
        "2. <bước>\n"
        "Code:\n"
        "```python\n"
        "<code>\n"
        "```\n\n"
        "Quy tắc code:\n"
        "- Chỉ dùng: date, datetime, timedelta, relativedelta (đã có trong scope)\n"
        "- Gán kết quả vào `answer` theo mẫu 'Tháng M, YYYY'\n"
        "- Không import, không print(), tối đa 15 dòng"
    ),
}

# ---------------------------------------------------------------------------
# Layer 2B – Duration CoT + KB prompts
# No code synthesis — LLM reasons step by step aided by KB range hint
# ---------------------------------------------------------------------------
_DURATION_COT_SYSTEM: dict[str, str] = {
    "en": (
        "You are a duration plausibility evaluator. "
        "Think step by step then give a yes/no answer.\n\n"
        "Steps:\n"
        "1. Identify the event type from context\n"
        "2. Recall the typical duration range for this event\n"
        "3. Compare candidate duration with the typical range\n"
        "4. Conclude\n\n"
        "Output format:\n"
        "Reasoning: <your step-by-step reasoning>\n"
        "Answer: yes\n"
        "or\n"
        "Answer: no"
    ),
    "vi": (
        "Bạn là bộ đánh giá tính hợp lý khoảng thời gian. "
        "Suy nghĩ từng bước rồi đưa ra câu trả lời yes/no.\n\n"
        "Các bước:\n"
        "1. Xác định loại sự kiện từ ngữ cảnh\n"
        "2. Nhớ lại khoảng thời gian điển hình cho sự kiện này\n"
        "3. So sánh khoảng thời gian đề xuất với khoảng điển hình\n"
        "4. Kết luận\n\n"
        "Định dạng đầu ra:\n"
        "Reasoning: <suy luận từng bước>\n"
        "Answer: yes\n"
        "hoặc\n"
        "Answer: no"
    ),
}

# ---------------------------------------------------------------------------
# Layer 5 – Retrospective Verifier prompt
# ---------------------------------------------------------------------------
_VERIFIER_SYSTEM = (
    "You are a faithful reasoning verifier for temporal questions. "
    "Given reasoning steps and a final answer, check:\n"
    "1. Are the reasoning steps logically consistent with each other?\n"
    "2. Does the final answer follow from the reasoning?\n"
    "3. Are there any arithmetic errors?\n\n"
    "Output ONLY: VALID or INVALID\n"
    "If INVALID, add one brief reason on the same line."
)

# ---------------------------------------------------------------------------
# Layer 4 – Self-correction prompts
# ---------------------------------------------------------------------------
_CORRECTION_SYSTEM = (
    "You are a Python debugging assistant for temporal reasoning programs. "
    "Fix the given program so it executes correctly and assigns the right value "
    "to the variable `answer`. "
    "Output ONLY the corrected Python code, nothing else."
)
_CORRECTION_SYSTEM_VI = (
    "Bạn là trợ lý debug Python cho các chương trình tính toán thời gian. "
    "Sửa chương trình đã cho để nó chạy đúng và gán giá trị đúng vào biến `answer`. "
    "Chỉ xuất code Python đã sửa, không có gì khác."
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_yes_no(text: str) -> str | None:
    """Extract yes/no from CoT output. Prefers explicit 'Answer:' label."""
    m = re.search(r"\bAnswer:\s*(yes|no)\b", text, re.I)
    if m:
        return m.group(1).lower()
    matches = re.findall(r"\b(yes|no)\b", text.lower())
    return matches[-1] if matches else None


def _extract_reasoning(text: str) -> str:
    """Extract Reasoning section from guided synthesis / CoT output."""
    m = re.search(r"Reasoning:\s*(.*?)(?:Code:|```|Answer:|$)", text, re.S | re.I)
    return m.group(1).strip() if m else text[:500]


# ---------------------------------------------------------------------------
# Main method
# ---------------------------------------------------------------------------

class SymbolicCoTMethod:
    """Hybrid SymbolicCoT+: rule-based fast path + CoT planner +
    guided synthesis (date_arith) + CoT KB path (duration) +
    retrospective verification + self-correction + majority vote.

    predict(sample) -> str   ← interface unchanged.
    """

    name = "symbolic_cot"

    def __init__(
        self,
        model: ChatLM,
        enable_thinking: bool = False,
        n_hypotheses: int = 1,
        max_correction_attempts: int = 1,
        use_planner: bool = True,
        use_kb_for_duration: bool = True,
        use_retrospective_verify: bool = True,
    ) -> None:
        self.model = model
        self.enable_thinking = enable_thinking
        self.n_hypotheses = n_hypotheses
        self.max_correction_attempts = max_correction_attempts
        self.use_planner = use_planner
        self.use_kb_for_duration = use_kb_for_duration
        self.use_retrospective_verify = use_retrospective_verify

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict(self, sample: Sample) -> str:
        task = sample["task"]
        lang = sample.get("language", "en")

        # Layer 0: Rule-based fast path (0 LLM calls)
        rule_result = self._try_rule_based(sample, task)
        if rule_result is not None:
            return rule_result

        # Layer 1: Planner — CoT decomposition
        plan: str | None = None
        if self.use_planner:
            plan = self._plan(sample, lang)

        # Smart routing by task type
        if task == "duration":
            return self._duration_cot_path(sample, lang, plan)
        return self._date_arith_path(sample, lang, plan)

    # ------------------------------------------------------------------
    # Layer 0: Rule-based fast path
    # ------------------------------------------------------------------

    def _try_rule_based(self, sample: Sample, task: str) -> str | None:
        if task == "date_arith":
            return solve_date_arith(sample)
        if task == "duration":
            return solve_duration(sample)
        return None

    # ------------------------------------------------------------------
    # Layer 1: Planner (CoT, Wei et al.)
    # ------------------------------------------------------------------

    def _plan(self, sample: Sample, lang: str) -> str | None:
        sys_prompt = _PLANNER_SYSTEM.get(lang, _PLANNER_SYSTEM["en"])
        q = sample.get("question", "")
        ctx = (sample.get("context") or "").strip()
        user_content = f"Context: {ctx}\nQuestion: {q}" if ctx else f"Question: {q}"
        msgs = [
            ChatMessage(role="system", content=sys_prompt),
            ChatMessage(role="user", content=user_content),
        ]
        raw = self.model.generate(
            msgs,
            max_new_tokens=150,
            temperature=0.0,
            do_sample=False,
            enable_thinking=False,
        )
        return raw.strip() or None

    # ------------------------------------------------------------------
    # Layer 2B: Duration CoT + KB path (no code synthesis)
    # ------------------------------------------------------------------

    def _duration_cot_path(self, sample: Sample, lang: str, plan: str | None) -> str:
        context = sample.get("context") or ""
        question = sample.get("question") or ""
        candidate = (sample.get("meta") or {}).get("candidate_answer", "")

        # KB range hint from activity knowledge base
        kb_hint = ""
        if self.use_kb_for_duration:
            act_range = _match_activity(context, question)
            if act_range:
                lo, hi = act_range
                kb_hint = (
                    f"\nKnowledge base hint: typical duration for this event is "
                    f"{lo / 60:.0f} min – {hi / 3600:.1f} h."
                )

        parts: list[str] = []
        if context:
            parts.append(f"Context: {context}")
        parts.append(f"Question: {question}")
        parts.append(f"Candidate duration: {candidate}")
        if plan:
            parts.append(f"Plan:\n{plan}")
        if kb_hint:
            parts.append(kb_hint)
        user_content = "\n".join(parts)

        sys_prompt = _DURATION_COT_SYSTEM.get(lang, _DURATION_COT_SYSTEM["en"])
        msgs = [
            ChatMessage(role="system", content=sys_prompt),
            ChatMessage(role="user", content=user_content),
        ]
        raw = self.model.generate(
            msgs,
            max_new_tokens=250,
            temperature=0.0,
            do_sample=False,
            enable_thinking=False,
        )

        answer = _extract_yes_no(raw)

        # Layer 5: Retrospective verify
        if answer and self.use_retrospective_verify:
            reasoning = _extract_reasoning(raw) or raw
            if not self._retrospective_verify(reasoning, answer, sample):
                raw2 = self.model.generate(
                    msgs,
                    max_new_tokens=250,
                    temperature=0.5,
                    do_sample=True,
                    enable_thinking=False,
                )
                answer2 = _extract_yes_no(raw2)
                if answer2:
                    answer = answer2

        return answer if answer else self._fallback(sample)

    # ------------------------------------------------------------------
    # Layer 2A + 3 + 4: Date arith guided synthesis path
    # ------------------------------------------------------------------

    def _date_arith_path(self, sample: Sample, lang: str, plan: str | None) -> str:
        candidates: list[str] = []
        any_result: str | None = None

        for h in range(self.n_hypotheses):
            result = self._run_hypothesis(sample, lang, h, plan)
            if result is not None:
                any_result = result
                candidates.append(result)

        if candidates:
            winner = self._vote(candidates)
            # Layer 5: Retrospective verify on winner
            if self.use_retrospective_verify and plan:
                if not self._retrospective_verify(plan, winner, sample):
                    counts = Counter(c.strip().lower() for c in candidates)
                    ordered = counts.most_common()
                    if len(ordered) > 1:
                        runner_up_lower = ordered[1][0]
                        for c in candidates:
                            if c.strip().lower() == runner_up_lower:
                                return c.strip()
            return winner

        if any_result is not None:
            return any_result
        return self._fallback(sample)

    def _run_hypothesis(
        self,
        sample: Sample,
        lang: str,
        h_idx: int,
        plan: str | None,
    ) -> str | None:
        temp = _HYPOTHESIS_TEMPS[min(h_idx, len(_HYPOTHESIS_TEMPS) - 1)]
        do_sample = _HYPOTHESIS_DO_SAMPLE[min(h_idx, len(_HYPOTHESIS_DO_SAMPLE) - 1)]
        task = sample["task"]

        # Layer 2A+3: Guided synthesis (CoT reasoning + code)
        program = self._synthesize_guided(sample, task, lang, plan, temp, do_sample)
        if not program.strip():
            return None

        last_answer: str | None = None
        for attempt in range(self.max_correction_attempts + 1):
            # Layer 3: Execute
            answer, error = execute_program(program, task, lang)
            last_answer = answer

            if answer is not None and verify_answer(answer, task, lang):
                return answer

            # Layer 4: Self-correct on runtime error
            if attempt < self.max_correction_attempts and error is not None:
                program = self._correct_program(program, error, sample, task, lang)
                if not program.strip():
                    break

        return last_answer

    # ------------------------------------------------------------------
    # Layer 2A+3: Guided synthesis — CoT reasoning steps + Python code
    # ------------------------------------------------------------------

    def _synthesize_guided(
        self,
        sample: Sample,
        task: str,
        lang: str,
        plan: str | None,
        temperature: float,
        do_sample: bool,
    ) -> str:
        sys_prompt = _GUIDED_SYNTHESIS_SYSTEM.get(
            (task, lang),
            _GUIDED_SYNTHESIS_SYSTEM[("date_arith", "en")],
        )
        q = sample.get("question", "")
        ctx = (sample.get("context") or "").strip()

        parts: list[str] = []
        if ctx:
            parts.append(f"Context: {ctx}")
        parts.append(f"Question: {q}")
        if plan:
            parts.append(f"Plan:\n{plan}")
        user_content = "\n".join(parts)

        msgs = [
            ChatMessage(role="system", content=sys_prompt),
            ChatMessage(role="user", content=user_content),
        ]
        return self.model.generate(
            msgs,
            max_new_tokens=300,
            temperature=temperature,
            do_sample=do_sample,
            enable_thinking=False,
        )

    # ------------------------------------------------------------------
    # Layer 4: Self-correction
    # ------------------------------------------------------------------

    def _correct_program(
        self,
        program: str,
        error: str,
        sample: Sample,
        task: str,
        lang: str,
    ) -> str:
        sys_prompt = _CORRECTION_SYSTEM_VI if lang == "vi" else _CORRECTION_SYSTEM
        q = sample.get("question", "")
        if lang == "vi":
            user_content = (
                f"Lỗi: {error}\n\nChương trình lỗi:\n{program}\n\n"
                f"Câu hỏi: {q}\n\nHãy viết chương trình đã sửa:"
            )
        else:
            user_content = (
                f"Error: {error}\n\nBuggy program:\n{program}\n\n"
                f"Question: {q}\n\nWrite the corrected program:"
            )
        msgs = [
            ChatMessage(role="system", content=sys_prompt),
            ChatMessage(role="user", content=user_content),
        ]
        return self.model.generate(
            msgs,
            max_new_tokens=200,
            temperature=0.0,
            do_sample=False,
            enable_thinking=False,
        )

    # ------------------------------------------------------------------
    # Layer 5: Retrospective Verifier
    # ------------------------------------------------------------------

    def _retrospective_verify(
        self,
        reasoning: str,
        answer: str,
        sample: Sample,
    ) -> bool:
        q = sample.get("question", "")
        user_content = (
            f"Question: {q}\n\n"
            f"Reasoning:\n{reasoning}\n\n"
            f"Final Answer: {answer}\n\n"
            "Is this reasoning faithful and consistent with the answer?"
        )
        msgs = [
            ChatMessage(role="system", content=_VERIFIER_SYSTEM),
            ChatMessage(role="user", content=user_content),
        ]
        raw = self.model.generate(
            msgs,
            max_new_tokens=50,
            temperature=0.0,
            do_sample=False,
            enable_thinking=False,
        )
        text = raw.upper()
        return "VALID" in text and "INVALID" not in text

    # ------------------------------------------------------------------
    # Layer 6: Majority Vote
    # ------------------------------------------------------------------

    @staticmethod
    def _vote(candidates: Sequence[str]) -> str:
        counts = Counter(c.strip().lower() for c in candidates)
        winner_lower, _ = counts.most_common(1)[0]
        for c in candidates:
            if c.strip().lower() == winner_lower:
                return c.strip()
        return candidates[0].strip()

    # ------------------------------------------------------------------
    # Fallback: zero-shot direct generation
    # ------------------------------------------------------------------

    def _fallback(self, sample: Sample) -> str:
        msgs = build_messages(sample, shots=())
        kwargs = gen_kwargs_for(sample["task"])
        kwargs["enable_thinking"] = self.enable_thinking
        if self.enable_thinking:
            kwargs["max_new_tokens"] = max(kwargs["max_new_tokens"], 256)
        return self.model.generate(msgs, **kwargs)

    # ------------------------------------------------------------------
    # Batch predict (falls back to per-sample for hybrid layers)
    # ------------------------------------------------------------------

    def predict_batch(self, samples: list[Sample]) -> list[str]:
        return [self.predict(s) for s in samples]
