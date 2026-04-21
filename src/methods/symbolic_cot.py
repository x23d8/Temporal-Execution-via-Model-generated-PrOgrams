"""Symbolic Chain-of-Thought method.

Hiện thực hoá kiến trúc 5-layer từ note.txt:

  Layer 1 – Temporal Understanding  : LLM trích structured temporal info
  Layer 2 – Temporal Normalization  : tích hợp trong prompt synthesis
  Layer 3 – Program Synthesis       : LLM sinh Python datetime program
  Layer 4 – Symbolic Execution      : execute_program() chạy program
  Layer 5 – Verification & Correction: verify_answer() + self-correction loop

Thêm:
  - Multi-Hypothesis Voting (§4): chạy n_hypotheses chương trình độc lập
  - Self-Correction Loop (§3): max_correction_attempts lần feedback → regen

Interface KHÔNG THAY ĐỔI: predict(sample) -> str   (raw string cho extractor)
Fallback: nếu mọi hypothesis fail → gọi zero-shot direct generation.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Sequence

from ..data.schema import Sample
from ..models.base import ChatLM, ChatMessage
from ..prompts.templates import build_messages
from ..utils.temporal_executor import execute_program, verify_answer
from .base import gen_kwargs_for

# ---------------------------------------------------------------------------
# Temperatures cho từng hypothesis (index 0 = greedy, 1+ = sampling)
# ---------------------------------------------------------------------------

_HYPOTHESIS_TEMPS = [0.0, 0.3, 0.6, 0.9]
_HYPOTHESIS_DO_SAMPLE = [False, True, True, True]

# ---------------------------------------------------------------------------
# System prompts cho Program Synthesis (Layer 3)
# Không sửa templates.py — đây là internal prompts của method này
# ---------------------------------------------------------------------------

_SYNTHESIS_SYSTEM: dict[tuple[str, str], str] = {
    ("date_arith", "en"): (
        "You are a temporal computation engine. "
        "Given a date arithmetic question, write a short Python program "
        "that computes the answer using datetime arithmetic.\n\n"
        "Rules:\n"
        "- Use only: date, datetime, timedelta, relativedelta (already in scope)\n"
        "- MONDAY=0 … SUNDAY=6 are available for weekday checks\n"
        "- Assign the final date string to a variable named `answer` "
        "in MM/DD/YYYY format (e.g. answer = '05/01/2021')\n"
        "- Do NOT import anything; do NOT use print()\n"
        "- Maximum 15 lines\n\n"
        "Output ONLY the Python code, nothing else."
    ),
    ("date_arith", "vi"): (
        "Bạn là engine tính toán thời gian. "
        "Cho câu hỏi về phép tính ngày tháng, hãy viết một chương trình Python ngắn "
        "để tính câu trả lời bằng datetime arithmetic.\n\n"
        "Quy tắc:\n"
        "- Chỉ dùng: date, datetime, timedelta, relativedelta (đã có trong scope)\n"
        "- Gán chuỗi ngày kết quả vào biến `answer` theo đúng mẫu 'Tháng M, YYYY' "
        "(ví dụ: answer = 'Tháng 4, 2021')\n"
        "- Không import; không dùng print()\n"
        "- Tối đa 15 dòng\n\n"
        "Chỉ xuất code Python, không có gì khác."
    ),
    ("duration", "en"): (
        "You are a plausibility evaluation engine. "
        "Given a context and a candidate duration, write a Python program "
        "to determine whether the candidate is a plausible duration for the event.\n\n"
        "Rules:\n"
        "- Assign either 'yes' or 'no' to a variable named `answer`\n"
        "- Estimate min_seconds and max_seconds for this type of event\n"
        "- Convert the candidate to candidate_seconds\n"
        "- answer = 'yes' if min_seconds <= candidate_seconds <= max_seconds else 'no'\n"
        "- Use only basic arithmetic; do NOT import anything; do NOT use print()\n"
        "- Comment each estimate briefly\n"
        "- Maximum 20 lines\n\n"
        "Output ONLY the Python code, nothing else."
    ),
    ("duration", "vi"): (
        "Bạn là engine đánh giá tính hợp lý. "
        "Cho ngữ cảnh và khoảng thời gian đề xuất, hãy viết chương trình Python "
        "để xác định xem khoảng thời gian đó có hợp lý cho sự kiện không.\n\n"
        "Quy tắc:\n"
        "- Gán 'yes' hoặc 'no' vào biến `answer`\n"
        "- Ước tính min_seconds và max_seconds cho loại sự kiện này\n"
        "- Đổi ứng viên sang candidate_seconds\n"
        "- answer = 'yes' if min_seconds <= candidate_seconds <= max_seconds else 'no'\n"
        "- Chỉ dùng arithmetic cơ bản; không import; không dùng print()\n"
        "- Comment ngắn mỗi ước tính\n"
        "- Tối đa 20 dòng\n\n"
        "Chỉ xuất code Python, không có gì khác."
    ),
}

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
# User message renderers cho synthesis
# ---------------------------------------------------------------------------

def _synthesis_user(sample: Sample, task: str, lang: str) -> str:
    q = sample["question"]
    ctx = (sample.get("context") or "").strip()
    cand = (sample.get("meta") or {}).get("candidate_answer", "")

    if task == "date_arith":
        if lang == "vi":
            parts = [f"Câu hỏi: {q}"]
            if ctx:
                parts.insert(0, f"Ngữ cảnh: {ctx}")
            return "\n".join(parts)
        return f"Question: {q}"

    # duration
    if lang == "vi":
        lines = []
        if ctx:
            lines.append(f"Ngữ cảnh: {ctx}")
        lines.append(f"Câu hỏi: {q}")
        if cand:
            lines.append(f"Khoảng thời gian đề xuất: {cand}")
        return "\n".join(lines)

    lines = []
    if ctx:
        lines.append(f"Context: {ctx}")
    lines.append(f"Question: {q}")
    if cand:
        lines.append(f"Candidate duration: {cand}")
    return "\n".join(lines)


def _correction_user(
    program: str,
    error: str,
    sample: Sample,
    lang: str,
) -> str:
    q = sample["question"]
    if lang == "vi":
        return (
            f"Lỗi: {error}\n\n"
            f"Chương trình lỗi:\n{program}\n\n"
            f"Câu hỏi: {q}\n\n"
            "Hãy viết chương trình đã sửa:"
        )
    return (
        f"Error: {error}\n\n"
        f"Buggy program:\n{program}\n\n"
        f"Question: {q}\n\n"
        "Write the corrected program:"
    )


# ---------------------------------------------------------------------------
# Main method
# ---------------------------------------------------------------------------


class SymbolicCoTMethod:
    """Symbolic Chain-of-Thought: LLM sinh program → symbolic exec → verify → vote.

    predict(sample) -> str   ← interface không đổi; output qua extractor hiện tại.
    """

    name = "symbolic_cot"

    def __init__(
        self,
        model: ChatLM,
        enable_thinking: bool = False,
        n_hypotheses: int = 1,
        max_correction_attempts: int = 1,
    ) -> None:
        self.model = model
        self.enable_thinking = enable_thinking
        self.n_hypotheses = n_hypotheses
        self.max_correction_attempts = max_correction_attempts

    # ------------------------------------------------------------------
    # Public interface (unchanged)
    # ------------------------------------------------------------------

    def predict(self, sample: Sample) -> str:
        task = sample["task"]
        lang = sample["language"]

        candidates: list[str] = []
        any_result: str | None = None

        for h in range(self.n_hypotheses):
            result = self._run_hypothesis(sample, task, lang, h)
            if result is not None:
                any_result = result
                candidates.append(result)

        if candidates:
            return self._vote(candidates)

        # Tous les hypotheses ont échoué — fallback zero-shot
        if any_result is not None:
            return any_result
        return self._fallback(sample)

    # ------------------------------------------------------------------
    # Per-hypothesis pipeline (Layers 3–5 + self-correction)
    # ------------------------------------------------------------------

    def _run_hypothesis(
        self,
        sample: Sample,
        task: str,
        lang: str,
        h_idx: int,
    ) -> str | None:
        temp = _HYPOTHESIS_TEMPS[min(h_idx, len(_HYPOTHESIS_TEMPS) - 1)]
        do_sample = _HYPOTHESIS_DO_SAMPLE[min(h_idx, len(_HYPOTHESIS_DO_SAMPLE) - 1)]

        program = self._synthesize_program(sample, task, lang, temp, do_sample)
        if not program.strip():
            return None

        last_answer: str | None = None
        last_error: str | None = "No execution attempted"

        for attempt in range(self.max_correction_attempts + 1):
            answer, error = execute_program(program, task, lang)
            last_answer = answer
            last_error = error

            if answer is not None and verify_answer(answer, task, lang):
                return answer  # Layer 5 passed

            if attempt < self.max_correction_attempts and error is not None:
                program = self._correct_program(program, error, sample, task, lang)
                if not program.strip():
                    break

        # Trả về kết quả cuối dù chưa pass verify (dùng cho voting fallback)
        return last_answer

    # ------------------------------------------------------------------
    # Layer 3: Program Synthesis
    # ------------------------------------------------------------------

    def _synthesize_program(
        self,
        sample: Sample,
        task: str,
        lang: str,
        temperature: float,
        do_sample: bool,
    ) -> str:
        sys_prompt = _SYNTHESIS_SYSTEM.get(
            (task, lang),
            _SYNTHESIS_SYSTEM[("date_arith", "en")],
        )
        user_content = _synthesis_user(sample, task, lang)
        msgs = [
            ChatMessage(role="system", content=sys_prompt),
            ChatMessage(role="user", content=user_content),
        ]
        raw = self.model.generate(
            msgs,
            max_new_tokens=512,
            temperature=temperature,
            do_sample=do_sample,
            enable_thinking=False,  # giữ non-thinking cho tốc độ
        )
        return raw

    # ------------------------------------------------------------------
    # Layer 5 (self-correction): Program Correction
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
        user_content = _correction_user(program, error, sample, lang)
        msgs = [
            ChatMessage(role="system", content=sys_prompt),
            ChatMessage(role="user", content=user_content),
        ]
        raw = self.model.generate(
            msgs,
            max_new_tokens=512,
            temperature=0.0,
            do_sample=False,
            enable_thinking=False,
        )
        return raw

    # ------------------------------------------------------------------
    # Multi-Hypothesis Voting (§4)
    # ------------------------------------------------------------------

    @staticmethod
    def _vote(candidates: Sequence[str]) -> str:
        counts = Counter(c.strip().lower() for c in candidates)
        winner_lower, _ = counts.most_common(1)[0]
        # Trả về candidate gốc (giữ đúng case) match với winner_lower
        for c in candidates:
            if c.strip().lower() == winner_lower:
                return c.strip()
        return candidates[0].strip()

    # ------------------------------------------------------------------
    # Fallback: zero-shot direct generation (dùng templates hiện tại)
    # ------------------------------------------------------------------

    def _fallback(self, sample: Sample) -> str:
        msgs = build_messages(sample, shots=())
        kwargs = gen_kwargs_for(sample["task"])
        kwargs["enable_thinking"] = self.enable_thinking
        if self.enable_thinking:
            kwargs["max_new_tokens"] = max(kwargs["max_new_tokens"], 256)
        return self.model.generate(msgs, **kwargs)
