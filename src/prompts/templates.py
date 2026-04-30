"""Prompt templates cho (task × ngôn ngữ).

Mỗi entry gồm:
- system: chuỗi ràng buộc format.
- render(sample) -> user message.
- few_shot_block(example_samples) -> đoạn text chèn trước câu hỏi thật.

Few-shot dùng messages format: mỗi shot là 1 cặp user/assistant bơm trước user cuối.
"""

from __future__ import annotations

import json as _json
from dataclasses import dataclass
from typing import Callable, Sequence


def _md_json(**kwargs) -> str:
    """Render kwargs as an indented JSON markdown code block.

    Matches the format the local Ollama/Gemma model emits so that HF models
    shown this example in the system prompt or few-shot will reproduce the
    same raw_output structure.
    """
    body = _json.dumps(kwargs, ensure_ascii=False, indent=2)
    return f"```json\n{body}\n```"

from ..data.schema import Sample
from ..models.base import ChatMessage


@dataclass
class PromptTemplate:
    system: str
    render_user: Callable[[Sample], str]
    render_shot_user: Callable[[Sample], str]
    render_shot_assistant: Callable[[Sample], str]
    system_thinking: str | None = None
    render_shot_assistant_thinking: Callable[[Sample], str] | None = None


# ------------------------- Duration EN (UDST) -------------------------
_SYS_DURATION_EN = (
    "You are a careful reasoning model. "
    "First, silently think through whether the duration is plausible given the context. "
    "Do NOT output your reasoning. "
    "Then respond with ONLY one word: 'yes' or 'no'. "
    "No explanation, no punctuation, no extra tokens."
)

_SYS_DURATION_EN_THINK = (
    "You are a careful reasoning model. "
    "Respond with a JSON object in a markdown code block with exactly two keys: "
    '"thinking" (your step-by-step reasoning about plausibility) and "answer" ("yes" or "no"). '
    "Example:\n" + _md_json(
        thinking="Boiling water takes 3-5 minutes, so 3 minutes is plausible.",
        answer="yes",
    )
)


def _user_duration_en(s: Sample) -> str:
    cand = s["meta"].get("candidate_answer", "")
    return (
        f"Context: {s['context']}\n"
        f"Question: {s['question']}\n"
        f"Candidate duration: {cand}\n"
        f"Is this a plausible duration? Answer 'yes' or 'no'."
    )


def _shot_assistant_yes_no(s: Sample) -> str:
    return s["gold"]


def _shot_assistant_yes_no_thinking(s: Sample) -> str:
    return _md_json(thinking="Evaluated plausibility.", answer=s["gold"])


DURATION_EN = PromptTemplate(
    system=_SYS_DURATION_EN,
    system_thinking=_SYS_DURATION_EN_THINK,
    render_user=_user_duration_en,
    render_shot_user=_user_duration_en,
    render_shot_assistant=_shot_assistant_yes_no,
    render_shot_assistant_thinking=_shot_assistant_yes_no_thinking,
)


# ------------------------- Duration VI (VLSP) -------------------------
_SYS_DURATION_VI = (
    "Bạn là một mô hình suy luận cẩn thận. "
    "Trước tiên, hãy tự suy nghĩ (ẩn) xem khoảng thời gian có hợp lý với ngữ cảnh hay không. "
    "KHÔNG được in ra phần suy nghĩ. "
    "Sau đó chỉ trả lời duy nhất một từ: 'yes' hoặc 'no'. "
    "Không giải thích, không thêm ký tự nào khác."
)

_SYS_DURATION_VI_THINK = (
    "Bạn là một mô hình suy luận cẩn thận. "
    "Trả lời bằng một JSON object trong markdown code block có đúng hai key: "
    '"thinking" (quá trình suy luận về tính hợp lý) và "answer" ("yes" hoặc "no"). '
    "Ví dụ:\n" + _md_json(
        thinking="Pha cà phê mất khoảng 5 phút, vì vậy 5 phút là hợp lý.",
        answer="yes",
    )
)


def _user_duration_vi(s: Sample) -> str:
    cand = s["meta"].get("candidate_answer", "")
    return (
        f"Ngữ cảnh: {s['context']}\n"
        f"Câu hỏi: {s['question']}\n"
        f"Khoảng thời gian đề xuất: {cand}\n"
        f"Khoảng thời gian này có hợp lý không? Trả lời 'yes' hoặc 'no'."
    )


def _shot_assistant_yes_no_vi_thinking(s: Sample) -> str:
    return _md_json(thinking="Đánh giá tính hợp lý.", answer=s["gold"])


DURATION_VI = PromptTemplate(
    system=_SYS_DURATION_VI,
    system_thinking=_SYS_DURATION_VI_THINK,
    render_user=_user_duration_vi,
    render_shot_user=_user_duration_vi,
    render_shot_assistant=_shot_assistant_yes_no,
    render_shot_assistant_thinking=_shot_assistant_yes_no_vi_thinking,
)


# ------------------------- DateArith EN (BigBench) -------------------------
_SYS_DATE_EN = (
    "You are a precise date arithmetic solver. "
    "First, compute the answer step by step internally. "
    "Do NOT reveal your reasoning. "
    "Then output ONLY the final date in MM/DD/YYYY format. "
    "No words, no explanation, only the date."
)

_SYS_DATE_EN_THINK = (
    "You are a precise date arithmetic solver. "
    "Respond with a JSON object in a markdown code block with exactly two keys: "
    '"thinking" (your step-by-step reasoning) and "answer" (the final date in MM/DD/YYYY format). '
    "Example:\n" + _md_json(
        thinking="Jan 15 + 10 days = Jan 25, 2020.",
        answer="01/25/2020",
    )
)


def _user_date_en(s: Sample) -> str:
    return s["question"]


def _shot_assistant_date_en(s: Sample) -> str:
    return s["gold"]


def _shot_assistant_date_en_thinking(s: Sample) -> str:
    return _md_json(thinking="Computed step by step.", answer=s["gold"])


DATE_EN = PromptTemplate(
    system=_SYS_DATE_EN,
    system_thinking=_SYS_DATE_EN_THINK,
    render_user=_user_date_en,
    render_shot_user=_user_date_en,
    render_shot_assistant=_shot_assistant_date_en,
    render_shot_assistant_thinking=_shot_assistant_date_en_thinking,
)


# ------------------------- DateArith VI (VLSP) -------------------------
_SYS_DATE_VI = (
    "Bạn là bộ giải bài toán tính toán thời gian chính xác. "
    "Trước tiên, hãy tự tính toán từng bước một cách ẩn. "
    "KHÔNG được hiển thị quá trình suy nghĩ. "
    "Sau đó chỉ trả lời duy nhất theo định dạng: 'Tháng M, YYYY'. "
    "Không giải thích, không thêm bất kỳ ký tự nào khác."
)

_SYS_DATE_VI_THINK = (
    "Bạn là bộ giải bài toán tính toán thời gian chính xác. "
    "Trả lời bằng một JSON object trong markdown code block có đúng hai key: "
    '"thinking" (quá trình tính toán từng bước) và "answer" (kết quả theo định dạng "Tháng M, YYYY"). '
    "Ví dụ:\n" + _md_json(
        thinking="Tháng 1, 1800 + 5 năm = Tháng 1, 1805.",
        answer="Tháng 1, 1805",
    )
)


def _user_date_vi(s: Sample) -> str:
    ctx = s.get("context") or ""
    if ctx:
        return f"Ngữ cảnh: {ctx}\nCâu hỏi: {s['question']}"
    return s["question"]


def _shot_assistant_date_vi(s: Sample) -> str:
    return s["gold"]


def _shot_assistant_date_vi_thinking(s: Sample) -> str:
    return _md_json(thinking="Tính từng bước.", answer=s["gold"])


DATE_VI = PromptTemplate(
    system=_SYS_DATE_VI,
    system_thinking=_SYS_DATE_VI_THINK,
    render_user=_user_date_vi,
    render_shot_user=_user_date_vi,
    render_shot_assistant=_shot_assistant_date_vi,
    render_shot_assistant_thinking=_shot_assistant_date_vi_thinking,
)


# ------------------------- Registry -------------------------

TEMPLATES: dict[tuple[str, str], PromptTemplate] = {
    ("duration", "en"): DURATION_EN,
    ("duration", "vi"): DURATION_VI,
    ("date_arith", "en"): DATE_EN,
    ("date_arith", "vi"): DATE_VI,
}


def get_template(task: str, language: str) -> PromptTemplate:
    key = (task, language)
    if key not in TEMPLATES:
        raise KeyError(f"No prompt template for {key}")
    return TEMPLATES[key]


def build_messages(
    sample: Sample,
    shots: Sequence[Sample] = (),
    enable_thinking: bool = False,
) -> list[ChatMessage]:
    tmpl = get_template(sample["task"], sample["language"])
    system = tmpl.system_thinking if enable_thinking and tmpl.system_thinking else tmpl.system
    msgs: list[ChatMessage] = [ChatMessage(role="system", content=system)]
    shot_assistant_render = (
        tmpl.render_shot_assistant_thinking
        if enable_thinking and tmpl.render_shot_assistant_thinking
        else tmpl.render_shot_assistant
    )
    for shot in shots:
        msgs.append(ChatMessage(role="user", content=tmpl.render_shot_user(shot)))
        msgs.append(ChatMessage(role="assistant", content=shot_assistant_render(shot)))
    msgs.append(ChatMessage(role="user", content=tmpl.render_user(sample)))
    return msgs
