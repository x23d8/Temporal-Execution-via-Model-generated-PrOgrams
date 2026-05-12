"""Schema thống nhất cho samples.

Mỗi sample sau khi loader trả về là 1 dict với các key chuẩn:
- sample_id: str
- task: "duration" | "date_arith"
- language: "en" | "vi"
- dataset: tên dataset nguồn
- context: str (có thể rỗng)
- question: str
- gold: str — đáp án chuẩn đã normalize:
    + duration: "yes" | "no"
    + date_arith (EN): "MM/DD/YYYY"
    + date_arith (VI): "Tháng M, YYYY"
- meta: dict tuỳ dataset (candidate_answer cho duration binary, v.v.)
"""

from __future__ import annotations

from typing import Literal, TypedDict


class Sample(TypedDict, total=False):
    sample_id: str
    task: Literal["duration", "date_arith"]
    language: Literal["en", "vi"]
    dataset: str
    context: str
    question: str
    gold: str
    meta: dict


class McqSample(TypedDict, total=False):
    """One arithmetic MCQ item with four labelled options.

    gold is always a single uppercase letter: 'A' | 'B' | 'C' | 'D'.
    """
    sample_id: str
    category: str   # e.g. 'Hour Adjustment (24h)', 'Year Shift', …
    dataset: str
    question: str
    option_a: str
    option_b: str
    option_c: str
    option_d: str
    gold: str       # 'A' | 'B' | 'C' | 'D'
