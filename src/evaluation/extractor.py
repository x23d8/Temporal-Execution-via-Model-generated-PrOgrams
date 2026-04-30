"""Trích & chuẩn hoá output của LLM để so với gold.

- yes_no: duration reasoning (EN + VI).
- bigbench_date: chuỗi MM/DD/YYYY.
- vlsp_date: "Tháng M, YYYY" (VI date arithmetic).

Extraction priority:
  1. JSON {"thinking": ..., "answer": ...} — used when enable_thinking=True.
  2. Regex fallback — for plain text output (non-thinking mode).
"""

from __future__ import annotations

import json as _json
import re

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

_YES_PAT = re.compile(r"\byes\b|\bcó\b|\bđúng\b", re.IGNORECASE)
_NO_PAT = re.compile(r"\bno\b|\bkhông\b|\bsai\b", re.IGNORECASE)

_DATE_MMDDYYYY_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b")

_VI_MONTH_RE = re.compile(
    r"[Tt]h[áa]ng\s*(\d{1,2})\s*[,/\-\s]+\s*(\d{3,4})"
)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks that Qwen3.5-9B may emit."""
    return _THINK_RE.sub("", text).strip()


_JSON_ANSWER_RE = re.compile(r'"answer"\s*:\s*"([^"]*)"')


def _extract_json_answer(text: str) -> str | None:
    """Return the 'answer' field from a JSON object in the text, or None.

    Tries strict JSON parse first, then a bare-regex scan so partial/truncated
    JSON (model ran out of tokens mid-thinking) still yields the answer.
    Works reliably when the format puts "answer" before "thinking" so the
    answer field is emitted before any long reasoning that might get cut off.
    """
    # Strict parse — full text
    try:
        data = _json.loads(text.strip())
        if isinstance(data, dict) and "answer" in data:
            return str(data["answer"]).strip()
    except _json.JSONDecodeError:
        pass
    # Strict parse — first {...} block (model may add prose around JSON)
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            data = _json.loads(m.group())
            if isinstance(data, dict) and "answer" in data:
                return str(data["answer"]).strip()
        except _json.JSONDecodeError:
            pass
    # Regex fallback — handles truncated / malformed JSON
    m = _JSON_ANSWER_RE.search(text)
    if m:
        return m.group(1).strip()
    return None


def extract_yes_no(raw: str) -> str | None:
    """Return 'yes' or 'no' (positive class = yes). None if undecidable."""
    if raw is None:
        return None
    text = strip_thinking(raw)

    # JSON path
    json_ans = _extract_json_answer(text)
    if json_ans is not None:
        ans = json_ans.strip().lower()
        if ans in ("yes", "no"):
            return ans
        if _YES_PAT.search(ans) and not _NO_PAT.search(ans):
            return "yes"
        if _NO_PAT.search(ans) and not _YES_PAT.search(ans):
            return "no"

    # Regex fallback
    lowered = text.strip().lower()
    if not lowered:
        return None
    head = lowered[:40]
    y = _YES_PAT.search(head)
    n = _NO_PAT.search(head)
    if y and not n:
        return "yes"
    if n and not y:
        return "no"
    y_any = _YES_PAT.search(lowered)
    n_any = _NO_PAT.search(lowered)
    if y_any and not n_any:
        return "yes"
    if n_any and not y_any:
        return "no"
    if y_any and n_any:
        return "yes" if y_any.start() < n_any.start() else "no"
    return None


def extract_mmddyyyy(raw: str) -> str | None:
    """Extract MM/DD/YYYY: JSON 'answer' field first, then last regex match."""
    if raw is None:
        return None
    text = strip_thinking(raw)

    # JSON path
    json_ans = _extract_json_answer(text)
    if json_ans is not None:
        m = _DATE_MMDDYYYY_RE.search(json_ans)
        if m:
            mm, dd, yyyy = m.group(1), m.group(2), m.group(3)
            if len(yyyy) != 2:
                return f"{int(mm):02d}/{int(dd):02d}/{int(yyyy):04d}"

    # Regex fallback — take last match so CoT reasoning doesn't shadow the answer
    matches = _DATE_MMDDYYYY_RE.findall(text)
    if not matches:
        return None
    mm, dd, yyyy = matches[-1]
    if len(yyyy) == 2:
        return None
    return f"{int(mm):02d}/{int(dd):02d}/{int(yyyy):04d}"


def normalize_mmddyyyy(s: str) -> str | None:
    """Normalize gold like '5/1/2021' or '05/01/2021' to '05/01/2021'."""
    if s is None:
        return None
    m = _DATE_MMDDYYYY_RE.search(s.strip())
    if not m:
        return None
    mm, dd, yyyy = m.group(1), m.group(2), m.group(3)
    if len(yyyy) != 4:
        return None
    return f"{int(mm):02d}/{int(dd):02d}/{int(yyyy):04d}"


def extract_vi_month_year(raw: str) -> str | None:
    """Extract 'Tháng M, YYYY': JSON 'answer' field first, then last regex match."""
    if raw is None:
        return None
    text = strip_thinking(raw)

    # JSON path
    json_ans = _extract_json_answer(text)
    if json_ans is not None:
        m = _VI_MONTH_RE.search(json_ans)
        if m:
            month, year = int(m.group(1)), int(m.group(2))
            if 1 <= month <= 12:
                return f"Tháng {month}, {year}"

    # Regex fallback — take last match
    matches = list(_VI_MONTH_RE.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    month, year = int(m.group(1)), int(m.group(2))
    if not (1 <= month <= 12):
        return None
    return f"Tháng {month}, {year}"


def normalize_vi_month_year(s: str) -> str | None:
    """Normalize gold 'Tháng 4, 1321' variants to canonical form."""
    if s is None:
        return None
    m = _VI_MONTH_RE.search(s.strip())
    if not m:
        return None
    month, year = int(m.group(1)), int(m.group(2))
    return f"Tháng {month}, {year}"


TASK_EXTRACTORS = {
    ("duration", "en"): extract_yes_no,
    ("duration", "vi"): extract_yes_no,
    ("date_arith", "en"): extract_mmddyyyy,
    ("date_arith", "vi"): extract_vi_month_year,
}

TASK_GOLD_NORMALIZERS = {
    ("duration", "en"): lambda s: s.strip().lower() if s else None,
    ("duration", "vi"): lambda s: s.strip().lower() if s else None,
    ("date_arith", "en"): normalize_mmddyyyy,
    ("date_arith", "vi"): normalize_vi_month_year,
}


def extract(task: str, language: str, raw: str) -> str | None:
    fn = TASK_EXTRACTORS.get((task, language))
    if fn is None:
        raise KeyError(f"No extractor for ({task!r}, {language!r})")
    return fn(raw)


def normalize_gold(task: str, language: str, gold: str) -> str | None:
    fn = TASK_GOLD_NORMALIZERS.get((task, language))
    if fn is None:
        raise KeyError(f"No gold normalizer for ({task!r}, {language!r})")
    return fn(gold)
