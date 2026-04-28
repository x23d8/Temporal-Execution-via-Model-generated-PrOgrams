"""FreeThink method — no system prompt, model reasons freely, answer extracted by regex + code.

Pipeline
--------
1. Build a user-only message (no system role) with the raw question/context.
2. Call the model with enable_thinking=True and a large token budget.
3. Split output at </think> to get the clean answer section.
4. Run a regex waterfall on the clean section; fall back to the full text.
5. Manual-parse / validate every match in Python before accepting it.
"""

from __future__ import annotations

import re
from datetime import date as _date

from ..data.schema import Sample
from ..models.base import ChatLM, ChatMessage

# ── token budget (thinking needs much more room than constrained prompts) ──────
_TOKENS_DATE   = 2048
_TOKENS_DUR    = 1024

# ── split thinking block from answer ─────────────────────────────────────────
_THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)

# ── answer-marker phrases (EN + VI) ──────────────────────────────────────────
_MARKER_RE = re.compile(
    r"(?:"
    r"(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)"
    r"|result\s*(?:is|:)"
    r"|conclusion\s*(?:is|:)"
    r"|the\s+date\s+is"
    r"|therefore[,\s]+"
    r"|so[,\s]+(?:the\s+)?(?:date|answer)\s*(?:is|:)?"
    r"|câu\s+trả\s+lời\s*(?:là|:)"
    r"|kết\s+quả\s*(?:là|:)"
    r"|vậy[,\s]+"
    r"|do\s+đó[,\s]+"
    r")\s*",
    re.IGNORECASE,
)

# ── MM/DD/YYYY ────────────────────────────────────────────────────────────────
_MMDDYYYY_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")

# ── ISO date YYYY-MM-DD ───────────────────────────────────────────────────────
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")

# ── Written EN date: "June 22, 2023" or "22 June 2023" ───────────────────────
_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MON = "(?:" + "|".join(_MONTH_MAP) + ")"
_WRITTEN_MDY_RE = re.compile(  # June 22, 2023
    rf"\b({_MON})\s+(\d{{1,2}})(?:st|nd|rd|th)?\s*,?\s*(\d{{4}})\b",
    re.IGNORECASE,
)
_WRITTEN_DMY_RE = re.compile(  # 22 June 2023
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MON})\s*,?\s*(\d{{4}})\b",
    re.IGNORECASE,
)

# ── Vietnamese date: "Tháng M, YYYY" or "M/YYYY" ─────────────────────────────
_VI_MONTH_RE   = re.compile(r"[Tt]h[áa]ng\s*(\d{1,2})\s*[,/\-\s]+\s*(\d{3,4})")
_VI_SLASH_RE   = re.compile(r"\b(\d{1,2})\s*/\s*(\d{4})\b")

# ── Yes / No (EN + VI) ────────────────────────────────────────────────────────
_YES_RE = re.compile(r"\byes\b|\bcó\b|\bđúng\b", re.IGNORECASE)
_NO_RE  = re.compile(r"\bno\b|\bkhông\b|\bsai\b", re.IGNORECASE)

# ── Markdown bold stripper ────────────────────────────────────────────────────
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")


# ── helpers ───────────────────────────────────────────────────────────────────

def _valid_date(y: int, m: int, d: int) -> bool:
    try:
        _date(y, m, d)
        return True
    except ValueError:
        return False


def _fmt(m: int, d: int, y: int) -> str:
    return f"{m:02d}/{d:02d}/{y:04d}"


def _split_thinking(raw: str) -> tuple[str, str]:
    """Return (thinking_text, answer_text). If no </think>, thinking is empty."""
    m = _THINK_CLOSE_RE.search(raw)
    if m:
        return raw[: m.end()], raw[m.end():].strip()
    return "", raw.strip()


def _unbold(text: str) -> str:
    """Replace **foo** with foo so regex doesn't miss dates inside bold."""
    return _BOLD_RE.sub(r"\1", text)


# ── per-task regex waterfall ──────────────────────────────────────────────────

def _extract_date_en(text: str) -> str | None:
    """Try patterns in priority order; validate each match in Python."""
    text = _unbold(text)

    # 1. After an explicit answer marker
    for m_marker in _MARKER_RE.finditer(text):
        tail = text[m_marker.end(): m_marker.end() + 30]
        for pattern, handler in _DATE_EN_PATTERNS:
            m = pattern.match(tail.lstrip())
            if m:
                result = handler(m)
                if result:
                    return result

    # 2. Scan the whole text in order of pattern priority
    for pattern, handler in _DATE_EN_PATTERNS:
        for m in pattern.finditer(text):
            result = handler(m)
            if result:
                return result

    return None


def _handle_mmddyyyy(m: re.Match) -> str | None:
    mo, d, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return _fmt(mo, d, y) if _valid_date(y, mo, d) else None


def _handle_iso(m: re.Match) -> str | None:
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return _fmt(mo, d, y) if _valid_date(y, mo, d) else None


def _handle_mdy(m: re.Match) -> str | None:
    mo = _MONTH_MAP.get(m.group(1).lower())
    d, y = int(m.group(2)), int(m.group(3))
    if mo and _valid_date(y, mo, d):
        return _fmt(mo, d, y)
    return None


def _handle_dmy(m: re.Match) -> str | None:
    d = int(m.group(1))
    mo = _MONTH_MAP.get(m.group(2).lower())
    y = int(m.group(3))
    if mo and _valid_date(y, mo, d):
        return _fmt(mo, d, y)
    return None


# Ordered list: (regex, handler)
_DATE_EN_PATTERNS = [
    (_MMDDYYYY_RE,   _handle_mmddyyyy),
    (_ISO_DATE_RE,   _handle_iso),
    (_WRITTEN_MDY_RE, _handle_mdy),
    (_WRITTEN_DMY_RE, _handle_dmy),
]


def _extract_date_vi(text: str) -> str | None:
    text = _unbold(text)

    # 1. After marker
    for m_marker in _MARKER_RE.finditer(text):
        tail = text[m_marker.end(): m_marker.end() + 25]
        for pattern, handler in _DATE_VI_PATTERNS:
            mm = pattern.match(tail.lstrip())
            if mm:
                result = handler(mm)
                if result:
                    return result

    # 2. Full scan
    for pattern, handler in _DATE_VI_PATTERNS:
        for mm in pattern.finditer(text):
            result = handler(mm)
            if result:
                return result

    return None


def _handle_vi_month(m: re.Match) -> str | None:
    mo, y = int(m.group(1)), int(m.group(2))
    return f"Tháng {mo}, {y}" if 1 <= mo <= 12 and y > 100 else None


def _handle_vi_slash(m: re.Match) -> str | None:
    mo, y = int(m.group(1)), int(m.group(2))
    return f"Tháng {mo}, {y}" if 1 <= mo <= 12 else None


_DATE_VI_PATTERNS = [
    (_VI_MONTH_RE, _handle_vi_month),
    (_VI_SLASH_RE, _handle_vi_slash),
]


def _extract_yes_no(text: str) -> str | None:
    text = _unbold(text)

    # 1. After explicit answer marker
    for m_marker in _MARKER_RE.finditer(text):
        tail = text[m_marker.end(): m_marker.end() + 20].strip()
        y = _YES_RE.match(tail)
        n = _NO_RE.match(tail)
        if y and not n:
            return "yes"
        if n and not y:
            return "no"

    # 2. Last non-empty line
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        y = _YES_RE.search(line)
        n = _NO_RE.search(line)
        if y and not n:
            return "yes"
        if n and not y:
            return "no"
        break  # only check the very last line

    # 3. First occurrence anywhere
    y = _YES_RE.search(text)
    n = _NO_RE.search(text)
    if y and not n:
        return "yes"
    if n and not y:
        return "no"
    if y and n:
        return "yes" if y.start() < n.start() else "no"

    return None


_EXTRACTORS = {
    ("date_arith", "en"): _extract_date_en,
    ("date_arith", "vi"): _extract_date_vi,
    ("duration",   "en"): _extract_yes_no,
    ("duration",   "vi"): _extract_yes_no,
}

_MAX_TOKENS = {
    "date_arith": _TOKENS_DATE,
    "duration":   _TOKENS_DUR,
}


# ── user message builders (no system prompt) ──────────────────────────────────

def _user_msg_date_en(s: Sample) -> str:
    ctx = (s.get("context") or "").strip()
    return f"Context: {ctx}\n{s['question']}" if ctx else s["question"]


def _user_msg_date_vi(s: Sample) -> str:
    ctx = (s.get("context") or "").strip()
    return f"Ngữ cảnh: {ctx}\n{s['question']}" if ctx else s["question"]


def _user_msg_duration_en(s: Sample) -> str:
    cand = (s.get("meta") or {}).get("candidate_answer", "")
    return (
        f"Context: {s.get('context', '')}\n"
        f"Question: {s['question']}\n"
        f"Candidate duration: {cand}"
    )


def _user_msg_duration_vi(s: Sample) -> str:
    cand = (s.get("meta") or {}).get("candidate_answer", "")
    return (
        f"Ngữ cảnh: {s.get('context', '')}\n"
        f"Câu hỏi: {s['question']}\n"
        f"Khoảng thời gian đề xuất: {cand}"
    )


_USER_BUILDERS = {
    ("date_arith", "en"): _user_msg_date_en,
    ("date_arith", "vi"): _user_msg_date_vi,
    ("duration",   "en"): _user_msg_duration_en,
    ("duration",   "vi"): _user_msg_duration_vi,
}


# ── method class ──────────────────────────────────────────────────────────────

class FreeThinkMethod:
    name = "free_think"

    def __init__(self, model: ChatLM, enable_thinking: bool = True) -> None:
        self.model = model
        # Thinking is on by default for this method — that's its whole point.
        self.enable_thinking = enable_thinking

    def predict(self, sample: Sample) -> str:
        task = sample["task"]
        lang = sample.get("language", "en")
        key  = (task, lang)

        build_user = _USER_BUILDERS.get(key)
        if build_user is None:
            raise KeyError(f"FreeThinkMethod: no user builder for {key}")

        msgs = [ChatMessage(role="user", content=build_user(sample))]

        raw = self.model.generate(
            msgs,
            max_new_tokens=_MAX_TOKENS.get(task, 1024),
            temperature=0.0,
            do_sample=False,
            enable_thinking=self.enable_thinking,
        )

        return raw

    def extract_answer(self, task: str, language: str, raw: str) -> str | None:
        """Called by the runner instead of the default extractor."""
        key = (task, language)
        extractor = _EXTRACTORS.get(key)
        if extractor is None:
            raise KeyError(f"FreeThinkMethod: no extractor for ({task!r}, {language!r})")

        _, answer_text = _split_thinking(raw)

        # Try the clean answer section first, then fall back to the full text.
        result = extractor(answer_text) if answer_text else None
        if result is None:
            result = extractor(raw)
        return result
