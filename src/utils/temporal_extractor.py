"""Deterministic temporal extractor + normalizer for date_arith and duration tasks.

Pipeline:
  1. Extract  — regex pull anchor dates, offsets, candidate durations
  2. Normalize — resolve relative → absolute date / compare duration range
  3. Format   — output in task-correct string format

Returns None at any step when ambiguous; caller falls back to LLM.

Supported patterns:
  EN date_arith: MM/DD/YYYY, Month DD YYYY, DD Month YYYY, ISO + relative
                 (yesterday/today/tomorrow/N days|weeks|months ago|from/last|next weekday)
  VI date_arith: tháng M, YYYY + N năm Y tháng sau/trước  (relativedelta arithmetic)
  EN/VI duration: exact ("36 minutes") + vague ("a few days", "vài tuần") → (min,max) range
                  activity-keyword lookup → plausibility range → overlap check
"""

from __future__ import annotations

import re
import datetime
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════════════
# Month / weekday tables
# ═══════════════════════════════════════════════════════════════════════════════

_MONTH_EN: dict[str, int] = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9,
    "oct": 10, "nov": 11, "dec": 12,
}
_MNAMES = "|".join(sorted(_MONTH_EN, key=len, reverse=True))

_WEEKDAY_EN: dict[str, int] = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
    "mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6,
}

# ═══════════════════════════════════════════════════════════════════════════════
# Absolute date extraction — English
# ═══════════════════════════════════════════════════════════════════════════════

_P_MNAME_DY = re.compile(rf"\b({_MNAMES})\.?\s+(\d{{1,2}}),?\s+(\d{{4}})\b", re.I)
_P_D_MNAME_Y = re.compile(rf"\b(\d{{1,2}})\s+({_MNAMES})\.?\s+(\d{{4}})\b", re.I)
_P_MDY = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
_P_YMD = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")


def _safe_date(y: int, m: int, d: int) -> datetime.date | None:
    try:
        return datetime.date(y, m, d)
    except ValueError:
        return None


def _extract_dates_en(text: str, uk_fmt: bool = False) -> list[datetime.date]:
    found: list[datetime.date] = []
    for m in _P_MNAME_DY.finditer(text):
        mo = _MONTH_EN.get(m.group(1).lower())
        if mo:
            d = _safe_date(int(m.group(3)), mo, int(m.group(2)))
            if d:
                found.append(d)
    for m in _P_D_MNAME_Y.finditer(text):
        mo = _MONTH_EN.get(m.group(2).lower())
        if mo:
            d = _safe_date(int(m.group(3)), mo, int(m.group(1)))
            if d:
                found.append(d)
    for m in _P_MDY.finditer(text):
        if uk_fmt:
            # DD/MM/YYYY → swap day and month
            d = _safe_date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        else:
            d = _safe_date(int(m.group(3)), int(m.group(1)), int(m.group(2)))
        if d:
            found.append(d)
    for m in _P_YMD.finditer(text):
        d = _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if d:
            found.append(d)
    # Deduplicate preserving order
    seen: set[datetime.date] = set()
    return [x for x in found if not (x in seen or seen.add(x))]  # type: ignore[func-returns-value]


# ═══════════════════════════════════════════════════════════════════════════════
# Anchor label → "today" offset  (English)
# ═══════════════════════════════════════════════════════════════════════════════
# Each entry: (pattern, delta_to_today)
# delta_to_today = how many days to ADD to the stated date to get "today"
# e.g. "yesterday was [DATE]" → today = DATE + 1

_ANCHOR_LABELS_EN: list[tuple[re.Pattern, int | None]] = [
    # "today is in fact [DATE]" — explicit correction override, must be first
    (re.compile(r"\btoday\s+is\s+in\s+fact\b", re.I), 0),
    (re.compile(r"\bthe\s+day\s+before\s+yesterday\s+(?:was|is)\b", re.I), +2),
    (re.compile(r"\byesterday\s+(?:was|is)\b", re.I), +1),
    (re.compile(r"\btoday\s+(?:was|is|it\s+is)\b", re.I), 0),
    (re.compile(r"\bit\s+is\s+today\b", re.I), 0),
    (re.compile(r"\btoday\b.*\bis\b", re.I), 0),   # "Today, [date], is ..."
    (re.compile(r"\btomorrow\s+(?:will\s+be|is)\b", re.I), -1),
    # "booked a flight for tomorrow, March 5, 2021" → today = March 4
    (re.compile(r"\bfor\s+tomorrow\b", re.I), -1),
    (re.compile(r"\bthe\s+day\s+after\s+tomorrow\s+(?:will\s+be|is)\b", re.I), -2),
    # "N days ago was [DATE]"  or  "in N days it will be [DATE]"
    (re.compile(r"\b(\d+)\s+days?\s+ago\s+(?:was|is)\b", re.I), None),
    (re.compile(r"\bin\s+(\d+)\s+days?\s+(?:it\s+(?:will\s+be|is)|will\s+be|is)\b", re.I), None),
]


def _find_today_en(text: str, dates: list[datetime.date]) -> datetime.date | None:
    # "2015 is coming in 36 hours" → today = Jan 1, 2015 - ceil(36/24) days
    m = re.search(r"\b(\d{4})\s+is\s+coming\s+in\s+(\d+)\s+hours?\b", text, re.I)
    if m:
        try:
            jan1 = datetime.date(int(m.group(1)), 1, 1)
            days_before = (int(m.group(2)) + 23) // 24
            return jan1 - datetime.timedelta(days=days_before)
        except ValueError:
            pass

    if not dates:
        return None
    for pat, offset in _ANCHOR_LABELS_EN:
        m = pat.search(text)
        if m is None:
            continue
        if offset is None:
            try:
                n = int(m.group(1))
                offset = +n if "ago" in pat.pattern else -n
            except (IndexError, ValueError):
                continue
        candidate = _first_date_after(text, m.end(), dates)
        if candidate:
            return candidate + datetime.timedelta(days=offset)
    # Fallback: single date → assume it IS today
    if len(dates) == 1:
        return dates[0]
    return None


def _first_date_after(
    text: str, pos: int, dates: list[datetime.date]
) -> datetime.date | None:
    """Return the date whose textual match begins closest to or after `pos`."""
    candidates: list[tuple[int, datetime.date]] = []
    for pat, parser in [
        (_P_MNAME_DY, _parse_mname_dy),
        (_P_D_MNAME_Y, _parse_d_mname_y),
        (_P_MDY, _parse_mdy),
        (_P_YMD, _parse_ymd),
    ]:
        for m in pat.finditer(text):
            if m.start() >= pos:
                d = parser(m)
                if d and d in dates:
                    candidates.append((m.start(), d))
    return min(candidates, key=lambda x: x[0])[1] if candidates else None


def _parse_mname_dy(m: re.Match) -> datetime.date | None:
    mo = _MONTH_EN.get(m.group(1).lower())
    return _safe_date(int(m.group(3)), mo, int(m.group(2))) if mo else None


def _parse_d_mname_y(m: re.Match) -> datetime.date | None:
    mo = _MONTH_EN.get(m.group(2).lower())
    return _safe_date(int(m.group(3)), mo, int(m.group(1))) if mo else None


def _parse_mdy(m: re.Match) -> datetime.date | None:
    return _safe_date(int(m.group(3)), int(m.group(1)), int(m.group(2)))


def _parse_ymd(m: re.Match) -> datetime.date | None:
    return _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))


# ═══════════════════════════════════════════════════════════════════════════════
# Target expression → delta from "today"  (English)
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_target_phrase_en(question: str) -> str | None:
    """Pull the temporal phrase from 'what is the date [PHRASE] in MM/DD/YYYY'."""
    m = re.search(
        r"what\s+is\s+the\s+date\s+(.+?)(?:\s+in\s+mm/dd/yyyy|\?|$)",
        question, re.I
    )
    return m.group(1).strip() if m else None


def _phrase_to_delta(phrase: str) -> int | None:
    p = phrase.lower().strip()

    _SIMPLE: dict[str, int] = {
        "today": 0, "now": 0,
        "yesterday": -1,
        "tomorrow": 1,
        "the day before yesterday": -2,
        "the day after tomorrow": 2,
        "two days ago": -2, "2 days ago": -2,
        "three days ago": -3, "3 days ago": -3,
    }
    if p in _SIMPLE:
        return _SIMPLE[p]

    # "N days ago"
    m = re.match(r"(\d+)\s+days?\s+ago$", p)
    if m:
        return -int(m.group(1))

    # "N days from now/today/then" / "N days later" / "N days after today"
    m = re.match(r"(\d+)\s+days?\s+(?:from\s+(?:now|today|then)|later|after\s+(?:today|now))", p)
    if m:
        return int(m.group(1))

    # "N days before [something]" — the [something] gives a new anchor; punt to LLM
    # unless [something] is a synonym for "today"
    m = re.match(r"(\d+)\s+days?\s+before\s*(today|now)?", p)
    if m:
        return -int(m.group(1)) if m.group(2) else None

    # "N days after [something]"
    m = re.match(r"(\d+)\s+days?\s+after\s*(today|now)?", p)
    if m:
        return int(m.group(1)) if m.group(2) else None

    # "a/one/N week(s) from now/today / ago / later"
    m = re.match(r"(?:a|an|one|(\d+))\s+weeks?\s+(ago|from\s+(?:now|today)|later)", p)
    if m:
        n = int(m.group(1)) if m.group(1) else 1
        return -n * 7 if m.group(2) == "ago" else n * 7

    # "N hours later / from now / after today"
    m = re.match(r"(\d+)\s+hours?\s+(?:later|from\s+(?:now|today)|after\s+(?:today|now))", p)
    if m:
        return round(int(m.group(1)) / 24)

    # "N hours ago"
    m = re.match(r"(\d+)\s+hours?\s+ago", p)
    if m:
        return -round(int(m.group(1)) / 24)

    # "a month ago" — let relativedelta handle; return None to bubble up
    return None  # needs relativedelta or is ambiguous


def _parse_target_weekday_en(
    question: str, today: datetime.date
) -> datetime.date | None:
    m = re.search(
        r"(?:what\s+is\s+the\s+date\s+.*)?(last|next)\s+"
        r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
        question, re.I
    )
    if not m:
        return None
    direction = m.group(1).lower()
    target_wd = _WEEKDAY_EN[m.group(2).lower()]
    current_wd = today.weekday()
    if direction == "next":
        delta = (target_wd - current_wd) % 7 or 7
        return today + datetime.timedelta(days=delta)
    delta = (current_wd - target_wd) % 7 or 7
    return today - datetime.timedelta(days=delta)


def _try_relativedelta_en(
    question: str, today: datetime.date
) -> datetime.date | None:
    """Handle month/year offsets using dateutil.relativedelta."""
    phrase = _extract_target_phrase_en(question)
    if not phrase:
        return None
    p = phrase.lower()
    m = re.search(
        r"(?:a|an|one|(\d+))\s+(months?|years?)\s+"
        r"(ago|from\s+(?:now|today)|later)",
        p
    )
    if not m:
        return None
    try:
        from dateutil.relativedelta import relativedelta
    except ImportError:
        return None
    n = int(m.group(1)) if m.group(1) else 1
    unit = m.group(2).rstrip("s")
    sign = -1 if m.group(3) == "ago" else 1
    return today + relativedelta(**{unit + "s": sign * n})


# ═══════════════════════════════════════════════════════════════════════════════
# Vietnamese date_arith — tháng M, YYYY + N năm Y tháng sau/trước
# ═══════════════════════════════════════════════════════════════════════════════

# Anchor: "tháng M, YYYY"  (month-level only — no day in VLSP)
_P_VI_MONTH_YEAR = re.compile(r"\btháng\s+(\d{1,2}),?\s+(\d{4})\b", re.I)

# Offset: "X năm [và] Y tháng" | "X năm" | "Y tháng"
_P_VI_YM = re.compile(r"(\d+)\s+năm\s+(?:và\s+)?(\d+)\s+tháng")
_P_VI_Y  = re.compile(r"(\d+)\s+năm")
_P_VI_M  = re.compile(r"(\d+)\s+tháng")

# Exclude "tháng M, YYYY" matches from month-only extraction
_P_VI_ANCHOR_CTX = re.compile(r"\btháng\s+\d{1,2},?\s+\d{4}")


def _solve_vi_date(sample: dict) -> str | None:
    question = sample.get("question", "")
    q = question  # keep original for regex; lowercase only for direction

    # 1. Extract anchor: tháng M, YYYY
    anchors = _P_VI_MONTH_YEAR.findall(q)
    if not anchors:
        return None
    month_str, year_str = anchors[0]
    anchor_month, anchor_year = int(month_str), int(year_str)

    # 2. Extract offset — try compound form first
    m_ym = _P_VI_YM.search(q)
    if m_ym: 
        years_off = int(m_ym.group(1))
        months_off = int(m_ym.group(2))
    else:
        years_off = int(m_y.group(1)) if (m_y := _P_VI_Y.search(q)) else 0
        # For month-only, don't match the "tháng M" in anchor
        months_off = 0
        q_stripped = _P_VI_ANCHOR_CTX.sub("", q)  # remove "tháng M, YYYY"
        m_m = _P_VI_M.search(q_stripped)
        if m_m:
            months_off = int(m_m.group(1))

    if years_off == 0 and months_off == 0:
        return None

    # 3. Direction: "sau" → forward, "trước" → backward
    ql = q.lower()
    has_sau   = bool(re.search(r"\bsau\b", ql))
    has_truoc = bool(re.search(r"\btrước\b", ql))
    if has_sau == has_truoc:
        return None  # ambiguous or neither
    sign = 1 if has_sau else -1

    # 4. Compute with relativedelta
    try:
        from dateutil.relativedelta import relativedelta
    except ImportError:
        return None

    anchor = datetime.date(anchor_year, anchor_month, 1)
    result = anchor + relativedelta(years=sign * years_off, months=sign * months_off)
    return f"Tháng {result.month}, {result.year}"


# ═══════════════════════════════════════════════════════════════════════════════
# Duration range parsing  (exact + vague)
# ═══════════════════════════════════════════════════════════════════════════════

# (min_seconds, max_seconds) for vague English quantifiers
_VAGUE_EN: list[tuple[re.Pattern, tuple[float, float]]] = [
    # "a/several/a few/some UNIT"  and  "for UNIT"
    (re.compile(r"\b(?:a\s+few|several|some|for\s+a?\s*few)\s+seconds?\b", re.I), (2, 30)),
    (re.compile(r"\b(?:a\s+few|several|some)\s+minutes?\b", re.I), (120, 900)),
    (re.compile(r"\bfor\s+(?:a\s+few\s+)?minutes?\b", re.I), (60, 1800)),
    (re.compile(r"\b(?:a\s+few|several|some)\s+hours?\b", re.I), (3600, 36000)),
    (re.compile(r"\bfor\s+(?:a\s+few\s+)?hours?\b", re.I), (3600, 86400)),
    (re.compile(r"\bfor\s+(?:a\s+few\s+)?days?\b", re.I), (86400, 604800)),
    (re.compile(r"\b(?:a\s+few|several|some)\s+days?\b", re.I), (86400, 604800)),
    (re.compile(r"\bfor\s+(?:a\s+few\s+)?weeks?\b", re.I), (604800, 2592000)),
    (re.compile(r"\b(?:a\s+few|several|some)\s+weeks?\b", re.I), (604800, 2592000)),
    (re.compile(r"\bfor\s+(?:a\s+few\s+)?months?\b", re.I), (2592000, 15552000)),
    (re.compile(r"\b(?:a\s+few|several|some)\s+months?\b", re.I), (2592000, 15552000)),
    (re.compile(r"\b(?:a\s+few|several|some)\s+years?\b", re.I), (31536000, 157680000)),
    # "a [unit]" without quantifier adjective
    (re.compile(r"\ba\s+minute\b", re.I), (30, 180)),
    (re.compile(r"\ba\s+(?:short\s+)?while\b", re.I), (60, 3600)),
    (re.compile(r"\ba\s+moment\b", re.I), (5, 300)),
    (re.compile(r"\bmoments?\b", re.I), (1, 60)),
]

# Vague Vietnamese quantifiers
_VAGUE_VI: list[tuple[re.Pattern, tuple[float, float]]] = [
    (re.compile(r"\b(?:vài|một\s+vài)\s+giây\b", re.I), (2, 30)),
    (re.compile(r"\b(?:vài|một\s+vài)\s+phút\b", re.I), (120, 900)),
    (re.compile(r"\bvài\s+(?:tiếng|giờ)\b|một\s+vài\s+(?:tiếng|giờ)\b", re.I), (3600, 36000)),
    (re.compile(r"\btrong\s+nhiều\s+giờ\b", re.I), (3600, 86400)),
    (re.compile(r"\b(?:vài|một\s+vài)\s+ngày\b", re.I), (86400, 604800)),
    (re.compile(r"\b(?:vài|một\s+vài)\s+tuần\b", re.I), (604800, 2592000)),
    (re.compile(r"\b(?:vài|một\s+vài)\s+tháng\b", re.I), (2592000, 15552000)),
    (re.compile(r"\b(?:vài|một\s+vài)\s+năm\b", re.I), (31536000, 157680000)),
    (re.compile(r"\bmột\s+lúc\b", re.I), (60, 3600)),
    (re.compile(r"\bmột\s+chút\b", re.I), (30, 1800)),
]

# Exact unit multipliers (ordered largest→smallest so greedy match avoids re-counting)
_EXACT_EN: list[tuple[re.Pattern, float]] = [
    (re.compile(r"(\d+(?:\.\d+)?)\s*decades?", re.I), 315360000.0),
    (re.compile(r"(\d+(?:\.\d+)?)\s*(?:years?|yr)", re.I), 31536000.0),
    (re.compile(r"(\d+(?:\.\d+)?)\s*months?", re.I), 2592000.0),
    (re.compile(r"(\d+(?:\.\d+)?)\s*weeks?", re.I), 604800.0),
    (re.compile(r"(\d+(?:\.\d+)?)\s*days?", re.I), 86400.0),
    (re.compile(r"(\d+(?:\.\d+)?)\s*hours?", re.I), 3600.0),
    (re.compile(r"(\d+(?:\.\d+)?)\s*(?:minutes?|mins?)", re.I), 60.0),
    (re.compile(r"(\d+(?:\.\d+)?)\s*(?:seconds?|secs?)", re.I), 1.0),
]

_EXACT_VI: list[tuple[re.Pattern, float]] = [
    (re.compile(r"(\d+(?:\.\d+)?)\s*thập\s*kỷ", re.I), 315360000.0),
    (re.compile(r"(\d+(?:\.\d+)?)\s*năm", re.I), 31536000.0),
    (re.compile(r"(\d+(?:\.\d+)?)\s*tháng", re.I), 2592000.0),
    (re.compile(r"(\d+(?:\.\d+)?)\s*tuần", re.I), 604800.0),
    (re.compile(r"(\d+(?:\.\d+)?)\s*ngày", re.I), 86400.0),
    (re.compile(r"(\d+(?:\.\d+)?)\s*(?:tiếng|giờ)", re.I), 3600.0),
    (re.compile(r"(\d+(?:\.\d+)?)\s*phút", re.I), 60.0),
    (re.compile(r"(\d+(?:\.\d+)?)\s*giây", re.I), 1.0),
]


def parse_duration_range(text: str, lang: str = "en") -> tuple[float, float] | None:
    """Parse a duration string into a (min_sec, max_sec) range.

    Exact values → (n, n); vague phrases → (lo, hi); None if unparseable.
    """
    # Try vague patterns first (they must match the whole candidate cleanly)
    vague_pats = _VAGUE_EN if lang == "en" else _VAGUE_VI
    for pat, rng in vague_pats:
        if pat.search(text):
            return rng

    # Try exact patterns (sum up all matched units)
    exact_pats = _EXACT_EN if lang == "en" else _EXACT_VI
    total = 0.0
    matched = False
    for pat, mult in exact_pats:
        for m in pat.finditer(text):
            total += float(m.group(1)) * mult
            matched = True
    if matched and total > 0:
        return (total, total)

    return None


# backward-compat alias used by tests/old code
def parse_duration_seconds(text: str, lang: str = "en") -> float | None:
    r = parse_duration_range(text, lang)
    if r is None:
        return None
    return (r[0] + r[1]) / 2.0


# ═══════════════════════════════════════════════════════════════════════════════
# Activity-duration knowledge base
# (min_seconds, max_seconds) — conservative bounds, intersected when
# multiple keywords match so specificity wins
# ═══════════════════════════════════════════════════════════════════════════════

_ACTIVITY_EN: dict[str, tuple[float, float]] = {
    # Sleep / rest
    "nap": (300, 7200),
    "sleep": (10800, 57600),
    "rest": (300, 28800),
    "doze": (120, 3600),
    # Personal hygiene
    "shower": (60, 1800),
    "bath": (300, 3600),
    "brush": (30, 300),
    "haircut": (300, 5400),
    "shave": (60, 900),
    "get dressed": (60, 900),
    # Food / drink
    "eat": (120, 7200),
    "meal": (120, 7200),
    "breakfast": (120, 3600),
    "lunch": (120, 5400),
    "dinner": (300, 10800),
    "snack": (30, 1800),
    "cook": (300, 14400),
    "bake": (300, 21600),
    "make tea": (60, 600),
    "tea": (60, 600),
    "coffee": (60, 900),
    "drink": (60, 3600),
    # Entertainment
    "movie": (3600, 1209600),    # watch (1-4h) OR film on location (days); wide range
    "film": (3600, 14400),
    "show": (1200, 14400),
    "episode": (900, 7200),
    "concert": (3600, 21600),
    "play": (1800, 14400),
    "performance": (1800, 18000),
    "game": (900, 28800),
    "match": (1800, 14400),
    "read": (300, 86400),
    "book": (3600, 86400),
    "chapter": (300, 7200),
    "listen to music": (300, 14400),
    "music": (60, 14400),
    "podcast": (600, 7200),
    # Travel / movement
    "drive": (60, 86400),
    "commute": (300, 14400),
    "walk": (60, 86400),
    "jog": (600, 14400),
    "run": (60, 86400),
    "marathon": (7200, 43200),
    "sprint": (5, 120),
    "race": (300, 86400),
    "hike": (1800, 86400),
    "flight": (1800, 86400),
    "fly": (1800, 86400),
    "trip": (1800, 2592000),
    "vacation": (43200, 2592000),
    "holiday": (43200, 2592000),
    "cruise": (86400, 1209600),
    "bike": (120, 86400),
    "swim": (300, 14400),
    "drive to": (60, 86400),
    # Medical
    "surgery": (1800, 86400),
    "operation": (1800, 86400),
    "appointment": (900, 14400),
    "checkup": (900, 7200),
    "therapy": (1800, 7200),
    "pregnancy": (23328000, 31536000),
    "childbirth": (1800, 86400),
    "labor": (1800, 86400),
    "recover": (3600, 2592000),
    "healing": (3600, 2592000),
    # Work / study
    "meeting": (300, 28800),
    "class": (2700, 10800),
    "lecture": (2700, 10800),
    "school": (14400, 32400),
    "work": (14400, 57600),
    "shift": (10800, 57600),
    "semester": (7776000, 14515200),
    "semester": (7776000, 14515200),
    "course": (3600, 14515200),
    "exam": (1800, 21600),
    "test": (300, 14400),
    "interview": (900, 7200),
    "presentation": (300, 14400),
    "speech": (60, 10800),
    "homework": (300, 14400),
    "study": (1800, 28800),
    "research": (3600, 31536000),
    "develop": (86400, 31536000),
    "build": (3600, 31536000),
    "project": (3600, 31536000),
    "write": (300, 86400),
    "type": (60, 14400),
    "post": (30, 1800),
    # Exercise / sport
    "workout": (600, 14400),
    "exercise": (300, 14400),
    "yoga": (600, 7200),
    "dance": (300, 14400),
    "tennis": (1800, 14400),
    "soccer": (4500, 10800),
    "football": (3600, 14400),
    "basketball": (3600, 10800),
    "baseball": (7200, 18000),
    "golf": (7200, 28800),
    # Social / events
    "phone call": (30, 7200),
    "call": (30, 7200),
    "conversation": (60, 7200),
    "chat": (30, 7200),
    "party": (3600, 57600),
    "wedding": (7200, 57600),
    "funeral": (3600, 21600),
    "ceremony": (1800, 21600),
    "event": (3600, 86400),
    "conference": (3600, 86400),
    "seminar": (3600, 28800),
    # Chores / maintenance
    "clean": (300, 28800),
    "laundry": (1800, 7200),
    "cook dinner": (900, 7200),
    "repair": (300, 86400),
    "fix": (60, 86400),
    "paint": (1800, 86400),
    "garden": (600, 28800),
    "plant": (300, 28800),
    "mow": (1800, 14400),
    # Mental / abstract
    "think": (1, 86400),
    "decide": (1, 86400),
    "plan": (300, 86400),
    "dream": (30, 3600),
    "meditate": (300, 7200),
}

_ACTIVITY_VI: dict[str, tuple[float, float]] = {
    "ngủ": (10800, 57600),
    "chợp mắt": (300, 7200),
    "tắm": (60, 3600),
    "ăn": (120, 7200),
    "ăn sáng": (120, 3600),
    "ăn trưa": (120, 5400),
    "ăn tối": (300, 10800),
    "nấu ăn": (300, 14400),
    "nấu": (300, 14400),
    "pha trà": (60, 600),
    "uống": (30, 3600),
    "đi bộ": (60, 86400),
    "chạy": (60, 86400),
    "bơi": (300, 14400),
    "đạp xe": (120, 86400),
    "lái xe": (60, 86400),
    "bay": (1800, 86400),
    "du lịch": (43200, 2592000),
    "kỳ nghỉ": (43200, 2592000),
    "nghe nhạc": (60, 14400),
    "xem phim": (3600, 14400),
    "đọc sách": (300, 86400),
    "đọc": (300, 86400),
    "chơi game": (900, 28800),
    "trò chuyện": (60, 7200),
    "họp": (300, 28800),
    "học": (1800, 32400),
    "bài tập": (300, 14400),
    "làm việc": (14400, 57600),
    "phát triển ứng dụng": (86400, 31536000),
    "ứng dụng": (86400, 31536000),
    "sự kiện": (3600, 86400),
    "hội nghị": (3600, 86400),
    "buổi họp": (300, 28800),
    "sửa xe": (300, 86400),
    "sửa chữa": (300, 86400),
    "dọn dẹp": (300, 28800),
    "trồng cây": (300, 28800),
    "tiệc": (3600, 57600),
    "đám cưới": (7200, 57600),
    "phẫu thuật": (1800, 86400),
    "chăm sóc": (300, 2592000),
    "điều trị": (1800, 2592000),
    "tập thể dục": (300, 14400),
    "yoga": (600, 7200),
    "thiền": (300, 7200),
    "chụp ảnh": (300, 86400),
    "nhiếp ảnh": (300, 86400),
    "liên hệ": (60, 3600),
    "gọi điện": (30, 7200),
    "nhắn tin": (5, 3600),
    "viết": (300, 86400),
    "đọc báo": (300, 7200),
    "xem tin tức": (300, 7200),
    "tập gym": (1800, 7200),
    "chơi thể thao": (1800, 14400),
    "mua sắm": (1800, 28800),
    "nấu bữa": (300, 7200),
}


def _match_activity(context: str, question: str) -> tuple[float, float] | None:
    text = (context + " " + question).lower()
    best: tuple[float, float] | None = None

    def _intersect(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
        lo = max(a[0], b[0])
        hi = min(a[1], b[1])
        return (lo, hi) if lo <= hi else (min(a[0], b[0]), max(a[1], b[1]))

    for keyword, rng in _ACTIVITY_EN.items():
        if keyword in text:
            best = _intersect(best, rng) if best else rng

    for keyword, rng in _ACTIVITY_VI.items():
        if keyword in text:
            best = _intersect(best, rng) if best else rng

    return best


# ═══════════════════════════════════════════════════════════════════════════════
# Nth-visit / periodic-event counting
# ═══════════════════════════════════════════════════════════════════════════════

_P_NTH_VISIT = re.compile(
    r"(\d+)(?:st|nd|rd|th)\s+(?:visit|trip|appointment|time)\b"
    r"(?:[^.]*?(?:start(?:ing|ed)?\s+(?:from\s+|in\s+)?)?)?"
    r"(?:\bon\s+)?(" + _MNAMES + r")\.?\s+(\d{4})"
    r"[^.]*?every\s+(week|month|year|day)",
    re.I,
)


def _check_nth_visit(question: str) -> datetime.date | None:
    """'5th visit starting from October 2009, every month' → Oct 2009 + 4 months."""
    m = _P_NTH_VISIT.search(question)
    if not m:
        return None
    n = int(m.group(1)) - 1
    start_mo = _MONTH_EN.get(m.group(2).lower())
    if not start_mo:
        return None
    try:
        year = int(m.group(3))
        unit = m.group(4).lower()
        start = datetime.date(year, start_mo, 1)
        from dateutil.relativedelta import relativedelta
        if unit == "month":
            return start + relativedelta(months=n)
        elif unit == "week":
            return start + datetime.timedelta(weeks=n)
        elif unit == "year":
            return start + relativedelta(years=n)
        elif unit == "day":
            return start + datetime.timedelta(days=n)
    except (ValueError, ImportError):
        pass
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Format helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt_en(d: datetime.date) -> str:
    return f"{d.month:02d}/{d.day:02d}/{d.year}"


def _fmt_vi(d: datetime.date) -> str:
    return f"Tháng {d.month}, {d.year}"


# ═══════════════════════════════════════════════════════════════════════════════
# Public solvers
# ═══════════════════════════════════════════════════════════════════════════════

def solve_date_arith(sample: dict) -> str | None:
    """Try to solve a date_arith sample deterministically.

    Returns formatted date string or None (→ caller uses LLM fallback).
    """
    lang = sample.get("language", "en")

    if lang == "vi":
        return _solve_vi_date(sample)

    # ── English path ──────────────────────────────────────────────────────────
    question = sample.get("question", "")
    text = (sample.get("context") or "") + " " + question

    # UK/European date format: DD/MM/YYYY instead of MM/DD/YYYY
    uk_fmt = bool(re.search(
        r"\b(?:UK|British|European?)\b[^.]*?\bformat\b|\bDD/MM\b|\bdd/mm\b",
        text, re.I
    ))

    # "first day of YEAR" — direct target resolution
    m_first = re.search(r"\bfirst\s+day\s+of\s+(\d{4})\b", question, re.I)
    if m_first:
        try:
            return _fmt_en(datetime.date(int(m_first.group(1)), 1, 1))
        except ValueError:
            pass

    # "Nth visit/trip/appointment … [MONTH YEAR] … every [unit]"
    nth = _check_nth_visit(question)
    if nth:
        return _fmt_en(nth)

    dates = _extract_dates_en(text, uk_fmt=uk_fmt)
    today = _find_today_en(text, dates)
    if today is None:
        return None

    # 1. Weekday navigation (last/next Monday …)
    wd_result = _parse_target_weekday_en(question, today)
    if wd_result:
        return _fmt_en(wd_result)

    # 2. Simple day/week delta
    phrase = _extract_target_phrase_en(question)
    if phrase:
        delta = _phrase_to_delta(phrase)
        if delta is not None:
            return _fmt_en(today + datetime.timedelta(days=delta))

    # 3. Month/year delta via relativedelta
    rel_result = _try_relativedelta_en(question, today)
    if rel_result:
        return _fmt_en(rel_result)

    return None


def solve_duration(sample: dict) -> str | None:
    """Try to solve a duration (yes/no) sample deterministically.

    Returns 'yes' | 'no' or None (→ caller uses LLM fallback).
    """
    lang = sample.get("language", "en")
    meta = sample.get("meta") or {}
    candidate_str = (meta.get("candidate_answer") or "").strip()

    if not candidate_str:
        candidate_str = sample.get("question", "")

    # 1. Parse candidate → (min_sec, max_sec)
    cand_range = parse_duration_range(candidate_str, lang)
    if cand_range is None:
        return None
    cand_lo, cand_hi = cand_range

    # 2. Match activity → plausibility range
    context = sample.get("context") or ""
    question = sample.get("question") or ""
    act_range = _match_activity(context, question)
    if act_range is None:
        return None
    act_lo, act_hi = act_range

    # 3. Overlap check
    # Ranges overlap when: cand_lo <= act_hi AND cand_hi >= act_lo
    overlap = (cand_lo <= act_hi) and (cand_hi >= act_lo)
    if overlap:
        # Confirm it's not a borderline case: at least 50% of cand range inside activity range
        inside_lo = max(cand_lo, act_lo)
        inside_hi = min(cand_hi, act_hi)
        cand_span = (cand_hi - cand_lo) or 1.0
        overlap_frac = (inside_hi - inside_lo) / cand_span
        if cand_lo == cand_hi or overlap_frac >= 0.5:
            return "yes"
        return None  # borderline — let LLM decide
    else:
        # No overlap — is it clearly outside (>2× beyond edge)?
        if cand_hi < act_lo / 2 or cand_lo > act_hi * 2:
            return "no"
        return None  # just outside range edge — ambiguous
