"""Rule-based parameter extraction + deterministic executor for arithmetic MCQ.

Replaces LLM-based Stage 1 with regex parsing for all structured categories.
The executor always produces the mathematically correct answer — no token
generation, no arithmetic hallucinations.

Pipeline (called from mcq_predict):
    params  = parse_params(question, category)   # regex, instant, no LLM
    if params:
        computed = execute(params)               # deterministic Python
    else:
        computed = None                          # caller falls back to LLM

Categories handled by regex + executor
---------------------------------------
    Hour Adjustment (24h)  — HH:MM ± HH:MM
    Hour Adjustment (12h)  — HH:MM AM/PM ± HH:MM
    Year Shift             — N years after/before YYYY
    Month Shift            — N months after/before MonthName
    Week Identification    — ISO week number for MM-DD-YYYY
    Time Computation       — convert days/minutes/seconds; add/subtract min+sec
    Date Computation       — add/subtract days / months+days / weeks+days / years+months
    Application            — constant-speed distance-time (partial; others → None)
    Time Zone Conversion   — H AM/PM on Month D, YYYY in Zone1 → Zone2

All other sub-types return None from parse_params and the caller falls back
to LLM compute → LLM match.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta

# ─────────────────────────────────────────────────────────────────────────────
# Shared constants
# ─────────────────────────────────────────────────────────────────────────────

_MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
_MONTH_RE = "|".join(_MONTHS)

# Unicode minus U+2212 used in the 12h questions, plus ASCII hyphen
_MINUS = r"[-−]"
_OP    = r"(?P<op>[+\-−])"

_TZ_OFFSET: dict[str, float] = {
    "UTC": 0, "GMT": 0,
    "US/Eastern": -5,  "EST": -5,  "EDT": -5,
    "US/Central": -6,  "CST": -6,  "CDT": -6,
    "US/Mountain": -7, "MST": -7,  "MDT": -6,
    "US/Pacific": -7,  "PST": -8,  "PDT": -7,
    "US/Hawaii": -10,  "HST": -10,
    "Europe/London": 0, "BST": 1,
    "Europe/Paris": 1,  "CET": 1,   "CEST": 2,
    "Europe/Athens": 2, "EET": 2,   "EEST": 3,
    "Asia/Shanghai": 8, "Asia/Singapore": 8, "SGT": 8,
    "Asia/Tokyo": 9,    "JST": 9,
    "Asia/Kolkata": 5.5, "IST": 5.5,
    "Australia/Sydney": 10, "AEST": 10,
    "Asia/Riyadh": 3,
}


# ─────────────────────────────────────────────────────────────────────────────
# Regex parsers — one per category
# ─────────────────────────────────────────────────────────────────────────────

def _parse_hour_24h(q: str) -> dict | None:
    """'What is HH:MM ± HH:MM?'"""
    m = re.search(
        r"(\d{1,2}:\d{2})\s*" + _OP + r"\s*(\d{1,2}:\d{2})",
        q,
    )
    if not m:
        return None
    op = "+" if m.group("op") == "+" else "-"
    return {"task": "hour_24h", "time1": m.group(1), "op": op, "time2": m.group(3)}


def _parse_hour_12h(q: str) -> dict | None:
    """'What is HH:MM AM/PM ± HH:MM?'"""
    m = re.search(
        r"(\d{1,2}:\d{2})\s*(AM|PM)\s*" + _OP + r"\s*(\d{1,2}:\d{2})",
        q, re.IGNORECASE,
    )
    if not m:
        return None
    op = "+" if m.group("op") == "+" else "-"
    return {
        "task": "hour_12h",
        "time1": m.group(1),
        "period": m.group(2).upper(),
        "op": op,
        "time2": m.group(4),
    }


def _parse_year_shift(q: str) -> dict | None:
    """'Which year comes/was N years after/before YYYY?'"""
    m = re.search(r"(\d+)\s+years?\s+(after|before)\s+(\d{3,4})", q, re.IGNORECASE)
    if not m:
        return None
    return {
        "task": "year_shift",
        "n": int(m.group(1)),
        "direction": m.group(2).lower(),
        "year": int(m.group(3)),
    }


def _parse_month_shift(q: str) -> dict | None:
    """'Which month comes/was N months after/before MonthName?'"""
    m = re.search(
        rf"(\d+)\s+months?\s+(after|before)\s+({_MONTH_RE})",
        q, re.IGNORECASE,
    )
    if not m:
        return None
    return {
        "task": "month_shift",
        "n": int(m.group(1)),
        "direction": m.group(2).lower(),
        "month": m.group(3).capitalize(),
    }


def _parse_week_id(q: str) -> dict | None:
    """'In which week of year Y does MM-DD-YYYY occur?'"""
    m = re.search(r"(\d{2})-(\d{2})-(\d{4})", q)
    if not m:
        return None
    return {
        "task": "week_id",
        "month": int(m.group(1)),
        "day":   int(m.group(2)),
        "year":  int(m.group(3)),
    }


def _parse_time_computation(q: str) -> dict | None:
    """Handles convert and add/subtract minutes+seconds questions."""
    q_l = q.lower()

    # Convert N days into minutes
    m = re.search(r"convert\s+(\d+)\s+days?\s+into\s+minutes", q_l)
    if m:
        return {"task": "convert", "value": int(m.group(1)), "from": "days", "to": "minutes"}

    # Convert N minutes into hours
    m = re.search(r"convert\s+(\d+)\s+minutes?\s+into\s+hours", q_l)
    if m:
        return {"task": "convert", "value": int(m.group(1)), "from": "minutes", "to": "hours"}

    # Convert N days into hours
    m = re.search(r"convert\s+(\d+)\s+days?\s+into\s+hours", q_l)
    if m:
        return {"task": "convert", "value": int(m.group(1)), "from": "days", "to": "hours"}

    # Convert N seconds into hours
    m = re.search(r"convert\s+(\d+)\s+seconds?\s+into\s+hours", q_l)
    if m:
        return {"task": "convert", "value": int(m.group(1)), "from": "seconds", "to": "hours"}

    # Add M1 minutes S1 seconds and M2 minutes S2 seconds
    m = re.search(
        r"add\s+(\d+)\s+minutes?\s+(\d+)\s+seconds?\s+and\s+(\d+)\s+minutes?\s+(\d+)\s+seconds?",
        q_l,
    )
    if m:
        return {
            "task": "add_min_sec",
            "m1": int(m.group(1)), "s1": int(m.group(2)),
            "m2": int(m.group(3)), "s2": int(m.group(4)),
        }

    # Subtract M minutes S seconds from H hours M2 minutes
    m = re.search(
        r"subtract\s+(\d+)\s+minutes?\s+(\d+)\s+seconds?\s+from\s+(\d+)\s+hours?\s+(\d+)\s+minutes?",
        q_l,
    )
    if m:
        return {
            "task": "sub_from_hm",
            "sub_m": int(m.group(1)), "sub_s": int(m.group(2)),
            "base_h": int(m.group(3)), "base_m": int(m.group(4)),
        }

    return None


def _parse_date_computation(q: str) -> dict | None:
    """Handles add/subtract days / months+days / weeks+days / years+months."""
    q_l = q.lower()

    # "What will be the time X years and Y months after Month YYYY?"
    m = re.search(
        rf"(\d+)\s+years?\s+and\s+(\d+)\s+months?\s+after\s+({_MONTH_RE})\s+(\d{{3,4}})",
        q, re.IGNORECASE,
    )
    if m:
        return {
            "task": "date_years_months",
            "delta_years":  int(m.group(1)),
            "delta_months": int(m.group(2)),
            "base_month":   m.group(3).capitalize(),
            "base_year":    int(m.group(4)),
            "op": "add",
        }

    # "If you add/subtract N days to the date MM-DD-YYYY"
    m = re.search(
        r"(add|subtract)\s+(\d+)\s+days?\s+to\s+the\s+date\s+(\d{2}-\d{2}-\d{4})",
        q_l,
    )
    if m:
        return {
            "task": "date_days",
            "op":   m.group(1),
            "days": int(m.group(2)),
            "base": m.group(3),
        }

    # "If you add/subtract N months and M days to the date MM-DD-YYYY"
    m = re.search(
        r"(add|subtract)\s+(\d+)\s+months?\s+and\s+(\d+)\s+days?\s+to\s+the\s+date\s+(\d{2}-\d{2}-\d{4})",
        q_l,
    )
    if m:
        return {
            "task": "date_months_days",
            "op":     m.group(1),
            "months": int(m.group(2)),
            "days":   int(m.group(3)),
            "base":   m.group(4),
        }

    # "If you add/subtract N weeks and M days to the date MM-DD-YYYY"
    m = re.search(
        r"(add|subtract)\s+(\d+)\s+weeks?\s+and\s+(\d+)\s+days?\s+to\s+the\s+date\s+(\d{2}-\d{2}-\d{4})",
        q_l,
    )
    if m:
        return {
            "task": "date_weeks_days",
            "op":    m.group(1),
            "weeks": int(m.group(2)),
            "days":  int(m.group(3)),
            "base":  m.group(4),
        }

    return None


def _parse_application(q: str) -> dict | None:
    """Handles constant-speed distance-time and rest-walk questions."""
    q_l = q.lower()

    # "A transportation ... speed of S km/h ... distance of D km(eters) in minutes"
    m = re.search(
        r"speed\s+of\s+([\d.]+)\s*km/h.*?distance\s+of\s+([\d.]+)\s*kilo?m",
        q_l,
    )
    if m:
        return {
            "task": "speed_dist_minutes",
            "speed": float(m.group(1)),
            "dist":  float(m.group(2)),
        }

    # "walks at a speed of S km/hr ... rest for R minutes ... cover D km"
    m = re.search(
        r"speed\s+of\s+([\d.]+)\s*km/hr.*?rest\s+for\s+(\d+)\s+minutes.*?cover\s+(\d+)\s*km",
        q_l,
    )
    if m:
        return {
            "task": "rest_walk",
            "speed": float(m.group(1)),
            "rest":  int(m.group(2)),
            "dist":  int(m.group(3)),
        }

    return None


def _parse_tz_conversion(q: str) -> dict | None:
    """'If it's H AM/PM on Month D, YYYY in Zone, what's the time in Zone2?'"""
    m = re.search(
        rf"it'?s\s+(\d{{1,2}})\s*(AM|PM)\s+on\s+({_MONTH_RE})\s+(\d{{1,2}}),\s*(\d{{3,4}})"
        rf"\s+in\s+([A-Za-z/_]+),.*?in\s+([A-Za-z/_]+)",
        q, re.IGNORECASE,
    )
    if not m:
        return None
    return {
        "task":     "tz_conversion",
        "hour":     int(m.group(1)),
        "period":   m.group(2).upper(),
        "month":    m.group(3).capitalize(),
        "day":      int(m.group(4)),
        "year":     int(m.group(5)),
        "src_tz":   m.group(6),
        "tgt_tz":   m.group(7).rstrip("?.,"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Unified parser dispatch
# ─────────────────────────────────────────────────────────────────────────────

_PARSERS = {
    "Hour Adjustment (24h)": _parse_hour_24h,
    "Hour Adjustment (12h)": _parse_hour_12h,
    "Year Shift":            _parse_year_shift,
    "Month Shift":           _parse_month_shift,
    "Week Identification":   _parse_week_id,
    "Time Computation":      _parse_time_computation,
    "Date Computation":      _parse_date_computation,
    "Application":           _parse_application,
    "Time Zone Conversion":  _parse_tz_conversion,
}


def detect_category(question: str) -> str | None:
    """Infer the MCQ category from the question text using regex heuristics.

    Returns one of the 9 known category strings, or None if unrecognised.
    Order matters: more specific patterns are checked first.
    """
    q  = question.strip()
    ql = q.lower()

    # Hour Adjustment (12h) — has AM/PM before the operator
    if re.search(r"\d{1,2}:\d{2}\s*(AM|PM)", q, re.IGNORECASE):
        return "Hour Adjustment (12h)"

    # Hour Adjustment (24h) — "What is HH:MM ± HH:MM?"
    if re.search(r"what is\s+\d{1,2}:\d{2}\s*[+\-−]", ql):
        return "Hour Adjustment (24h)"

    # Week Identification
    if "which week of year" in ql:
        return "Week Identification"

    # Date Computation — years+months phrasing (must come before Month Shift)
    if re.search(r"\d+\s+years?\s+and\s+\d+\s+months?\s+after", ql):
        return "Date Computation"

    # Year Shift
    if re.search(r"\d+\s+years?\s+(after|before)\s+\d{3,4}", ql):
        return "Year Shift"

    # Month Shift
    if re.search(rf"\d+\s+months?\s+(after|before)\s+({_MONTH_RE})", q, re.IGNORECASE):
        return "Month Shift"

    # Time Computation — convert / add / subtract minutes+seconds
    if re.search(
        r"(convert\s+\d+\s+(days?|minutes?|seconds?)\s+into"
        r"|add\s+\d+\s+minutes?"
        r"|subtract\s+\d+\s+minutes?\s+\d+\s+seconds?\s+from)",
        ql,
    ):
        return "Time Computation"

    # Date Computation — add/subtract days / months+days / weeks+days
    if re.search(
        r"(add|subtract)\s+\d+\s+(days?|months?|weeks?)"
        r"(\s+and\s+\d+\s+\w+)?\s+(to|from)\s+the\s+date",
        ql,
    ):
        return "Date Computation"

    # Time Zone Conversion — "it's H AM/PM ... in <zone>" pattern
    if re.search(r"it'?s\s+\d{1,2}\s*(am|pm)", ql) and re.search(
        r"\bin\s+([A-Za-z_]+(?:/[A-Za-z_]+)?)", ql
    ):
        return "Time Zone Conversion"

    # Application — speed/distance word problems
    if re.search(r"(km/h|km/hr|kilometers?|distance|speed)", ql):
        return "Application"

    return None


def parse_params(question: str, category: str) -> dict | None:
    """Extract structured parameters from a question using regex (no LLM).

    Returns a params dict on success, None if the category is unknown or
    the question does not match any known pattern.
    """
    parser = _PARSERS.get(category)
    if parser is None:
        return None
    return parser(question)


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic executor
# ─────────────────────────────────────────────────────────────────────────────

def execute(params: dict) -> str | None:
    """Run the extracted params deterministically and return the answer string.

    Returns None if execution fails (caller should fall back to LLM compute).
    """
    task = params.get("task", "")

    # ── Hour Adjustment 24h ──────────────────────────────────────────────────
    if task == "hour_24h":
        try:
            h1, m1 = map(int, params["time1"].split(":"))
            h2, m2 = map(int, params["time2"].split(":"))
            base  = h1 * 60 + m1
            delta = h2 * 60 + m2
            res = (base + delta) % 1440 if params["op"] == "+" else (base - delta) % 1440
            return f"{res // 60:02d}:{res % 60:02d}"
        except Exception:
            return None

    # ── Hour Adjustment 12h ──────────────────────────────────────────────────
    elif task == "hour_12h":
        try:
            h, m   = map(int, params["time1"].split(":"))
            period = params["period"]
            h2, m2 = map(int, params["time2"].split(":"))
            if period == "PM" and h != 12:
                h += 12
            elif period == "AM" and h == 12:
                h = 0
            base  = h * 60 + m
            delta = h2 * 60 + m2
            res = (base + delta) % 1440 if params["op"] == "+" else (base - delta) % 1440
            rh, rm = res // 60, res % 60
            p_out  = "AM" if rh < 12 else "PM"
            rh12   = rh % 12 or 12
            return f"{rh12}:{rm:02d} {p_out}"
        except Exception:
            return None

    # ── Year Shift ───────────────────────────────────────────────────────────
    elif task == "year_shift":
        try:
            n, year = int(params["n"]), int(params["year"])
            result = year + n if params["direction"] == "after" else year - n
            return str(result)
        except Exception:
            return None

    # ── Month Shift ──────────────────────────────────────────────────────────
    elif task == "month_shift":
        try:
            n    = int(params["n"])
            idx  = next(i for i, mn in enumerate(_MONTHS)
                        if mn.lower() == params["month"].lower())
            res  = (idx + n) % 12 if params["direction"] == "after" else (idx - n) % 12
            return _MONTHS[res]
        except Exception:
            return None

    # ── Week Identification ───────────────────────────────────────────────────
    elif task == "week_id":
        try:
            d = date(params["year"], params["month"], params["day"])
            return f"Week {d.isocalendar()[1]}"
        except Exception:
            return None

    # ── Time Computation: unit conversion ────────────────────────────────────
    elif task == "convert":
        try:
            v, frm, to = params["value"], params["from"], params["to"]
            if frm == "days"    and to == "minutes": return str(v * 24 * 60)
            if frm == "days"    and to == "hours":   return str(v * 24)
            if frm == "minutes" and to == "hours":
                h = v / 60
                return str(int(h) if h == int(h) else round(h, 4))
            if frm == "seconds" and to == "hours":
                h = v / 3600
                return str(int(h) if h == int(h) else round(h, 4))
        except Exception:
            return None

    # ── Time Computation: add minutes+seconds ────────────────────────────────
    elif task == "add_min_sec":
        try:
            total_s = params["s1"] + params["s2"]
            total_m = params["m1"] + params["m2"] + total_s // 60
            rem_s   = total_s % 60
            return f"{total_m} minutes {rem_s} seconds"
        except Exception:
            return None

    # ── Time Computation: subtract minutes+seconds from hours+minutes ────────
    elif task == "sub_from_hm":
        try:
            base_s  = params["base_h"] * 3600 + params["base_m"] * 60
            sub_s   = params["sub_m"]  * 60   + params["sub_s"]
            res_s   = base_s - sub_s
            if res_s < 0:
                return None
            rm, rs = divmod(res_s, 60)
            rh, rm = divmod(rm, 60)
            if rh > 0:
                return (f"{rh} hours {rm} minutes {rs} seconds"
                        if rs else f"{rh} hours {rm} minutes")
            return f"{rm} minutes {rs} seconds" if rs else f"{rm} minutes"
        except Exception:
            return None

    # ── Date Computation: years + months ─────────────────────────────────────
    elif task == "date_years_months":
        try:
            m_idx = next(i for i, mn in enumerate(_MONTHS)
                         if mn.lower() == params["base_month"].lower())
            total_m = (m_idx + 1) + params["delta_months"] + params["delta_years"] * 12
            year    = params["base_year"] + (total_m - 1) // 12
            month   = (total_m - 1) % 12
            return f"{_MONTHS[month]} {year}"
        except Exception:
            return None

    # ── Date Computation: add/subtract days ──────────────────────────────────
    elif task == "date_days":
        try:
            mm, dd, yyyy = params["base"].split("-")
            d = date(int(yyyy), int(mm), int(dd))
            delta = timedelta(days=params["days"])
            result = d + delta if params["op"] == "add" else d - delta
            return result.strftime("%m-%d-%Y")
        except Exception:
            return None

    # ── Date Computation: add/subtract months + days ─────────────────────────
    elif task == "date_months_days":
        try:
            from dateutil.relativedelta import relativedelta
            mm, dd, yyyy = params["base"].split("-")
            d = date(int(yyyy), int(mm), int(dd))
            rd = relativedelta(months=params["months"]) + timedelta(days=params["days"])
            result = d + rd if params["op"] == "add" else d - rd
            return result.strftime("%m-%d-%Y")
        except Exception:
            return None

    # ── Date Computation: add/subtract weeks + days ──────────────────────────
    elif task == "date_weeks_days":
        try:
            mm, dd, yyyy = params["base"].split("-")
            d = date(int(yyyy), int(mm), int(dd))
            delta = timedelta(weeks=params["weeks"], days=params["days"])
            result = d + delta if params["op"] == "add" else d - delta
            return result.strftime("%m-%d-%Y")
        except Exception:
            return None

    # ── Application: speed-distance → minutes ────────────────────────────────
    elif task == "speed_dist_minutes":
        try:
            mins = params["dist"] / params["speed"] * 60
            return str(int(round(mins)))
        except Exception:
            return None

    # ── Application: rest-walk ───────────────────────────────────────────────
    elif task == "rest_walk":
        try:
            walk_min  = params["dist"] / params["speed"] * 60
            rest_min  = (params["dist"] - 1) * params["rest"]
            total     = walk_min + rest_min
            return str(int(total))
        except Exception:
            return None

    # ── Time Zone Conversion ─────────────────────────────────────────────────
    elif task == "tz_conversion":
        try:
            src_off = _TZ_OFFSET.get(params["src_tz"])
            tgt_off = _TZ_OFFSET.get(params["tgt_tz"])
            if src_off is None or tgt_off is None:
                return None
            h = params["hour"]
            if params["period"] == "PM" and h != 12:
                h += 12
            elif params["period"] == "AM" and h == 12:
                h = 0
            diff     = int(tgt_off - src_off)
            new_h24  = (h + diff) % 24
            day_delta = (h + diff) // 24
            p_out    = "AM" if new_h24 < 12 else "PM"
            rh12     = new_h24 % 12 or 12
            # Reconstruct date
            try:
                m_idx = next(i for i, mn in enumerate(_MONTHS)
                             if mn.lower() == params["month"].lower())
                d = date(params["year"], m_idx + 1, params["day"])
                d += timedelta(days=day_delta)
                month_out = _MONTHS[d.month - 1]
                return f"{rh12} {p_out} on {month_out} {d.day}, {d.year}"
            except Exception:
                return f"{rh12} {p_out}"
        except Exception:
            return None

    return None
