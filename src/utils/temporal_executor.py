"""Symbolic execution engine cho LLM-generated temporal programs.

Pipeline từ note.txt §2.3–2.5:
- Layer 3 (Program Synthesis): extract_code_block + clean_code chuẩn bị code LLM sinh ra
- Layer 4 (Symbolic Execution): execute_program chạy code trong sandbox datetime
- Layer 5 (Verification): verify_answer kiểm tra tính hợp lệ của kết quả
"""

from __future__ import annotations

import datetime
import re
import signal
from typing import Any

# ---------------------------------------------------------------------------
# Execution namespace — pre-populate datetime objects; dùng builtins thật
# để strftime và các C-level method hoạt động đúng trên mọi platform.
# Bảo vệ chủ yếu qua timeout (SIGALRM trên Linux/Colab).
# ---------------------------------------------------------------------------

SAFE_GLOBALS: dict[str, Any] = {
    # Pre-populate để LLM không cần viết import
    "date": datetime.date,
    "datetime": datetime.datetime,
    "timedelta": datetime.timedelta,
    # Weekday constants (Monday=0) để LLM dễ dùng weekday arithmetic
    "MONDAY": 0,
    "TUESDAY": 1,
    "WEDNESDAY": 2,
    "THURSDAY": 3,
    "FRIDAY": 4,
    "SATURDAY": 5,
    "SUNDAY": 6,
}

try:
    from dateutil.relativedelta import relativedelta  # type: ignore
    SAFE_GLOBALS["relativedelta"] = relativedelta
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Code extraction & cleaning
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(
    r"```(?:python)?\s*\n?(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def extract_code_block(text: str) -> str:
    """Trích code từ markdown ``` ... ``` block; fallback về toàn bộ text."""
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def clean_code(code: str) -> str:
    """Loại bỏ import statements (tên đã có trong SAFE_GLOBALS) và print()."""
    lines: list[str] = []
    for line in code.split("\n"):
        s = line.strip()
        if s.startswith("import ") or (
            s.startswith("from ") and " import " in s
        ):
            continue
        if s.startswith("print("):
            continue
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Symbolic execution
# ---------------------------------------------------------------------------

class _Timeout(Exception):
    pass


def _alarm_handler(signum: int, frame: Any) -> None:  # noqa: ANN401
    raise _Timeout


def execute_program(
    code: str,
    task: str = "",
    language: str = "",
    timeout_sec: int = 5,
) -> tuple[str | None, str | None]:
    """Chạy LLM-generated program trong sandbox và trả về (answer, error).

    answer: chuỗi kết quả lấy từ biến `answer` trong program.
    error:  mô tả lỗi nếu execution thất bại, else None.
    """
    prepped = clean_code(extract_code_block(code))
    local_vars: dict[str, Any] = {}

    # Dùng SIGALRM để timeout (chỉ hoạt động trên Linux/Colab; Windows bỏ qua)
    use_alarm = hasattr(signal, "SIGALRM")
    try:
        if use_alarm:
            signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(timeout_sec)

        exec(  # noqa: S102
            compile(prepped, "<temporal_program>", "exec"),
            dict(SAFE_GLOBALS),
            local_vars,
        )

        if use_alarm:
            signal.alarm(0)

    except _Timeout:
        return None, "Execution timed out"
    except Exception as exc:  # noqa: BLE001
        if use_alarm:
            signal.alarm(0)
        return None, f"{type(exc).__name__}: {exc}"

    answer = local_vars.get("answer")
    if answer is None:
        return None, "Variable 'answer' was not assigned"

    # Tự động convert datetime.date object sang chuỗi đúng format
    if isinstance(answer, datetime.date):
        if task == "date_arith" and language == "vi":
            answer = f"Tháng {answer.month}, {answer.year}"
        else:
            answer = answer.strftime("%m/%d/%Y")

    return str(answer).strip(), None


# ---------------------------------------------------------------------------
# Constraint verification (Layer 5)
# ---------------------------------------------------------------------------

_MMDDYYYY_RE = re.compile(r"^\d{2}/\d{2}/\d{4}$")
_VI_MONTH_RE = re.compile(r"^Tháng \d{1,2}, \d{3,4}$")


def verify_answer(answer: str | None, task: str, language: str) -> bool:
    """Kiểm tra output có hợp lệ về format và calendar constraints."""
    if not answer or not answer.strip():
        return False

    a = answer.strip()

    if task == "duration":
        return a.lower() in ("yes", "no")

    if task == "date_arith":
        if language == "en":
            if not _MMDDYYYY_RE.match(a):
                return False
            try:
                mm, dd, yyyy = (int(x) for x in a.split("/"))
                datetime.date(yyyy, mm, dd)  # raises if invalid calendar date
                return 1000 <= yyyy <= 2200
            except (ValueError, TypeError):
                return False

        if language == "vi":
            if not _VI_MONTH_RE.match(a):
                return False
            try:
                parts = a.replace("Tháng ", "").replace(",", "").split()
                month, year = int(parts[0]), int(parts[1])
                return 1 <= month <= 12 and 1000 <= year <= 2200
            except (ValueError, IndexError):
                return False

    return False
