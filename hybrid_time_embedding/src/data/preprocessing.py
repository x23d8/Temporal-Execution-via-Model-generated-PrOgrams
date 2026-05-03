"""
preprocessing.py — Text and timestamp preprocessing utilities.
Handles timestamp extraction from text, normalization to model input range,
and injection of special [TIME_START]/[TIME_END] tokens into query strings.
"""

import re
from typing import List, Optional, Tuple


# Year range used for normalization
_YEAR_MIN = 0.0
_YEAR_MAX = 2100.0


def extract_timestamps(text: str) -> List[float]:
    """
    Extract year-like floats from a text string.

    Args:
        text: Raw query or context string.

    Returns:
        List of extracted year floats found in the text.
    """
    pattern = r"\b(1[0-9]{3}|20[0-9]{2}|[0-9]{1,4}(?:\.[0-9]+)?)\b"
    candidates = re.findall(pattern, text)
    years = []
    for c in candidates:
        val = float(c)
        if 0.0 <= val <= 2100.0:
            years.append(val)
    return years


def normalize_timestamp(timestamp: float, year_min: float = _YEAR_MIN, year_max: float = _YEAR_MAX) -> float:
    """
    Normalize a year timestamp to [0, 1] range.

    Args:
        timestamp: Raw year value (e.g. 1939.0).
        year_min: Minimum year for normalization.
        year_max: Maximum year for normalization.

    Returns:
        Normalized float in [0, 1].
    """
    return (timestamp - year_min) / (year_max - year_min)


def denormalize_timestamp(value: float, year_min: float = _YEAR_MIN, year_max: float = _YEAR_MAX) -> float:
    """
    Convert normalized [0, 1] value back to year scale.

    Args:
        value: Normalized value.
        year_min: Minimum year used during normalization.
        year_max: Maximum year used during normalization.

    Returns:
        Year value in original scale.
    """
    return value * (year_max - year_min) + year_min


def add_time_tokens(text: str, timestamps: Optional[List[float]] = None) -> str:
    """
    Wrap numeric year mentions in the text with [TIME_START] and [TIME_END] tokens.

    Args:
        text: Raw text string.
        timestamps: Optional explicit list of timestamps to mark. If None,
                    years are auto-detected.

    Returns:
        Text string with [TIME_START]YEAR[TIME_END] markers inserted.
    """
    pattern = r"\b(1[0-9]{3}|20[0-9]{2})\b"

    def replacer(m: re.Match) -> str:
        val = float(m.group(0))
        if timestamps is None or val in timestamps:
            return f"[TIME_START]{m.group(0)}[TIME_END]"
        return m.group(0)

    return re.sub(pattern, replacer, text)


def build_input_text(query: str, context: str, timestamps: Optional[List[float]] = None) -> str:
    """
    Combine query and context into a single model input string with time tokens.

    Args:
        query: Question string.
        context: Background context string.
        timestamps: Year values to wrap with special tokens.

    Returns:
        Formatted input string ready for tokenization.
    """
    combined = f"{context} {query}".strip()
    return add_time_tokens(combined, timestamps)
