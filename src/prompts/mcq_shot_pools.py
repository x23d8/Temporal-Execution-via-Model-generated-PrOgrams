"""Shot pools for the arithmetic MCQ task.

Shots are loaded from arithmetic_shots_mcq.csv and indexed by category.
The CSV path is supplied at load time (not hardcoded) so the module works
on any machine regardless of where the data directory lives.

Typical usage
-------------
    from src.prompts.mcq_shot_pools import load_mcq_shots, get_mcq_shots

    pool = load_mcq_shots('F:/arithmetic/arithmetic_shots_mcq.csv')
    shots = get_mcq_shots(pool, 'Year Shift', k=3)
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Union

from ..data.schema import McqSample

# Type alias: maps category name → ordered list of McqSample
ShotPool = dict[str, list[McqSample]]


def load_mcq_shots(csv_path: Union[str, Path]) -> ShotPool:
    """Load arithmetic_shots_mcq.csv and return a per-category shot pool.

    Args:
        csv_path: Absolute or relative path to the shots CSV file.
                  Expected columns: Question, Option A, Option B,
                  Option C, Option D, Answer, Category.

    Returns:
        Dict mapping category string → list[McqSample] in CSV row order.
    """
    pool: ShotPool = defaultdict(list)
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            pool[row["Category"]].append(
                McqSample(
                    sample_id=f"shot-mcq-{i}",
                    category=row["Category"],
                    dataset="arithmetic_shots_mcq",
                    question=row["Question"],
                    option_a=row["Option A"],
                    option_b=row["Option B"],
                    option_c=row["Option C"],
                    option_d=row["Option D"],
                    gold=row["Answer"].strip().upper(),
                )
            )
    return dict(pool)


def get_mcq_shots(
    pool: ShotPool,
    category: str,
    k: int,
    exclude_question: str | None = None,
) -> list[McqSample]:
    """Retrieve up to k shots for a given category.

    Args:
        pool:             Shot pool returned by load_mcq_shots().
        category:         Category string matching the test sample.
        k:                Maximum number of shots to return.
        exclude_question: If set, skip any shot whose question equals this
                          string (prevents leaking the test item as a demo).

    Returns:
        List of up to k McqSample objects (may be fewer if the pool is small).
    """
    candidates = pool.get(category, [])
    if exclude_question:
        candidates = [s for s in candidates if s["question"] != exclude_question]
    return candidates[:k]


def available_categories(pool: ShotPool) -> list[str]:
    """Return sorted list of categories present in the pool."""
    return sorted(pool.keys())
