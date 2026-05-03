"""
dataset.py — TemporalQADataset for loading date_arithmetic and date_duration
JSON splits. Each item provides text, timestamps, labels, and subtask metadata
for the Hybrid Time Embedding training pipeline.
"""

import json
import os
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from .preprocessing import normalize_timestamp, build_input_text


class TemporalQADataset(Dataset):
    """
    Dataset for temporal QA covering date_arithmetic and date_duration subtasks.

    Expected JSON schema per item:
    {
        "id": str,
        "subtask": "date_arithmetic" | "date_duration",
        "query": str,
        "context": str,
        "timestamps": [float, ...],
        "start_time": float,
        "duration": float (optional),
        "end_time": float (optional),
        "answer": float,
        "answer_type": "year" | "years"
    }

    Args:
        data_dir: Path to directory containing train/val/test JSON files.
        split: One of "train", "val", "test".
        subtasks: List of subtask names to include.
        year_min: Minimum year for timestamp normalization.
        year_max: Maximum year for timestamp normalization.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        subtasks: Optional[List[str]] = None,
        year_min: float = 0.0,
        year_max: float = 2100.0,
    ) -> None:
        self.year_min = year_min
        self.year_max = year_max
        self.subtasks = subtasks or ["date_arithmetic", "date_duration"]
        self.data: List[Dict] = []

        # Try combined file first, then per-subtask files
        combined_path = os.path.join(data_dir, f"{split}.json")
        if os.path.exists(combined_path):
            with open(combined_path) as f:
                raw = json.load(f)
            self.data = [r for r in raw if r.get("subtask") in self.subtasks]
        else:
            for subtask in self.subtasks:
                path = os.path.join(data_dir, subtask, f"{split}.json")
                if os.path.exists(path):
                    with open(path) as f:
                        raw = json.load(f)
                    self.data.extend(raw)

    @classmethod
    def from_samples(
        cls,
        samples,
        year_min: float = 0.0,
        year_max: float = 2100.0,
    ) -> "TemporalQADataset":
        """Build a dataset from ``src.data.registry.load_dataset`` Sample dicts.

        date_arith samples: gold is "MM/DD/YYYY" or similar → year extracted.
        duration  samples:  gold must be numeric (years); "yes"/"no" rows are skipped.
        """
        import re
        from .preprocessing import extract_timestamps

        def _parse_answer(gold: str, task: str) -> Optional[float]:
            if task == "date_arith":
                m = re.search(r"\b(\d{4})\b", gold)
                return float(m.group(1)) if m else None
            try:
                return float(gold)
            except ValueError:
                return None  # skip binary "yes"/"no" duration rows

        instance = object.__new__(cls)
        instance.year_min = year_min
        instance.year_max = year_max
        instance.subtasks = ["date_arithmetic", "date_duration"]
        instance.data = []

        for s in samples:
            task    = s.get("task", "")
            subtask = "date_arithmetic" if task == "date_arith" else "date_duration"
            answer  = _parse_answer(s.get("gold", ""), task)
            if answer is None:
                continue

            context  = s.get("context", "")
            question = s.get("question", "")
            timestamps = extract_timestamps(f"{context} {question}")
            start = timestamps[0] if timestamps else answer

            instance.data.append({
                "id":          s.get("sample_id", ""),
                "subtask":     subtask,
                "query":       question,
                "context":     context,
                "timestamps":  timestamps if timestamps else [start],
                "start_time":  start,
                "answer":      answer,
                "answer_type": "year" if task == "date_arith" else "years",
            })

        return instance

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns a single processed item.

        Returns dict with keys:
            text (str): Combined input text with time tokens.
            timestamp (float): Primary normalized timestamp.
            timestamps_raw (List[float]): All raw timestamps.
            start_time (float): Normalized start time.
            end_time (float): Normalized end time (0 if absent).
            arith_label (float): Ground truth for arithmetic subtask.
            dur_label (float): Ground truth for duration subtask.
            subtask (str): "date_arithmetic" or "date_duration".
            item_id (str): Original item identifier.
        """
        item = self.data[idx]
        timestamps = item.get("timestamps", [item.get("start_time", 0.0)])
        primary_ts = normalize_timestamp(timestamps[0], self.year_min, self.year_max)
        start_time = normalize_timestamp(item.get("start_time", timestamps[0]), self.year_min, self.year_max)
        end_time = normalize_timestamp(item.get("end_time", 0.0), self.year_min, self.year_max)

        text = build_input_text(item["query"], item.get("context", ""), timestamps)

        # Labels: arithmetic head predicts the answer year; duration head predicts duration
        answer = float(item["answer"])
        subtask = item.get("subtask", "date_arithmetic")
        arith_label = normalize_timestamp(answer, self.year_min, self.year_max) if subtask == "date_arithmetic" else 0.0
        dur_label = answer / (self.year_max - self.year_min) if subtask == "date_duration" else 0.0

        return {
            "text": text,
            "timestamp": primary_ts,
            "timestamps_raw": timestamps,
            "start_time": start_time,
            "end_time": end_time,
            "arith_label": arith_label,
            "dur_label": dur_label,
            "subtask": subtask,
            "item_id": item.get("id", str(idx)),
        }
