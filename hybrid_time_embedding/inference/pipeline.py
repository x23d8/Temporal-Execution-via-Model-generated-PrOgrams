"""
pipeline.py — TemporalQAPipeline: high-level interface for loading a trained
HybridTemporalModel checkpoint and running temporal QA inference.
"""

import os
from typing import Dict, List, Optional, Union

import torch
from transformers import AutoTokenizer

from ..src.models.full_model import HybridTemporalModel, TIME_START_TOKEN, TIME_END_TOKEN
from ..src.utils.config import HybridConfig
from ..src.data.preprocessing import build_input_text, normalize_timestamp, extract_timestamps
from ..src.training.callbacks import SmartCheckpointSaver


class TemporalQAPipeline:
    """
    End-to-end inference pipeline for the Hybrid Temporal Model.

    Loads a checkpoint, tokenizes inputs, runs the model, and returns
    denormalized predictions with confidence estimates.

    Args:
        checkpoint_dir: Path to a checkpoint folder (must contain model.safetensors + config.json).
        device: Torch device string.
    """

    def __init__(self, checkpoint_dir: str, device: str = "cuda") -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        config_path = os.path.join(checkpoint_dir, "config.json")
        self.config = HybridConfig.load(config_path)

        self.model = HybridTemporalModel.from_pretrained(self.config)
        self._load_weights(checkpoint_dir)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = self.model.tokenizer

        self.year_min = 0.0
        self.year_max = 2100.0

    def _load_weights(self, checkpoint_dir: str) -> None:
        from safetensors.torch import load_file
        state_dict = load_file(os.path.join(checkpoint_dir, "model.safetensors"))
        self.model.load_state_dict(state_dict, strict=False)

    @classmethod
    def from_best_checkpoint(cls, output_dir: str, device: str = "cuda") -> "TemporalQAPipeline":
        """
        Auto-detect and load the best checkpoint from a manifest.

        Args:
            output_dir: Models directory containing manifest.json.
            device: Torch device string.

        Returns:
            TemporalQAPipeline initialized from best checkpoint.
        """
        import json
        manifest_path = os.path.join(output_dir, "manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)
        checkpoints = sorted(manifest["checkpoints"], key=lambda c: c["val_mae"])
        best_folder = os.path.join(output_dir, checkpoints[0]["folder"])
        return cls(best_folder, device=device)

    @torch.no_grad()
    def predict(
        self,
        query: str,
        context: str = "",
        timestamps: Optional[List[float]] = None,
        subtask: str = "auto",
    ) -> Dict:
        """
        Run inference on a single query.

        Args:
            query: Question string.
            context: Optional background context.
            timestamps: Optional explicit year timestamps. Auto-extracted if None.
            subtask: "arithmetic", "duration", or "auto" (uses both heads).

        Returns:
            Dict with arith_pred, dur_pred, primary_timestamp, raw_year_arith, raw_year_dur.
        """
        if timestamps is None:
            timestamps = extract_timestamps(f"{context} {query}")
        if not timestamps:
            timestamps = [2000.0]

        text = build_input_text(query, context, timestamps)
        encoding = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        ts_norm = torch.tensor(
            [normalize_timestamp(timestamps[0], self.year_min, self.year_max)],
            dtype=torch.float32,
            device=self.device,
        )

        arith_pred, dur_pred, _ = self.model(input_ids, attention_mask, ts_norm)

        raw_arith = arith_pred.item() * (self.year_max - self.year_min) + self.year_min
        raw_dur = dur_pred.item() * (self.year_max - self.year_min)

        return {
            "arith_pred": arith_pred.item(),
            "dur_pred": dur_pred.item(),
            "raw_year_arith": raw_arith,
            "raw_duration": raw_dur,
            "primary_timestamp": timestamps[0],
            "gate_value": self.model.gate_value,
        }
