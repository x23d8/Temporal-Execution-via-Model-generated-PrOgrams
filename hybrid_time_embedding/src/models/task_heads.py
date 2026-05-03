"""
task_heads.py — Layer 5: AttentionPooling, ArithmeticHead, DurationHead.
The attention pooler aggregates the sequence with learned attention weights.
Both heads map pooled [batch, d_model] → scalar predictions.
DurationHead uses Softplus to ensure non-negative durations.
"""

import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    """
    Attention-weighted pooling over the sequence dimension.

    Learns a score for each token position and forms a weighted sum,
    masking padding tokens.

    Args:
        d_model: Hidden dimension of the sequence.
    """

    def __init__(self, d_model: int = 3584) -> None:
        super().__init__()
        self.attn_weights = nn.Linear(d_model, 1)

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden:         [batch, seq_len, d_model] encoder output.
            attention_mask: [batch, seq_len] int/bool mask (1 = valid, 0 = pad).

        Returns:
            Tensor of shape [batch, d_model] — pooled representation.
        """
        scores = self.attn_weights(hidden).squeeze(-1)  # [B, S]
        scores = scores.masked_fill(~attention_mask.bool(), -1e9)
        weights = torch.softmax(scores, dim=-1)  # [B, S]
        return (weights.unsqueeze(-1) * hidden).sum(dim=1)  # [B, D]


class ArithmeticHead(nn.Module):
    """
    Regression head for date arithmetic (predicts year value in R).

    Args:
        d_model: Input feature dimension.
        hidden_dim: Intermediate projection size.
    """

    def __init__(self, d_model: int = 3584, hidden_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, d_model] pooled features.

        Returns:
            [batch, 1] scalar predictions.
        """
        return self.net(x)


class DurationHead(nn.Module):
    """
    Regression head for date duration (predicts non-negative duration via Softplus).

    Args:
        d_model: Input feature dimension.
        hidden_dim: Intermediate projection size.
    """

    def __init__(self, d_model: int = 3584, hidden_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, d_model] pooled features.

        Returns:
            [batch, 1] non-negative duration predictions.
        """
        return self.net(x)
