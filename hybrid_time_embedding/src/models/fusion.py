"""
fusion.py — OptimalFusion (Layer 3).
Adds time embedding into token embeddings via a learned gate scalar.
Gate regularization penalizes gate values below the threshold.
"""

import torch
import torch.nn as nn


class OptimalFusion(nn.Module):
    """
    Fuses token embeddings (Layer 1) with time embedding (Layer 2) via gated add.

    Gate is initialized small (0.1) and learned during training.
    A regularization loss penalizes gate < gate_threshold to prevent collapse.

    Args:
        d_model: Hidden dimension (must match backbone and time embedding).
        gate_init: Initial gate scalar value.
        gate_threshold: Threshold below which gate_reg_loss activates.
    """

    def __init__(
        self,
        d_model: int = 3584,
        gate_init: float = 0.1,
        gate_threshold: float = 0.05,
    ) -> None:
        super().__init__()
        self.gate_threshold = gate_threshold
        self.gate = nn.Parameter(torch.tensor(gate_init))
        self.norm_token = nn.LayerNorm(d_model)
        self.norm_time = nn.LayerNorm(d_model)

    def forward(
        self,
        token_emb: torch.Tensor,
        time_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            token_emb: [batch, seq_len, d_model] token embeddings.
            time_emb:  [batch, d_model] time embedding from Layer 2.

        Returns:
            fused: [batch, seq_len, d_model] fused representation.
            gate_reg_loss: scalar regularization loss.

        Shape:
            Input:  token_emb [B, S, D], time_emb [B, D]
            Output: fused [B, S, D], gate_reg_loss scalar
        """
        time_broadcast = time_emb.unsqueeze(1).expand_as(token_emb)  # [B, S, D]
        fused = self.norm_token(token_emb) + self.gate * self.norm_time(time_broadcast)

        # Penalize if gate falls below threshold
        gate_reg_loss = torch.clamp(self.gate_threshold - self.gate, min=0.0) * 10.0
        return fused, gate_reg_loss

    @property
    def gate_value(self) -> float:
        """Current gate scalar as Python float."""
        return self.gate.item()
