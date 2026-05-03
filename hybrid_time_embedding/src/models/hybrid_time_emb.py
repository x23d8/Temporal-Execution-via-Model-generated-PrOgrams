"""
hybrid_time_emb.py — OptimalHybridTimeEmbedding (Layer 2).
Combines a linear branch and a toroidal Fourier branch to embed scalar
timestamps into d_model-dimensional vectors. Trainable at lr=1e-4.
"""

from typing import Optional

import torch
import torch.nn as nn


class OptimalHybridTimeEmbedding(nn.Module):
    """
    Hybrid temporal embedding combining linear and toroidal branches.

    Linear branch: maps scalar timestamp → d_model/2 via Linear + LayerNorm.
    Toroidal branch: P1 learned frequencies + P2 random Fourier features,
                     projected → d_model/2 via Linear + LayerNorm.
    Output: concatenated and LayerNorm'd → [batch, d_model].

    Args:
        d_model: Hidden size of the backbone model (3584 for Qwen2.5-7B).
        n_learned_freq: Number of learnable Fourier frequencies (P1).
        n_random_freq: Number of fixed random Fourier frequencies (P2).
    """

    def __init__(
        self,
        d_model: int = 3584,
        n_learned_freq: int = 8,
        n_random_freq: int = 16,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_learned_freq = n_learned_freq
        self.n_random_freq = n_random_freq
        half = d_model // 2
        n_torus = n_learned_freq + n_random_freq  # 24

        # ── Linear branch ──────────────────────────
        self.linear_proj = nn.Linear(1, half)
        self.linear_norm = nn.LayerNorm(half)

        # ── Toroidal branch ────────────────────────
        # P1: learnable log-frequencies initialized with temporal priors
        init_f = torch.log(torch.tensor([1.0, 0.5, 0.25, 4.0, 8.0, 10.0, 0.083, 0.02]))
        self.log_freq_learned = nn.Parameter(init_f)

        # P2: fixed random Fourier features (log-uniform in [-3, 3])
        freq_random = torch.exp(torch.FloatTensor(n_random_freq).uniform_(-3, 3))
        self.register_buffer("freq_random", freq_random)

        self.torus_proj = nn.Linear(n_torus * 2, half)  # sin+cos → half
        self.torus_norm = nn.LayerNorm(half)

        # ── Output norm ────────────────────────────
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timestamps: [batch] normalized float timestamps in [0, 1].

        Returns:
            Tensor of shape [batch, d_model].
        """
        # Linear branch
        linear_feat = self.linear_proj(timestamps.unsqueeze(-1))  # [B, half]
        linear_feat = self.linear_norm(linear_feat)

        # Toroidal branch
        all_freqs = torch.cat(
            [torch.exp(self.log_freq_learned), self.freq_random]
        )  # [n_torus]
        angles = timestamps.unsqueeze(-1) * all_freqs  # [B, n_torus]
        torus_raw = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # [B, 2*n_torus]
        torus_feat = self.torus_proj(torus_raw)  # [B, half]
        torus_feat = self.torus_norm(torus_feat)

        combined = torch.cat([linear_feat, torus_feat], dim=-1)  # [B, d_model]
        return self.out_norm(combined)
