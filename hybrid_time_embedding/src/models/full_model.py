"""
full_model.py — HybridTemporalModel: wires all 6 layers together.

Layer 1: Token Embedding (frozen Qwen2.5-7B embed_tokens)
Layer 2: OptimalHybridTimeEmbedding (trainable)
Layer 3: OptimalFusion (gated add, trainable)
Layer 4: Transformer Encoder with LoRA (partial freeze + LoRA adapters)
Layer 5: AttentionPooling + dual task heads (trainable)
Layer 6: GRPO RL phase (managed externally via PhaseAwareTrainer)
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .hybrid_time_emb import OptimalHybridTimeEmbedding
from .fusion import OptimalFusion
from .task_heads import AttentionPooling, ArithmeticHead, DurationHead
from ..utils.config import HybridConfig


TIME_START_TOKEN = "[TIME_START]"
TIME_END_TOKEN = "[TIME_END]"


class HybridTemporalModel(nn.Module):
    """
    Full 6-layer Hybrid Temporal Model for date arithmetic and duration QA.

    Combines frozen Qwen2.5-7B token embeddings with a trainable hybrid time
    embedding, fuses via a gated add, passes through LoRA-adapted transformer
    layers, and produces two scalar predictions via dual task heads.

    Args:
        config: HybridConfig with all hyperparameters.
        backbone: Pre-loaded Qwen2.5-7B model with LoRA applied.
        tokenizer: Tokenizer with [TIME_START]/[TIME_END] special tokens added.
    """

    def __init__(
        self,
        config: HybridConfig,
        backbone: nn.Module,
        tokenizer: AutoTokenizer,
    ) -> None:
        super().__init__()
        self.config = config

        # Store references to backbone components
        self.tokenizer = tokenizer
        self.backbone = backbone

        # Use backbone's actual hidden size — config.d_model may not match
        # (e.g. Qwen2.5-7B reports hidden_size=4096 at runtime despite docs)
        backbone_d = backbone.config.hidden_size

        # Layer 2
        self.time_embedding = OptimalHybridTimeEmbedding(
            d_model=backbone_d,
            n_learned_freq=config.n_learned_freq,
            n_random_freq=config.n_random_freq,
        )

        # Layer 3
        self.fusion = OptimalFusion(
            d_model=backbone_d,
            gate_init=config.gate_init,
            gate_threshold=config.gate_threshold,
        )

        # Layer 5
        self.pooler = AttentionPooling(d_model=backbone_d)
        self.arith_head = ArithmeticHead(d_model=backbone_d)
        self.dur_head = DurationHead(d_model=backbone_d)

        # Cast all custom layers to backbone compute dtype (e.g. bfloat16 with 4-bit quant)
        _compute_dtype = next(backbone.get_input_embeddings().parameters()).dtype
        for m in (self.time_embedding, self.fusion, self.pooler, self.arith_head, self.dur_head):
            m.to(_compute_dtype)

    def get_token_embeddings(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract frozen token embeddings from the backbone.

        Args:
            input_ids: [batch, seq_len] token IDs.

        Returns:
            [batch, seq_len, d_model] embedding matrix.
        """
        embed_layer = self.backbone.get_input_embeddings()
        return embed_layer(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        timestamps: torch.Tensor,
        start_times: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass through all 6 layers.

        Args:
            input_ids:      [batch, seq_len]
            attention_mask: [batch, seq_len]
            timestamps:     [batch] normalized primary timestamps
            start_times:    [batch] optional, used for consistency loss

        Returns:
            arith_pred:    [batch, 1] arithmetic prediction
            dur_pred:      [batch, 1] duration prediction (non-negative)
            gate_reg_loss: scalar gate regularization loss

        Shape summary:
            Layer 1 → [B, S, D]
            Layer 2 → [B, D]
            Layer 3 → [B, S, D]
            Layer 4 → [B, S, D]
            Layer 5 → arith[B,1], dur[B,1]
        """
        # Layer 1 — token embeddings (frozen)
        token_emb = self.get_token_embeddings(input_ids)  # [B, S, D]

        # Layer 2 — time embedding (align dtype to backbone compute dtype)
        time_emb = self.time_embedding(timestamps.to(token_emb.dtype))  # [B, D]

        # Layer 3 — fusion
        fused, gate_reg_loss = self.fusion(token_emb, time_emb)  # [B, S, D]

        # Layer 4 — transformer (inject fused embeddings via inputs_embeds)
        outputs = self.backbone(
            inputs_embeds=fused,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]  # [B, S, D]

        # Layer 5 — pooling + heads
        pooled = self.pooler(hidden, attention_mask)  # [B, D]
        arith_pred = self.arith_head(pooled)           # [B, 1]
        dur_pred = self.dur_head(pooled)               # [B, 1]

        return arith_pred, dur_pred, gate_reg_loss

    def freeze_backbone_layers(self, n_frozen: int) -> None:
        """
        Freeze the first n_frozen transformer blocks and the token embedding.

        Args:
            n_frozen: Number of initial transformer blocks to freeze.
        """
        # Freeze embed_tokens
        for p in self.backbone.get_input_embeddings().parameters():
            p.requires_grad = False

        # Freeze first n_frozen blocks (Qwen2 uses model.layers)
        layers = self.backbone.model.layers if hasattr(self.backbone, "model") else self.backbone.layers
        for i, layer in enumerate(layers):
            if i < n_frozen:
                for p in layer.parameters():
                    p.requires_grad = False

    def unfreeze_lora_layers(self) -> None:
        """Enable gradient for all LoRA adapter parameters."""
        for name, p in self.backbone.named_parameters():
            if "lora_" in name:
                p.requires_grad = True

    @property
    def gate_value(self) -> float:
        """Current fusion gate value."""
        return self.fusion.gate_value

    @classmethod
    def from_pretrained(
        cls,
        config: HybridConfig,
        load_in_4bit: bool = False,
    ) -> "HybridTemporalModel":
        """
        Load Qwen2.5-7B backbone, add special tokens, and wrap in HybridTemporalModel.

        Args:
            config: HybridConfig.
            load_in_4bit: Whether to load in 4-bit quantization (for inspection).

        Returns:
            HybridTemporalModel instance ready for training setup.
        """
        from transformers import BitsAndBytesConfig

        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name, trust_remote_code=True)
        tokenizer.add_special_tokens({"additional_special_tokens": [TIME_START_TOKEN, TIME_END_TOKEN]})
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        quant_config = None
        dtype = torch.bfloat16 if config.bf16 else torch.float32
        if load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )

        backbone = AutoModelForCausalLM.from_pretrained(
            config.base_model_name,
            torch_dtype=dtype,
            quantization_config=quant_config,
            trust_remote_code=True,
        )
        backbone.resize_token_embeddings(len(tokenizer))

        return cls(config=config, backbone=backbone, tokenizer=tokenizer)
