"""Controlled non-hierarchical and fixed-compression BYTE baselines."""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F
from torch import nn

from byte.compression import CompressionOutput
from byte.model import BYTEConfig, BYTEModel, BYTEOutput, sinusoidal_position_encoding
from byte.modeling import ByteEmbedding


def _validate_inputs(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    if input_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("input_ids must contain integer byte IDs")
    if input_ids.shape[1] == 0:
        raise ValueError("input sequence must contain at least one byte")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    elif attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must have the same shape as input_ids")
    elif attention_mask.dtype != torch.bool:
        raise TypeError("attention_mask must be a boolean tensor")
    return input_ids.masked_fill(~attention_mask, 0), attention_mask


def _make_output(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> BYTEOutput:
    batch_size, sequence_length = input_ids.shape
    if sequence_length > 1:
        prediction_mask = attention_mask[:, :-1] & attention_mask[:, 1:]
        losses = F.cross_entropy(
            logits[:, :-1].reshape(-1, 256),
            input_ids[:, 1:].reshape(-1),
            reduction="none",
        ).reshape(batch_size, sequence_length - 1)
        prediction_loss = (
            losses * prediction_mask.to(losses.dtype)
        ).sum() / prediction_mask.sum().clamp_min(1)
    else:
        prediction_loss = logits.sum() * 0.0

    budget_loss = logits.sum() * 0.0
    empty_gates = logits.new_empty((batch_size, 0, 0))
    empty_valid = torch.empty(
        (batch_size, 0),
        dtype=torch.bool,
        device=logits.device,
    )
    return BYTEOutput(
        logits=logits,
        prediction_loss=prediction_loss,
        budget_loss=budget_loss,
        total_loss=prediction_loss,
        gates=empty_gates,
        valid_blocks=empty_valid,
    )


class StandardByteTransformer(nn.Module):
    """Full-context causal byte Transformer without hierarchical compression."""

    def __init__(self, config: BYTEConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = ByteEmbedding(config.vocab_size, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=4 * config.d_model,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.local_layers + config.global_layers,
            enable_nested_tensor=False,
        )
        self.output_head = nn.Linear(config.d_model, config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> BYTEOutput:
        safe_ids, attention_mask = _validate_inputs(input_ids, attention_mask)
        batch_size, sequence_length = safe_ids.shape
        states = self.embedding(safe_ids)
        states = states + sinusoidal_position_encoding(
            sequence_length,
            self.config.d_model,
            device=states.device,
            dtype=states.dtype,
        )

        safe_mask = attention_mask.clone()
        empty_rows = ~safe_mask.any(dim=1)
        if empty_rows.any():
            safe_mask[empty_rows, 0] = True
        causal_mask = torch.triu(
            torch.ones(
                (sequence_length, sequence_length),
                dtype=torch.bool,
                device=states.device,
            ),
            diagonal=1,
        )
        encoded = self.encoder(
            states,
            mask=causal_mask,
            src_key_padding_mask=~safe_mask,
        )
        encoded = encoded.masked_fill(~attention_mask.unsqueeze(-1), 0.0)
        logits = self.output_head(encoded)
        return _make_output(logits, safe_ids, attention_mask)


class NonHierarchicalByteGRU(nn.Module):
    """Unidirectional byte GRU baseline without hierarchical compression."""

    def __init__(self, config: BYTEConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = ByteEmbedding(config.vocab_size, config.d_model)
        layers = config.local_layers + config.global_layers
        self.recurrent = nn.GRU(
            input_size=config.d_model,
            hidden_size=config.d_model,
            num_layers=layers,
            batch_first=True,
            dropout=config.dropout if layers > 1 else 0.0,
        )
        self.output_head = nn.Linear(config.d_model, config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> BYTEOutput:
        safe_ids, attention_mask = _validate_inputs(input_ids, attention_mask)
        states = self.embedding(safe_ids)
        states = states + sinusoidal_position_encoding(
            safe_ids.shape[1],
            self.config.d_model,
            device=states.device,
            dtype=states.dtype,
        )
        states = states.masked_fill(~attention_mask.unsqueeze(-1), 0.0)
        encoded, _ = self.recurrent(states)
        encoded = encoded.masked_fill(~attention_mask.unsqueeze(-1), 0.0)
        logits = self.output_head(encoded)
        return _make_output(logits, safe_ids, attention_mask)


class FixedSlotCompressor(nn.Module):
    """Learn exactly two latent slots per valid block without adaptive gates."""

    def __init__(
        self,
        d_model: int,
        num_slots: int = 2,
        nhead: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.latent_queries = nn.Parameter(torch.empty(num_slots, d_model))
        nn.init.normal_(self.latent_queries, mean=0.0, std=d_model**-0.5)
        self.cross_attention = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self,
        block_states: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> CompressionOutput:
        batch, blocks, block_size, d_model = block_states.shape
        if valid_mask is None:
            valid_mask = torch.ones(
                (batch, blocks, block_size),
                dtype=torch.bool,
                device=block_states.device,
            )
        flat_states = block_states.reshape(batch * blocks, block_size, d_model)
        flat_valid = valid_mask.reshape(batch * blocks, block_size)
        safe_valid = flat_valid.clone()
        empty_rows = ~safe_valid.any(dim=-1)
        if empty_rows.any():
            safe_valid[empty_rows, 0] = True

        queries = self.latent_queries.unsqueeze(0).expand(batch * blocks, -1, -1)
        latents, _ = self.cross_attention(
            queries,
            flat_states,
            flat_states,
            key_padding_mask=~safe_valid,
            need_weights=False,
        )
        valid_blocks = flat_valid.any(dim=-1).reshape(batch, blocks)
        latents = latents.reshape(batch, blocks, self.num_slots, d_model)
        latents = latents.masked_fill(~valid_blocks[:, :, None, None], 0.0)
        gates = valid_blocks.to(dtype=latents.dtype).unsqueeze(-1).expand(
            -1, -1, self.num_slots
        )
        return CompressionOutput(latents=latents, gates=gates)


class FixedSlotBYTEModel(BYTEModel):
    """Hierarchical BYTE ablation with exactly two slots per valid block."""

    def __init__(self, config: BYTEConfig) -> None:
        fixed_config = replace(
            config,
            num_slots=2,
            target_active_slots=2.0,
        )
        super().__init__(fixed_config)
        self.compressor = FixedSlotCompressor(
            d_model=fixed_config.d_model,
            num_slots=2,
            nhead=fixed_config.nhead,
            dropout=fixed_config.dropout,
        )


def build_model(variant: str, config: BYTEConfig) -> nn.Module:
    """Construct one of the four controlled BYTE experiment variants."""
    if variant == "byte":
        return BYTEModel(config)
    if variant == "standard_transformer":
        return StandardByteTransformer(config)
    if variant == "byte_gru":
        return NonHierarchicalByteGRU(config)
    if variant == "fixed_hierarchical":
        return FixedSlotBYTEModel(config)
    raise ValueError(
        "variant must be one of: byte, standard_transformer, byte_gru, fixed_hierarchical"
    )
