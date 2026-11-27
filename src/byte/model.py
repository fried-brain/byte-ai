"""Integrated causal BYTE language model."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn

from byte.compression import AdaptiveCompressor, compression_budget_loss
from byte.masks import build_block_causal_mask
from byte.modeling import ByteEmbedding, LocalCausalEncoder


@dataclass(frozen=True)
class BYTEConfig:
    """Configuration for the tiny BYTE v0.2 model."""

    vocab_size: int = 256
    d_model: int = 128
    block_size: int = 16
    local_layers: int = 2
    global_layers: int = 2
    nhead: int = 4
    num_slots: int = 4
    target_active_slots: float = 2.0
    gate_threshold: float = 0.5
    budget_weight: float = 0.01
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.vocab_size != 256:
            raise ValueError("BYTE requires exactly 256 raw byte IDs")
        if self.d_model <= 0 or self.d_model % 2 != 0:
            raise ValueError("d_model must be a positive even integer")
        if self.nhead <= 0 or self.d_model % self.nhead != 0:
            raise ValueError("d_model must be divisible by a positive nhead")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.local_layers <= 0 or self.global_layers <= 0:
            raise ValueError("local_layers and global_layers must be positive")
        if self.num_slots <= 0:
            raise ValueError("num_slots must be positive")
        if not 0.0 <= self.target_active_slots <= self.num_slots:
            raise ValueError("target_active_slots must be between zero and num_slots")
        if not 0.0 <= self.gate_threshold <= 1.0:
            raise ValueError("gate_threshold must be between zero and one")
        if self.budget_weight < 0.0:
            raise ValueError("budget_weight must be non-negative")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")


@dataclass
class BYTEOutput:
    """Model predictions, losses, and compression diagnostics."""

    logits: torch.Tensor
    prediction_loss: torch.Tensor
    budget_loss: torch.Tensor
    total_loss: torch.Tensor
    gates: torch.Tensor
    valid_blocks: torch.Tensor


def sinusoidal_position_encoding(
    length: int,
    d_model: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Create deterministic sinusoidal positions for any sequence length."""
    if length < 0:
        raise ValueError("length must be non-negative")
    if d_model <= 0 or d_model % 2 != 0:
        raise ValueError("d_model must be a positive even integer")

    positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    dimensions = torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
    frequencies = torch.exp(dimensions * (-math.log(10000.0) / d_model))
    angles = positions * frequencies.unsqueeze(0)

    encoding = torch.zeros((1, length, d_model), device=device, dtype=torch.float32)
    encoding[0, :, 0::2] = torch.sin(angles)
    encoding[0, :, 1::2] = torch.cos(angles)
    if dtype is not None:
        encoding = encoding.to(dtype=dtype)
    return encoding


class BYTEModel(nn.Module):
    """Tiny blockwise causal byte model with adaptive latent compression.

    Blocks are processed in order. A block's logits are computed using local
    causal states and a global context built only from completed earlier
    blocks. The current block is compressed only after its logits are made.
    """

    def __init__(self, config: BYTEConfig | None = None) -> None:
        super().__init__()
        self.config = config or BYTEConfig()
        cfg = self.config

        self.byte_embedding = ByteEmbedding(cfg.vocab_size, cfg.d_model)
        self.local_encoder = LocalCausalEncoder(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.local_layers,
            dropout=cfg.dropout,
        )
        self.compressor = AdaptiveCompressor(
            d_model=cfg.d_model,
            num_slots=cfg.num_slots,
            nhead=cfg.nhead,
            dropout=cfg.dropout,
        )

        global_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=4 * cfg.d_model,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.global_encoder = nn.TransformerEncoder(
            global_layer,
            num_layers=cfg.global_layers,
            enable_nested_tensor=False,
        )
        self.global_bos = nn.Parameter(torch.empty(1, 1, cfg.d_model))
        self.global_query = nn.Parameter(torch.empty(1, 1, cfg.d_model))
        self.slot_position_embedding = nn.Embedding(cfg.num_slots, cfg.d_model)
        nn.init.normal_(self.global_bos, mean=0.0, std=cfg.d_model**-0.5)
        nn.init.normal_(self.global_query, mean=0.0, std=cfg.d_model**-0.5)

        self.fusion_norm = nn.LayerNorm(cfg.d_model)
        self.prediction_head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def _global_context(
        self,
        prior_latents: list[torch.Tensor],
        prior_gates: list[torch.Tensor],
        prior_valid: list[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Encode only the latent slots belonging to completed prior blocks.

        Training keeps every slot with its differentiable gate applied.
        Evaluation physically selects only slots at or above gate_threshold.
        Each sample is encoded separately because adaptive slot counts vary.
        """
        contexts: list[torch.Tensor] = []
        for sample_index in range(batch_size):
            tokens = [self.global_bos[0, 0]]
            for block_index, (latents, gates, valid) in enumerate(
                zip(prior_latents, prior_gates, prior_valid)
            ):
                if not bool(valid[sample_index]):
                    continue

                if self.training:
                    slot_indices = torch.arange(
                        self.config.num_slots,
                        device=device,
                    )
                else:
                    slot_indices = torch.nonzero(
                        gates[sample_index] >= self.config.gate_threshold,
                        as_tuple=False,
                    ).flatten()

                if slot_indices.numel() == 0:
                    continue

                block_position = sinusoidal_position_encoding(
                    block_index + 1,
                    self.config.d_model,
                    device=device,
                    dtype=dtype,
                )[0, block_index]
                slot_positions = self.slot_position_embedding(slot_indices)
                selected_latents = latents[sample_index, slot_indices]
                selected_gates = gates[sample_index, slot_indices].unsqueeze(-1)
                selected = (
                    selected_latents * selected_gates
                    + block_position.unsqueeze(0)
                    + slot_positions
                )
                tokens.extend(selected.unbind(dim=0))

            query_position = sinusoidal_position_encoding(
                len(prior_latents) + 1,
                self.config.d_model,
                device=device,
                dtype=dtype,
            )[0, -1]
            tokens.append(self.global_query[0, 0] + query_position)

            sequence = torch.stack(tokens, dim=0).unsqueeze(0)
            causal_mask = build_block_causal_mask(
                sequence.shape[1],
                device=device,
            )
            causal_mask.fill_diagonal_(False)
            encoded = self.global_encoder(sequence, mask=causal_mask)
            contexts.append(encoded[0, -1, :])

        return torch.stack(contexts, dim=0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> BYTEOutput:
        """Return per-position next-byte logits and masked training losses."""
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must contain integer byte IDs")
        batch_size, sequence_length = input_ids.shape
        if sequence_length == 0:
            raise ValueError("input sequence must contain at least one byte")

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, sequence_length),
                dtype=torch.bool,
                device=input_ids.device,
            )
        elif attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids")
        elif attention_mask.dtype != torch.bool:
            raise TypeError("attention_mask must be a boolean tensor")

        loss_attention_mask = attention_mask
        target_ids = input_ids.masked_fill(~loss_attention_mask, 0)
        safe_ids = target_ids
        block_size = self.config.block_size
        num_blocks = (sequence_length + block_size - 1) // block_size
        padded_length = num_blocks * block_size
        padding_length = padded_length - sequence_length

        if padding_length:
            safe_ids = F.pad(safe_ids, (0, padding_length), value=0)
            attention_mask = F.pad(
                attention_mask,
                (0, padding_length),
                value=False,
            )

        embedded = self.byte_embedding(safe_ids)
        positions = sinusoidal_position_encoding(
            padded_length,
            self.config.d_model,
            device=embedded.device,
            dtype=embedded.dtype,
        )
        embedded = embedded + positions
        embedded = embedded.reshape(
            batch_size,
            num_blocks,
            block_size,
            self.config.d_model,
        )
        valid_bytes = attention_mask.reshape(batch_size, num_blocks, block_size)

        prior_latents: list[torch.Tensor] = []
        prior_gates: list[torch.Tensor] = []
        prior_valid: list[torch.Tensor] = []
        logits_by_block: list[torch.Tensor] = []
        gates_by_block: list[torch.Tensor] = []
        valid_blocks_by_block: list[torch.Tensor] = []

        for block_index in range(num_blocks):
            block_states = embedded[:, block_index : block_index + 1]
            block_valid = valid_bytes[:, block_index : block_index + 1]

            local_states = self.local_encoder(
                block_states,
                valid_mask=block_valid,
            )
            global_context = self._global_context(
                prior_latents,
                prior_gates,
                prior_valid,
                batch_size,
                embedded.device,
                embedded.dtype,
            )

            fused = self.fusion_norm(
                local_states + global_context[:, None, None, :]
            )
            block_logits = self.prediction_head(fused[:, 0])
            logits_by_block.append(block_logits)

            # Compress only after the current block's predictions are formed.
            compressed = self.compressor(
                local_states,
                valid_mask=block_valid,
            )
            block_latents = compressed.latents[:, 0]
            block_gates = compressed.gates[:, 0]

            block_is_valid = block_valid[:, 0].any(dim=-1)
            prior_latents.append(block_latents)
            prior_gates.append(block_gates)
            prior_valid.append(block_is_valid)
            gates_by_block.append(block_gates.unsqueeze(1))
            valid_blocks_by_block.append(block_is_valid.unsqueeze(1))

        logits = torch.cat(logits_by_block, dim=1)[:, :sequence_length]
        gates = torch.cat(gates_by_block, dim=1)
        valid_blocks = torch.cat(valid_blocks_by_block, dim=1)

        if sequence_length > 1:
            prediction_mask = (
                loss_attention_mask[:, :-1] & loss_attention_mask[:, 1:]
            )
            per_position_loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, self.config.vocab_size),
                target_ids[:, 1:].reshape(-1),
                reduction="none",
            ).reshape(batch_size, sequence_length - 1)
            masked_loss = per_position_loss * prediction_mask.to(
                dtype=per_position_loss.dtype
            )
            prediction_loss = masked_loss.sum() / prediction_mask.sum().clamp_min(1)
        else:
            prediction_loss = logits.sum() * 0.0

        budget_loss = compression_budget_loss(
            gates,
            valid_blocks,
            target_active_slots=self.config.target_active_slots,
        )
        total_loss = prediction_loss + self.config.budget_weight * budget_loss

        return BYTEOutput(
            logits=logits,
            prediction_loss=prediction_loss,
            budget_loss=budget_loss,
            total_loss=total_loss,
            gates=gates,
            valid_blocks=valid_blocks,
        )

    def predict_next(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return next-byte logits at each sample's last valid input position.

        This first implementation recomputes the supplied prefix and does not
        yet maintain a key/value cache between generation steps.
        """
        output = self.forward(input_ids, attention_mask=attention_mask)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        positions = torch.arange(
            input_ids.shape[1], device=input_ids.device
        ).expand_as(input_ids)
        last_valid = positions.masked_fill(~attention_mask, -1).max(dim=1).values
        if (last_valid < 0).any():
            raise ValueError("each sample must contain at least one valid byte")
        batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
        return output.logits[batch_indices, last_valid]
