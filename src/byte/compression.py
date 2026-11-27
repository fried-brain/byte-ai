"""Learned content-dependent compression for completed byte blocks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class CompressionOutput:
    """Outputs from the adaptive compressor.

    Packed tensors are populated only when inference=True.
    """

    latents: torch.Tensor
    gates: torch.Tensor
    packed_latents: torch.Tensor | None = None
    block_indices: torch.Tensor | None = None
    slot_indices: torch.Tensor | None = None


def pack_active_latents(
    latents: torch.Tensor,
    gates: torch.Tensor,
    valid_blocks: torch.Tensor,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack threshold-active latents in stable batch/block/slot order.

    Returns:
        packed_latents: [active_slots, d_model]
        block_indices: [active_slots, 2] containing batch and block indices
        slot_indices: [active_slots] containing the latent slot index
    """
    if latents.ndim != 4:
        raise ValueError("latents must have shape [batch, blocks, slots, d_model]")
    if gates.shape != latents.shape[:3]:
        raise ValueError("gates must have shape [batch, blocks, slots]")
    if valid_blocks.shape != latents.shape[:2]:
        raise ValueError("valid_blocks must have shape [batch, blocks]")
    if valid_blocks.dtype != torch.bool:
        raise TypeError("valid_blocks must be a boolean tensor")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    active = (gates >= threshold) & valid_blocks.unsqueeze(-1)
    coordinates = torch.nonzero(active, as_tuple=False)

    if coordinates.numel() == 0:
        return (
            latents.new_empty((0, latents.shape[-1])),
            torch.empty((0, 2), dtype=torch.long, device=latents.device),
            torch.empty((0,), dtype=torch.long, device=latents.device),
        )

    batch_indices, block_indices, slot_indices = coordinates.unbind(dim=1)
    packed = latents[batch_indices, block_indices, slot_indices]
    block_coordinates = coordinates[:, :2]
    return packed, block_coordinates, slot_indices


def compression_budget_loss(
    gates: torch.Tensor,
    valid_blocks: torch.Tensor,
    target_active_slots: float = 2.0,
) -> torch.Tensor:
    """Penalize deviations from a target mean gate sum per valid block."""
    if gates.ndim != 3:
        raise ValueError("gates must have shape [batch, blocks, slots]")
    if valid_blocks.shape != gates.shape[:2]:
        raise ValueError("valid_blocks must have shape [batch, blocks]")
    if valid_blocks.dtype != torch.bool:
        raise TypeError("valid_blocks must be a boolean tensor")
    if target_active_slots < 0:
        raise ValueError("target_active_slots must be non-negative")

    if not valid_blocks.any():
        return gates.sum() * 0.0

    per_block_gate_sum = gates.sum(dim=-1)
    mean_active_slots = per_block_gate_sum[valid_blocks].mean()
    target = gates.new_tensor(float(target_active_slots))
    return (mean_active_slots - target).square()


class AdaptiveCompressor(nn.Module):
    """Compress each completed block into gated, learned latent slots.

    The module accepts local encoder states shaped
    [batch, blocks, block_size, d_model]. Each block is compressed
    independently, so this module itself does not mix information across
    blocks. Causal ordering is enforced by the caller when these latents are
    added to the global context.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_slots: int = 4,
        nhead: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model <= 0 or num_slots <= 0 or nhead <= 0:
            raise ValueError("d_model, num_slots, and nhead must be positive")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        self.d_model = d_model
        self.num_slots = num_slots
        self.latent_queries = nn.Parameter(torch.empty(num_slots, d_model))
        nn.init.normal_(self.latent_queries, mean=0.0, std=d_model**-0.5)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.gate_network = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        block_states: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        *,
        inference: bool = False,
        threshold: float = 0.5,
    ) -> CompressionOutput:
        """Create candidate latents and gates for each completed byte block."""
        if block_states.ndim != 4:
            raise ValueError(
                "block_states must have shape [batch, blocks, block_size, d_model]"
            )
        batch, blocks, block_size, d_model = block_states.shape
        if d_model != self.d_model:
            raise ValueError(
                f"expected d_model={self.d_model}, received {d_model}"
            )

        expected_mask_shape = (batch, blocks, block_size)
        if valid_mask is None:
            valid_mask = torch.ones(
                expected_mask_shape,
                dtype=torch.bool,
                device=block_states.device,
            )
        elif valid_mask.shape != expected_mask_shape:
            raise ValueError(
                "valid_mask must have shape [batch, blocks, block_size]"
            )
        elif valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must be a boolean tensor")

        flat_states = block_states.reshape(batch * blocks, block_size, d_model)
        flat_valid = valid_mask.reshape(batch * blocks, block_size)
        valid_blocks = flat_valid.any(dim=-1)

        # MultiheadAttention produces NaNs when every key is masked. Temporarily
        # expose a dummy key for empty blocks, then zero their outputs below.
        safe_valid = flat_valid.clone()
        empty_rows = ~safe_valid.any(dim=-1)
        if empty_rows.any():
            safe_valid[empty_rows, 0] = True

        queries = self.latent_queries.unsqueeze(0).expand(
            batch * blocks, -1, -1
        )
        latent_states, _ = self.cross_attention(
            query=queries,
            key=flat_states,
            value=flat_states,
            key_padding_mask=~safe_valid,
            need_weights=False,
        )

        valid_weights = flat_valid.to(dtype=flat_states.dtype).unsqueeze(-1)
        denominator = valid_weights.sum(dim=1).clamp_min(1.0)
        summaries = (flat_states * valid_weights).sum(dim=1) / denominator
        summaries = summaries.unsqueeze(1).expand(-1, self.num_slots, -1)

        gate_inputs = torch.cat((latent_states, summaries), dim=-1)
        gate_values = torch.sigmoid(self.gate_network(gate_inputs)).squeeze(-1)

        valid_blocks = valid_blocks.reshape(batch, blocks)
        latent_states = latent_states.reshape(
            batch, blocks, self.num_slots, d_model
        )
        gate_values = gate_values.reshape(batch, blocks, self.num_slots)

        latent_states = latent_states.masked_fill(
            ~valid_blocks[:, :, None, None], 0.0
        )
        gate_values = gate_values.masked_fill(
            ~valid_blocks[:, :, None], 0.0
        )

        result = CompressionOutput(latents=latent_states, gates=gate_values)
        if inference:
            packed, block_indices, slot_indices = pack_active_latents(
                latent_states, gate_values, valid_blocks, threshold
            )
            result.packed_latents = packed
            result.block_indices = block_indices
            result.slot_indices = slot_indices
        return result
