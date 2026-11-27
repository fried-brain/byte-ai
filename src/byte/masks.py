"""Attention masks used by BYTE's hierarchical causal architecture."""

from __future__ import annotations

import torch


def build_block_causal_mask(
    num_blocks: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return a bool mask blocking the current and all future blocks.

    The result follows PyTorch attention-mask convention: True means that
    attention is disallowed. Query block j may therefore attend only to
    completed blocks with indices strictly smaller than j.
    """
    if isinstance(num_blocks, bool) or not isinstance(num_blocks, int):
        raise TypeError("num_blocks must be an integer")
    if num_blocks < 0:
        raise ValueError("num_blocks must be non-negative")

    positions = torch.arange(num_blocks, device=device)
    query_positions = positions.unsqueeze(1)
    key_positions = positions.unsqueeze(0)
    return key_positions >= query_positions
