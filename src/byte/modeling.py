"""Neural primitives for BYTE's raw-byte models."""

from __future__ import annotations

import torch
from torch import nn


class ByteEmbedding(nn.Module):
    """Map raw byte IDs (0..255) to learned vectors."""

    def __init__(self, vocab_size: int = 256, d_model: int = 128) -> None:
        super().__init__()
        if vocab_size != 256:
            raise ValueError("BYTE's raw UTF-8 byte vocabulary must contain 256 IDs")
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        """Embed integer byte IDs shaped [batch, sequence]."""
        if byte_ids.ndim != 2:
            raise ValueError(
                f"byte_ids must have shape [batch, sequence], got {tuple(byte_ids.shape)}"
            )
        return self.embedding(byte_ids)


class LocalCausalEncoder(nn.Module):
    """Apply a causal Transformer independently to each byte block.

    Input and output tensors have shape [batch, blocks, block_size, d_model].
    A True value in valid_mask marks a real byte; False marks padding.
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model <= 0 or nhead <= 0 or num_layers <= 0:
            raise ValueError("d_model, nhead, and num_layers must be positive")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.d_model = d_model

    def forward(
        self,
        block_states: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode blocks without allowing a position to attend to later bytes."""
        if block_states.ndim != 4:
            raise ValueError(
                "block_states must have shape [batch, blocks, block_size, d_model]"
            )
        batch, blocks, block_size, d_model = block_states.shape
        if d_model != self.d_model:
            raise ValueError(
                f"expected d_model={self.d_model}, received {d_model}"
            )

        if valid_mask is None:
            valid_mask = torch.ones(
                (batch, blocks, block_size),
                dtype=torch.bool,
                device=block_states.device,
            )
        elif valid_mask.shape != (batch, blocks, block_size):
            raise ValueError(
                "valid_mask must have shape [batch, blocks, block_size]"
            )
        elif valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must be a boolean tensor")

        flat_states = block_states.reshape(batch * blocks, block_size, d_model)
        flat_valid = valid_mask.reshape(batch * blocks, block_size)

        # Transformer attention returns NaNs if every key in a row is masked.
        # Temporarily expose one dummy key for wholly padded blocks; their
        # outputs are zeroed immediately after the encoder.
        safe_valid = flat_valid.clone()
        empty_rows = ~safe_valid.any(dim=1)
        if empty_rows.any():
            safe_valid[empty_rows, 0] = True

        causal_mask = torch.triu(
            torch.ones(
                (block_size, block_size),
                dtype=torch.bool,
                device=block_states.device,
            ),
            diagonal=1,
        )
        encoded = self.encoder(
            flat_states,
            mask=causal_mask,
            src_key_padding_mask=~safe_valid,
        )
        encoded = encoded.masked_fill(~flat_valid.unsqueeze(-1), 0.0)
        return encoded.reshape(batch, blocks, block_size, d_model)
