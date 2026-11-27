"""Dataset encoding, deterministic byte windows, and split utilities."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch

from byte.codec import encode_text


def encode_corpus(text: str) -> list[int]:
    """Convert a Unicode corpus to its exact UTF-8 byte sequence."""
    return encode_text(text)


def split_byte_ids(
    byte_ids: Sequence[int],
    train_fraction: float = 0.9,
    validation_fraction: float = 0.05,
) -> tuple[list[int], list[int], list[int]]:
    """Split one corpus contiguously into train, validation, and test regions.

    Windows are sampled separately from each returned region, so no batch can
    cross a split boundary.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between zero and one")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction plus validation_fraction must be below one")

    values = list(byte_ids)
    train_end = int(len(values) * train_fraction)
    validation_end = int(len(values) * (train_fraction + validation_fraction))
    return values[:train_end], values[train_end:validation_end], values[validation_end:]


def make_next_byte_batches(
    byte_ids: Sequence[int] | torch.Tensor,
    block_size: int,
    sequence_length: int,
    batch_size: int,
    seed: int,
) -> Iterator[dict[str, torch.Tensor]]:
    """Yield deterministic random windows from one already-separated split.

    Each input window contains sequence_length + 1 bytes. target_ids contains
    the next-byte labels aligned with logits at input positions 0 through L-1.
    The BYTE model computes the same shift internally for its prediction loss.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if sequence_length < block_size:
        raise ValueError("sequence_length must be at least block_size")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    values = torch.as_tensor(byte_ids, dtype=torch.long)
    if values.ndim != 1:
        raise ValueError("byte_ids must be a one-dimensional sequence")
    if values.numel() and ((values < 0).any() or (values > 255).any()):
        raise ValueError("byte_ids must contain only integer values from 0 to 255")

    window_size = sequence_length + 1
    max_start = values.numel() - window_size
    if max_start < 0:
        raise ValueError(
            "byte_ids must contain at least sequence_length + 1 bytes"
        )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    while True:
        starts = torch.randint(
            low=0,
            high=max_start + 1,
            size=(batch_size,),
            generator=generator,
        )
        windows = torch.stack(
            [values[start : start + window_size] for start in starts.tolist()]
        )
        yield {
            "input_ids": windows,
            "target_ids": windows[:, 1:].clone(),
            "attention_mask": torch.ones_like(windows, dtype=torch.bool),
        }
