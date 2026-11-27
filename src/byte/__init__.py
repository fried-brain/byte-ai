"""BYTE: experimental raw-byte text modeling tools."""

from byte.codec import decode_bytes, encode_text
from byte.compression import (
    AdaptiveCompressor,
    CompressionOutput,
    compression_budget_loss,
    pack_active_latents,
)
from byte.masks import build_block_causal_mask
from byte.model import BYTEConfig, BYTEModel, BYTEOutput, sinusoidal_position_encoding
from byte.modeling import ByteEmbedding, LocalCausalEncoder

__all__ = [
    "AdaptiveCompressor",
    "BYTEConfig",
    "BYTEModel",
    "BYTEOutput",
    "ByteEmbedding",
    "CompressionOutput",
    "LocalCausalEncoder",
    "build_block_causal_mask",
    "compression_budget_loss",
    "decode_bytes",
    "encode_text",
    "pack_active_latents",
    "sinusoidal_position_encoding",
]
