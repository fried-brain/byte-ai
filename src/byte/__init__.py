"""BYTE: experimental raw-byte text modeling tools."""

from byte.codec import decode_bytes, encode_text

from byte.masks import build_block_causal_mask
from byte.modeling import ByteEmbedding, LocalCausalEncoder

__all__ = [
    "ByteEmbedding",
    "LocalCausalEncoder",
    "build_block_causal_mask",
    "decode_bytes",
    "encode_text",
]
