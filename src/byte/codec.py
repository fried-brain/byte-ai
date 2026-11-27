"""Conversion between Unicode text and raw UTF-8 byte IDs.

The byte IDs are the actual UTF-8 byte values (0..255), not learned tokens
or a subword vocabulary.
"""

from collections.abc import Iterable


def encode_text(text: str) -> list[int]:
    """Encode text as its UTF-8 byte values.

    Args:
        text: A Python Unicode string.

    Returns:
        A list containing one integer per UTF-8 byte.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    return list(text.encode("utf-8"))


def decode_bytes(byte_ids: Iterable[int], *, errors: str = "strict") -> str:
    """Decode UTF-8 byte values into Unicode text.

    Args:
        byte_ids: Iterable of integer byte values in the inclusive range 0..255.
        errors: UTF-8 decoding error policy: 'strict', 'replace', or 'ignore'.

    Returns:
        Decoded Unicode text.

    Raises:
        ValueError: If a value is outside 0..255 or is not an integer.
        UnicodeDecodeError: If the byte sequence is malformed and errors='strict'.
    """
    if errors not in {"strict", "replace", "ignore"}:
        raise ValueError("errors must be 'strict', 'replace', or 'ignore'")

    values: list[int] = []
    for value in byte_ids:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("each byte ID must be an integer between 0 and 255")
        if not 0 <= value <= 255:
            raise ValueError("each byte ID must be an integer between 0 and 255")
        values.append(value)

    return bytes(values).decode("utf-8", errors=errors)
