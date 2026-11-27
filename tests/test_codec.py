import pytest

from byte.codec import decode_bytes, encode_text


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Hello, BYTE!",
        "नमस्ते दुनिया",
        "你好，世界",
        "🙂🚀🧠",
        "café — naïve",
    ],
)
def test_encode_decode_round_trip(text: str) -> None:
    assert decode_bytes(encode_text(text)) == text


def test_encoded_values_are_raw_utf8_bytes() -> None:
    text = "A🙂"
    encoded = encode_text(text)

    assert encoded == list(text.encode("utf-8"))
    assert all(0 <= value <= 255 for value in encoded)


def test_empty_text_encodes_to_empty_sequence() -> None:
    assert encode_text("") == []


def test_decode_rejects_out_of_range_byte_ids() -> None:
    with pytest.raises(ValueError, match="between 0 and 255"):
        decode_bytes([256])


def test_decode_is_strict_by_default() -> None:
    with pytest.raises(UnicodeDecodeError):
        decode_bytes([0xFF])


def test_decode_can_replace_malformed_utf8_when_requested() -> None:
    assert decode_bytes([0xFF], errors="replace") == "�"
