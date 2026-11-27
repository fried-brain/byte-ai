import pytest
import torch

from byte.modeling import ByteEmbedding, LocalCausalEncoder


def test_byte_embedding_preserves_batch_and_sequence_dimensions() -> None:
    embedding = ByteEmbedding(vocab_size=256, d_model=128)
    byte_ids = torch.tensor([[0, 65, 255], [1, 2, 3]])
    result = embedding(byte_ids)
    assert result.shape == (2, 3, 128)


def test_byte_embedding_accepts_highest_raw_byte_id() -> None:
    embedding = ByteEmbedding(vocab_size=256, d_model=16)
    result = embedding(torch.tensor([[255]]))
    assert result.shape == (1, 1, 16)


def test_byte_embedding_rejects_out_of_range_byte_id() -> None:
    embedding = ByteEmbedding(vocab_size=256, d_model=16)
    with pytest.raises(IndexError):
        embedding(torch.tensor([[256]]))


def test_local_encoder_cannot_see_later_bytes_in_same_block() -> None:
    torch.manual_seed(7)
    encoder = LocalCausalEncoder(d_model=16, nhead=4, num_layers=2, dropout=0.0)
    encoder.eval()
    first = torch.randn(1, 1, 6, 16)
    changed_future = first.clone()
    changed_future[:, :, 4:, :] += 100.0
    with torch.no_grad():
        output_a = encoder(first)
        output_b = encoder(changed_future)
    torch.testing.assert_close(output_a[:, :, :4, :], output_b[:, :, :4, :])


def test_local_encoder_ignores_padded_positions_as_attention_keys() -> None:
    torch.manual_seed(11)
    encoder = LocalCausalEncoder(d_model=16, nhead=4, num_layers=2, dropout=0.0)
    encoder.eval()
    values_a = torch.randn(1, 1, 5, 16)
    values_b = values_a.clone()
    values_b[:, :, 1, :] = 999.0
    values_b[:, :, 4, :] = -999.0
    valid = torch.tensor([[[True, False, True, True, False]]])
    with torch.no_grad():
        output_a = encoder(values_a, valid_mask=valid)
        output_b = encoder(values_b, valid_mask=valid)
    torch.testing.assert_close(output_a[:, :, 2:4, :], output_b[:, :, 2:4, :])
    assert torch.count_nonzero(output_a[:, :, 1, :]) == 0
    assert torch.count_nonzero(output_a[:, :, 4, :]) == 0
    assert torch.isfinite(output_a).all()


def test_local_encoder_handles_a_fully_padded_block() -> None:
    encoder = LocalCausalEncoder(d_model=16, nhead=4, num_layers=1, dropout=0.0)
    values = torch.randn(1, 1, 4, 16)
    valid = torch.zeros((1, 1, 4), dtype=torch.bool)
    output = encoder(values, valid_mask=valid)
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output) == 0
