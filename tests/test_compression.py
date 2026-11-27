import torch

from byte.compression import (
    AdaptiveCompressor,
    compression_budget_loss,
    pack_active_latents,
)


def test_compressor_returns_four_latents_and_gates_per_block() -> None:
    compressor = AdaptiveCompressor(d_model=16, num_slots=4, nhead=4, dropout=0.0)
    states = torch.randn(2, 3, 8, 16)

    output = compressor(states)

    assert output.latents.shape == (2, 3, 4, 16)
    assert output.gates.shape == (2, 3, 4)
    assert torch.isfinite(output.latents).all()
    assert torch.isfinite(output.gates).all()
    assert torch.all((output.gates >= 0) & (output.gates <= 1))


def test_padded_bytes_do_not_influence_valid_block_compression() -> None:
    torch.manual_seed(23)
    compressor = AdaptiveCompressor(d_model=16, num_slots=4, nhead=4, dropout=0.0)
    compressor.eval()
    states_a = torch.randn(1, 1, 5, 16)
    states_b = states_a.clone()
    states_b[:, :, 1, :] = 1000.0
    valid = torch.tensor([[[True, False, True, True, True]]])

    with torch.no_grad():
        output_a = compressor(states_a, valid_mask=valid)
        output_b = compressor(states_b, valid_mask=valid)

    torch.testing.assert_close(output_a.latents, output_b.latents)
    torch.testing.assert_close(output_a.gates, output_b.gates)


def test_budget_loss_is_zero_at_target_and_positive_off_target() -> None:
    gates = torch.tensor(
        [
            [[1.0, 1.0, 0.0, 0.0]],
            [[0.5, 0.5, 0.5, 0.5]],
        ]
    )
    valid_blocks = torch.tensor([[True], [True]])

    at_target = compression_budget_loss(gates, valid_blocks, target_active_slots=2.0)
    below_target = compression_budget_loss(
        torch.zeros_like(gates), valid_blocks, target_active_slots=2.0
    )

    assert at_target.item() == 0.0
    assert below_target.item() > 0.0


def test_budget_loss_ignores_invalid_blocks_and_is_zero_for_empty_batch() -> None:
    gates = torch.tensor([[[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]])
    valid_blocks = torch.tensor([[True, False]])
    loss = compression_budget_loss(gates, valid_blocks, target_active_slots=2.0)
    assert loss.item() == 0.0

    empty_loss = compression_budget_loss(
        gates, torch.zeros_like(valid_blocks), target_active_slots=2.0
    )
    assert empty_loss.item() == 0.0


def test_compressor_has_finite_gradients_for_queries_and_gate_network() -> None:
    torch.manual_seed(29)
    compressor = AdaptiveCompressor(d_model=16, num_slots=4, nhead=4, dropout=0.0)
    states = torch.randn(2, 2, 6, 16)
    output = compressor(states)
    loss = output.latents.square().mean() + output.gates.mean()
    loss.backward()

    assert compressor.latent_queries.grad is not None
    assert torch.isfinite(compressor.latent_queries.grad).all()
    gate_gradients = [p.grad for p in compressor.gate_network.parameters()]
    assert all(grad is not None and torch.isfinite(grad).all() for grad in gate_gradients)


def test_threshold_packing_preserves_batch_block_and_slot_order() -> None:
    latents = torch.arange(2 * 2 * 4 * 3, dtype=torch.float32).reshape(2, 2, 4, 3)
    gates = torch.tensor(
        [
            [[0.9, 0.1, 0.8, 0.2], [0.1, 0.7, 0.6, 0.4]],
            [[0.5, 0.5, 0.49, 0.0], [0.0, 0.0, 0.0, 0.0]],
        ]
    )
    valid_blocks = torch.tensor([[True, True], [True, True]])

    packed, block_indices, slot_indices = pack_active_latents(
        latents, gates, valid_blocks, threshold=0.5
    )

    assert packed.shape == (7, 3)
    assert block_indices.tolist() == [
        [0, 0], [0, 0], [0, 1], [0, 1], [1, 0], [1, 0]
    ][:packed.shape[0]] if False else block_indices.tolist() == [
        [0, 0], [0, 0], [0, 1], [0, 1], [1, 0], [1, 0]
    ]
    assert slot_indices.tolist() == [0, 2, 1, 2, 0, 1]
    torch.testing.assert_close(packed, torch.stack([
        latents[0, 0, 0], latents[0, 0, 2],
        latents[0, 1, 1], latents[0, 1, 2],
        latents[1, 0, 0], latents[1, 0, 1],
    ]))


def test_all_inactive_inference_returns_empty_packed_tensor() -> None:
    latents = torch.randn(2, 3, 4, 8)
    gates = torch.zeros(2, 3, 4)
    valid_blocks = torch.ones(2, 3, dtype=torch.bool)

    packed, block_indices, slot_indices = pack_active_latents(
        latents, gates, valid_blocks, threshold=0.5
    )

    assert packed.shape == (0, 8)
    assert block_indices.shape == (0, 2)
    assert slot_indices.shape == (0,)


def test_compressor_inference_output_contains_packed_slots() -> None:
    compressor = AdaptiveCompressor(d_model=16, num_slots=4, nhead=4, dropout=0.0)
    states = torch.randn(1, 2, 5, 16)
    output = compressor(states, inference=True, threshold=0.0)

    assert output.packed_latents is not None
    assert output.block_indices is not None
    assert output.slot_indices is not None
    assert output.packed_latents.shape[1] == 16
    assert output.block_indices.shape == (output.packed_latents.shape[0], 2)
