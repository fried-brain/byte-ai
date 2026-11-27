import torch
import torch.nn.functional as F

from byte.model import BYTEConfig, BYTEModel, sinusoidal_position_encoding


def tiny_config(**overrides):
    values = dict(
        d_model=16,
        block_size=4,
        local_layers=1,
        global_layers=1,
        nhead=4,
        num_slots=4,
        dropout=0.0,
        budget_weight=0.05,
    )
    values.update(overrides)
    return BYTEConfig(**values)


def test_model_returns_next_byte_logits_and_training_losses() -> None:
    model = BYTEModel(tiny_config())
    input_ids = torch.randint(0, 256, (2, 8))

    output = model(input_ids)

    assert output.logits.shape == (2, 8, 256)
    assert output.gates.shape == (2, 2, 4)
    assert output.valid_blocks.shape == (2, 2)
    assert torch.isfinite(output.logits).all()
    assert torch.isfinite(output.total_loss)


def test_prediction_loss_uses_next_byte_targets_and_valid_positions_only() -> None:
    model = BYTEModel(tiny_config())
    input_ids = torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17]])
    attention_mask = torch.tensor([[True, True, True, True, True, False, False, False]])

    output = model(input_ids, attention_mask=attention_mask)

    manual_mask = attention_mask[:, :-1] & attention_mask[:, 1:]
    manual = F.cross_entropy(
        output.logits[:, :-1, :][manual_mask],
        input_ids[:, 1:][manual_mask],
    )
    torch.testing.assert_close(output.prediction_loss, manual)
    assert output.valid_blocks.tolist() == [[True, True]]


def test_padding_does_not_contribute_to_budget_or_prediction_loss() -> None:
    model = BYTEModel(tiny_config())
    input_ids = torch.randint(0, 256, (2, 8))
    attention_mask = torch.tensor(
        [
            [True, True, False, False, False, False, False, False],
            [True, True, True, True, True, True, True, True],
        ]
    )

    output = model(input_ids, attention_mask=attention_mask)

    assert output.valid_blocks.tolist() == [[True, False], [True, True]]
    assert torch.count_nonzero(output.gates[0, 1]) == 0
    assert torch.isfinite(output.prediction_loss)
    assert torch.isfinite(output.budget_loss)


def test_earlier_block_predictions_do_not_depend_on_later_blocks() -> None:
    torch.manual_seed(31)
    model = BYTEModel(tiny_config()).eval()
    original = torch.randint(0, 256, (1, 12))
    changed = original.clone()
    changed[:, 4:] = (changed[:, 4:] + 73) % 256

    with torch.no_grad():
        output_a = model(original)
        output_b = model(changed)

    torch.testing.assert_close(output_a.logits[:, :4], output_b.logits[:, :4])


def test_earlier_positions_do_not_depend_on_later_bytes_in_same_block() -> None:
    torch.manual_seed(37)
    model = BYTEModel(tiny_config()).eval()
    original = torch.randint(0, 256, (1, 8))
    changed = original.clone()
    changed[:, 2:4] = (changed[:, 2:4] + 91) % 256

    with torch.no_grad():
        output_a = model(original)
        output_b = model(changed)

    torch.testing.assert_close(output_a.logits[:, :2], output_b.logits[:, :2])


def test_current_block_compression_is_not_used_for_its_own_predictions(monkeypatch) -> None:
    torch.manual_seed(41)
    model = BYTEModel(tiny_config()).eval()
    input_ids = torch.randint(0, 256, (1, 8))
    original_forward = model.compressor.forward

    def altered_compression(*args, **kwargs):
        result = original_forward(*args, **kwargs)
        result.latents = result.latents * 0.0
        return result

    with torch.no_grad():
        baseline = model(input_ids)
        monkeypatch.setattr(model.compressor, "forward", altered_compression)
        altered = model(input_ids)

    torch.testing.assert_close(baseline.logits[:, :4], altered.logits[:, :4])


def test_non_multiple_sequence_length_is_supported() -> None:
    model = BYTEModel(tiny_config()).eval()
    input_ids = torch.randint(0, 256, (2, 7))

    with torch.no_grad():
        output = model(input_ids)

    assert output.logits.shape == (2, 7, 256)
    assert output.gates.shape == (2, 2, 4)


def test_total_loss_combines_prediction_and_weighted_budget_losses() -> None:
    config = tiny_config(budget_weight=0.125)
    model = BYTEModel(config)
    output = model(torch.randint(0, 256, (2, 8)))

    expected = output.prediction_loss + config.budget_weight * output.budget_loss
    torch.testing.assert_close(output.total_loss, expected)


def test_sinusoidal_positions_support_long_sequences_deterministically() -> None:
    first = sinusoidal_position_encoding(2048, 16)
    second = sinusoidal_position_encoding(2048, 16)

    assert first.shape == (1, 2048, 16)
    torch.testing.assert_close(first, second)
    assert not torch.equal(first[:, 0], first[:, 1024])


def test_model_backward_produces_finite_gradients() -> None:
    model = BYTEModel(tiny_config())
    output = model(torch.randint(0, 256, (2, 8)))
    output.total_loss.backward()

    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)


def test_predict_next_uses_each_samples_last_valid_position() -> None:
    model = BYTEModel(tiny_config()).eval()
    input_ids = torch.randint(0, 256, (2, 8))
    attention_mask = torch.tensor(
        [
            [True, True, True, False, False, False, False, False],
            [True, True, True, True, True, True, True, True],
        ]
    )

    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask)
        next_logits = model.predict_next(input_ids, attention_mask=attention_mask)

    assert next_logits.shape == (2, 256)
    torch.testing.assert_close(next_logits[0], output.logits[0, 2])
    torch.testing.assert_close(next_logits[1], output.logits[1, 7])


def test_single_byte_input_has_zero_prediction_loss() -> None:
    model = BYTEModel(tiny_config())
    output = model(torch.tensor([[65]]))
    assert output.logits.shape == (1, 1, 256)
    assert output.prediction_loss.item() == 0.0


def test_eval_global_encoder_physically_skips_inactive_latent_slots(monkeypatch) -> None:
    model = BYTEModel(tiny_config()).eval()
    original_forward = model.compressor.forward
    sequence_lengths = []

    def controlled_gates(*args, **kwargs):
        result = original_forward(*args, **kwargs)
        result.gates = torch.tensor(
            [[[0.9, 0.1, 0.8, 0.1]]],
            device=result.gates.device,
            dtype=result.gates.dtype,
        )
        return result

    def record_sequence_length(module, args):
        sequence_lengths.append(args[0].shape[1])

    monkeypatch.setattr(model.compressor, "forward", controlled_gates)
    hook = model.global_encoder.register_forward_pre_hook(record_sequence_length)
    with torch.no_grad():
        model(torch.randint(0, 256, (1, 8)))
    hook.remove()

    # Block 0 uses BOS + query; block 1 adds only the two active slots.
    assert sequence_lengths == [2, 4]


def test_training_global_encoder_keeps_all_differentiable_slots(monkeypatch) -> None:
    model = BYTEModel(tiny_config()).train()
    original_forward = model.compressor.forward
    sequence_lengths = []

    def controlled_gates(*args, **kwargs):
        result = original_forward(*args, **kwargs)
        result.gates = torch.tensor(
            [[[0.9, 0.1, 0.8, 0.1]]],
            device=result.gates.device,
            dtype=result.gates.dtype,
        )
        return result

    def record_sequence_length(module, args):
        sequence_lengths.append(args[0].shape[1])

    monkeypatch.setattr(model.compressor, "forward", controlled_gates)
    hook = model.global_encoder.register_forward_pre_hook(record_sequence_length)
    model(torch.randint(0, 256, (1, 8)))
    hook.remove()

    # Training keeps all four slots and lets the gate values learn by gradient.
    assert sequence_lengths == [2, 6]
