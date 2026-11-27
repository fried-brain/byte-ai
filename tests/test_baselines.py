import torch

from byte.baselines import (
    FixedSlotBYTEModel,
    NonHierarchicalByteGRU,
    StandardByteTransformer,
    build_model,
)
from byte.model import BYTEConfig


def tiny_config() -> BYTEConfig:
    return BYTEConfig(
        d_model=16,
        block_size=4,
        local_layers=1,
        global_layers=1,
        nhead=4,
        num_slots=4,
        dropout=0.0,
    )


def test_standard_byte_transformer_predicts_next_byte_causally() -> None:
    torch.manual_seed(51)
    model = StandardByteTransformer(tiny_config()).eval()
    original = torch.randint(0, 256, (1, 8))
    changed = original.clone()
    changed[:, 4:] = (changed[:, 4:] + 17) % 256

    with torch.no_grad():
        output_a = model(original)
        output_b = model(changed)

    assert output_a.logits.shape == (1, 8, 256)
    torch.testing.assert_close(output_a.logits[:, :4], output_b.logits[:, :4])


def test_non_hierarchical_byte_gru_predicts_causally() -> None:
    torch.manual_seed(53)
    model = NonHierarchicalByteGRU(tiny_config()).eval()
    original = torch.randint(0, 256, (1, 8))
    changed = original.clone()
    changed[:, 4:] = (changed[:, 4:] + 31) % 256

    with torch.no_grad():
        output_a = model(original)
        output_b = model(changed)

    assert output_a.logits.shape == (1, 8, 256)
    torch.testing.assert_close(output_a.logits[:, :4], output_b.logits[:, :4])


def test_fixed_hierarchical_baseline_always_uses_two_slots() -> None:
    model = FixedSlotBYTEModel(tiny_config())
    output = model(torch.randint(0, 256, (2, 8)))

    assert model.config.num_slots == 2
    assert output.gates.shape == (2, 2, 2)
    assert torch.all(output.gates[output.valid_blocks] == 1.0)
    assert output.budget_loss.item() == 0.0


def test_model_factory_exposes_all_four_experiment_variants() -> None:
    variants = {
        "byte": build_model("byte", tiny_config()),
        "standard_transformer": build_model("standard_transformer", tiny_config()),
        "byte_gru": build_model("byte_gru", tiny_config()),
        "fixed_hierarchical": build_model("fixed_hierarchical", tiny_config()),
    }

    assert isinstance(variants["standard_transformer"], StandardByteTransformer)
    assert isinstance(variants["byte_gru"], NonHierarchicalByteGRU)
    assert isinstance(variants["fixed_hierarchical"], FixedSlotBYTEModel)
    for model in variants.values():
        output = model(torch.randint(0, 256, (1, 8)))
        assert output.logits.shape == (1, 8, 256)
