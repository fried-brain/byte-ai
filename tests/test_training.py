import math

import pytest
import torch

from byte.data import make_next_byte_batches
from byte.model import BYTEConfig, BYTEModel
from byte.train import evaluate_bits_per_byte, run_training, train_step


def tiny_config():
    return {
        "project": {"name": "BYTE", "version": "0.2.0", "seed": 17},
        "model": {
            "vocab_size": 256,
            "d_model": 16,
            "block_size": 4,
            "local_layers": 1,
            "global_layers": 1,
            "nhead": 4,
            "num_slots": 4,
            "target_active_slots": 2.0,
            "gate_threshold": 0.5,
            "budget_weight": 0.05,
            "dropout": 0.0,
        },
        "training": {
            "steps": 1,
            "batch_size": 2,
            "sequence_length": 8,
            "learning_rate": 0.001,
            "validation_steps": 1,
            "output_dir": "runs/test",
        },
        "data": {
            "train_fraction": 0.8,
            "validation_fraction": 0.1,
        },
    }


def test_train_step_updates_parameters_and_returns_finite_metrics() -> None:
    model = BYTEModel(BYTEConfig(d_model=16, block_size=4, local_layers=1, global_layers=1, nhead=4, dropout=0.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    batch = next(make_next_byte_batches([index % 256 for index in range(100)], 4, 8, 2, seed=9))
    before = model.byte_embedding.embedding.weight.detach().clone()

    metrics = train_step(model, batch, optimizer)

    assert torch.isfinite(torch.tensor(list(metrics.values()))).all()
    assert not torch.equal(before, model.byte_embedding.embedding.weight.detach())
    assert {"total_loss", "prediction_loss", "budget_loss", "bits_per_byte"} <= set(metrics)


def test_train_step_rejects_misaligned_targets() -> None:
    model = BYTEModel(BYTEConfig(d_model=16, block_size=4, local_layers=1, global_layers=1, nhead=4, dropout=0.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    batch = next(make_next_byte_batches([index % 256 for index in range(100)], 4, 8, 2, seed=3))
    batch["target_ids"] = torch.zeros_like(batch["target_ids"])
    with pytest.raises(ValueError, match="target_ids must equal"):
        train_step(model, batch, optimizer)


def test_evaluation_bits_per_byte_matches_weighted_manual_calculation() -> None:
    model = BYTEModel(BYTEConfig(d_model=16, block_size=4, local_layers=1, global_layers=1, nhead=4, dropout=0.0))
    corpus = [index % 256 for index in range(200)]
    batches = list(make_next_byte_batches(corpus, 4, 8, 2, seed=4) for _ in range(2))
    batches = [next(batch_iter) for batch_iter in batches]

    model.eval()
    with torch.no_grad():
        outputs = [model(batch["input_ids"], batch["attention_mask"]) for batch in batches]
    counts = [
        int((batch["attention_mask"][:, :-1] & batch["attention_mask"][:, 1:]).sum())
        for batch in batches
    ]
    expected_nats = sum(out.prediction_loss.item() * count for out, count in zip(outputs, counts)) / sum(counts)
    expected = expected_nats / math.log(2.0)

    assert evaluate_bits_per_byte(model, batches) == pytest.approx(expected, rel=1e-6)


def test_smoke_training_writes_checkpoint_and_metadata(tmp_path) -> None:
    config = tiny_config()
    corpus = ("BYTE predicts raw UTF-8 bytes. " * 100)
    metadata = {
        "kind": "synthetic_smoke_test",
        "source": "generated test fixture",
        "license": "not a benchmark corpus",
    }

    result = run_training(
        config,
        corpus,
        metadata,
        output_dir=tmp_path,
        device="cpu",
    )

    assert (tmp_path / "checkpoint.pt").is_file()
    assert (tmp_path / "metadata.json").is_file()
    assert result["training"]["steps"] == 1
    assert result["dataset"]["kind"] == "synthetic_smoke_test"
    assert math.isfinite(result["results"]["validation_bits_per_byte"])


@pytest.mark.parametrize(
    "variant",
    ["standard_transformer", "byte_gru", "fixed_hierarchical"],
)
def test_training_pipeline_supports_each_baseline_variant(tmp_path, variant) -> None:
    config = tiny_config()
    config["model"]["variant"] = variant
    corpus = "BYTE baseline comparison fixture. " * 100
    metadata = {
        "kind": "synthetic_smoke_test",
        "source": "generated test fixture",
        "license": "not a benchmark corpus",
    }

    result = run_training(
        config,
        corpus,
        metadata,
        output_dir=tmp_path / variant,
        device="cpu",
    )

    assert result["model_variant"] == variant
    assert result["parameter_count"] > 0
    assert (tmp_path / variant / "checkpoint.pt").is_file()
