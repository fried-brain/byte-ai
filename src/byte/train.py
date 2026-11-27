"""Tiny, reproducible BYTE training and evaluation utilities."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml

from byte.data import encode_corpus, make_next_byte_batches, split_byte_ids
from byte.model import BYTEConfig
from byte.baselines import build_model


def train_step(
    model: BYTEModel,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    """Perform one optimizer step on an aligned raw-byte batch."""
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    target_ids = batch.get("target_ids")
    if target_ids is not None:
        target_ids = target_ids.to(device)
        if target_ids.shape != input_ids[:, 1:].shape or not torch.equal(
            target_ids, input_ids[:, 1:]
        ):
            raise ValueError("target_ids must equal input_ids[:, 1:]")

    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(input_ids, attention_mask=attention_mask)
    if not torch.isfinite(output.total_loss):
        raise FloatingPointError("BYTE produced a non-finite training loss")

    output.total_loss.backward()
    for parameter in model.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("BYTE produced a non-finite gradient")
    optimizer.step()

    valid_gates = output.gates[output.valid_blocks]
    mean_active_slots = (
        float(valid_gates.sum(dim=-1).mean().detach().cpu())
        if valid_gates.numel()
        else 0.0
    )
    prediction_loss = float(output.prediction_loss.detach().cpu())
    return {
        "total_loss": float(output.total_loss.detach().cpu()),
        "prediction_loss": prediction_loss,
        "budget_loss": float(output.budget_loss.detach().cpu()),
        "bits_per_byte": prediction_loss / math.log(2.0),
        "mean_gate_sum": mean_active_slots,
    }


def evaluate_bits_per_byte(
    model: BYTEModel,
    batches: Iterable[dict[str, torch.Tensor]],
) -> float:
    """Compute token-count-weighted validation/test bits per byte."""
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    total_negative_log_likelihood = 0.0
    total_predictions = 0

    try:
        with torch.no_grad():
            for batch in batches:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                    prediction_mask = (
                        attention_mask[:, :-1] & attention_mask[:, 1:]
                    )
                else:
                    prediction_mask = torch.ones(
                        input_ids.shape[0],
                        input_ids.shape[1] - 1,
                        dtype=torch.bool,
                        device=device,
                    )

                count = int(prediction_mask.sum().item())
                if count == 0:
                    continue
                output = model(input_ids, attention_mask=attention_mask)
                total_negative_log_likelihood += (
                    float(output.prediction_loss.detach().cpu()) * count
                )
                total_predictions += count
    finally:
        model.train(was_training)

    if total_predictions == 0:
        raise ValueError("evaluation batches contain no valid next-byte targets")
    mean_nats = total_negative_log_likelihood / total_predictions
    return mean_nats / math.log(2.0)


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration and require a mapping at its root."""
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a YAML mapping")
    return config


def run_training(
    config: dict[str, Any],
    corpus_text: str,
    dataset_metadata: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Train a tiny model, evaluate validation data, and save a checkpoint."""
    project_config = config.get("project", {})
    training_config = config["training"]
    data_config = config.get("data", {})
    seed = int(project_config.get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model_values = dict(config["model"])
    model_variant = str(model_values.pop("variant", "byte"))
    model_config = BYTEConfig(**model_values)
    selected_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = build_model(model_variant, model_config).to(selected_device)
    model_config = model.config
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
    )

    byte_ids = encode_corpus(corpus_text)
    train_ids, validation_ids, test_ids = split_byte_ids(
        byte_ids,
        train_fraction=float(data_config.get("train_fraction", 0.9)),
        validation_fraction=float(data_config.get("validation_fraction", 0.05)),
    )

    block_size = model_config.block_size
    sequence_length = int(training_config["sequence_length"])
    batch_size = int(training_config["batch_size"])
    steps = int(training_config["steps"])
    validation_steps = int(training_config.get("validation_steps", 2))
    if steps <= 0 or validation_steps <= 0:
        raise ValueError("steps and validation_steps must be positive")

    training_batches = make_next_byte_batches(
        train_ids, block_size, sequence_length, batch_size, seed
    )
    last_metrics: dict[str, float] = {}
    for _ in range(steps):
        last_metrics = train_step(model, next(training_batches), optimizer)

    validation_batches = list(
        itertools.islice(
            make_next_byte_batches(
                validation_ids,
                block_size,
                sequence_length,
                batch_size,
                seed + 1,
            ),
            validation_steps,
        )
    )
    validation_bits_per_byte = evaluate_bits_per_byte(model, validation_batches)

    run_directory = Path(
        output_dir or training_config.get("output_dir", "runs/tiny")
    )
    run_directory.mkdir(parents=True, exist_ok=True)

    dataset_record = {
        **dataset_metadata,
        "sha256_utf8": hashlib.sha256(corpus_text.encode("utf-8")).hexdigest(),
        "byte_count": len(byte_ids),
        "split_byte_counts": {
            "train": len(train_ids),
            "validation": len(validation_ids),
            "test": len(test_ids),
        },
    }
    metadata: dict[str, Any] = {
        "project": project_config,
        "dataset": dataset_record,
        "model": asdict(model_config),
        "model_variant": model_variant,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training": {
            "steps": steps,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "learning_rate": float(training_config["learning_rate"]),
            "seed": seed,
            "device": str(selected_device),
            "hardware": (
                torch.cuda.get_device_name(selected_device)
                if selected_device.type == "cuda"
                else "CPU"
            ),
        },
        "results": {
            "last_train_step": last_metrics,
            "validation_bits_per_byte": validation_bits_per_byte,
        },
    }

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": asdict(model_config),
        "model_variant": model_variant,
        "metadata": metadata,
    }
    torch.save(checkpoint, run_directory / "checkpoint.pt")
    with (run_directory / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    return metadata


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the tiny BYTE research model")
    parser.add_argument("--config", required=True, help="Path to a YAML configuration")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use generated synthetic text; this is not a benchmark experiment",
    )
    args = parser.parse_args(argv)
    config = load_yaml_config(args.config)

    if args.smoke_test:
        corpus_text = (
            "BYTE predicts the next raw UTF-8 byte from preceding bytes. "
            "Adaptive compression is an experimental hypothesis. "
        ) * 500
        dataset_metadata = {
            "kind": "synthetic_smoke_test",
            "source": "deterministic text generated by byte.train",
            "license": "synthetic test fixture; not a benchmark corpus",
        }
    else:
        data_config = config.get("data", {})
        path = data_config.get("path")
        source = data_config.get("source")
        license_name = data_config.get("license")
        if not path or not source or not license_name:
            raise ValueError(
                "real training requires data.path, data.source, and data.license"
            )
        corpus_path = Path(path)
        corpus_text = corpus_path.read_text(encoding="utf-8")
        dataset_metadata = {
            "kind": "user_provided_text_file",
            "source": source,
            "license": license_name,
            "path": str(corpus_path),
        }

    metadata = run_training(config, corpus_text, dataset_metadata)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
