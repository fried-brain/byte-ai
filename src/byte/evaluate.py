"""Evaluate a trained BYTE checkpoint on the held-out test split."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import torch

from byte.data import encode_corpus, make_next_byte_batches, split_byte_ids
from byte.model import BYTEConfig, BYTEModel
from byte.train import evaluate_bits_per_byte, load_yaml_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a BYTE checkpoint")
    parser.add_argument("--config", required=True, help="Path to the training YAML configuration")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.pt")
    args = parser.parse_args(argv)

    config = load_yaml_config(args.config)
    data_config = config.get("data", {})
    path = data_config.get("path")
    source = data_config.get("source")
    license_name = data_config.get("license")
    if not path or not source or not license_name:
        raise ValueError(
            "test evaluation requires data.path, data.source, and data.license"
        )

    corpus_path = Path(path)
    corpus_text = corpus_path.read_text(encoding="utf-8")
    corpus_hash = hashlib.sha256(corpus_text.encode("utf-8")).hexdigest()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    trained_hash = checkpoint["metadata"]["dataset"]["sha256_utf8"]
    if corpus_hash != trained_hash:
        raise ValueError(
            "evaluation corpus checksum differs from the checkpoint's training corpus"
        )

    byte_ids = encode_corpus(corpus_text)
    _, _, test_ids = split_byte_ids(
        byte_ids,
        train_fraction=float(data_config.get("train_fraction", 0.9)),
        validation_fraction=float(data_config.get("validation_fraction", 0.05)),
    )

    training_config = config["training"]
    test_steps = int(training_config.get("test_steps", 2))
    if test_steps <= 0:
        raise ValueError("test_steps must be positive")
    batches = list(
        itertools.islice(
            make_next_byte_batches(
                test_ids,
                int(checkpoint["model_config"]["block_size"]),
                int(training_config["sequence_length"]),
                int(training_config["batch_size"]),
                int(config.get("project", {}).get("seed", 42)) + 2,
            ),
            test_steps,
        )
    )

    model_config = BYTEConfig(**checkpoint["model_config"])
    model = BYTEModel(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    bits_per_byte = evaluate_bits_per_byte(model, batches)

    result = {
        "metric": "test_bits_per_byte",
        "value": bits_per_byte,
        "dataset_source": source,
        "dataset_license": license_name,
        "dataset_sha256_utf8": corpus_hash,
        "test_bytes": len(test_ids),
        "checkpoint": str(Path(args.checkpoint)),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
