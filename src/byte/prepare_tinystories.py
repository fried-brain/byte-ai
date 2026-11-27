"""Prepare a small, revision-pinned TinyStories training subset."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Iterable

DATASET_ID = "roneneldan/TinyStories"
DATASET_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
DATASET_LICENSE = "CDLA-Sharing-1.0"


def write_story_subset(
    records: Iterable[dict[str, Any]],
    output_path: str | Path,
    *,
    max_stories: int,
    seed: int,
    dataset_id: str,
    revision: str,
    split: str,
    license_name: str,
) -> dict[str, Any]:
    """Write a deterministic prefix of a prepared record stream and its manifest."""
    if max_stories <= 0:
        raise ValueError("max_stories must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not dataset_id or not revision or not split or not license_name:
        raise ValueError("dataset_id, revision, split, and license_name are required")

    selected: list[str] = []
    for record in itertools.islice(records, max_stories):
        text = record["text"].strip()
        if text:
            selected.append(text)
    if not selected:
        raise ValueError("the selected dataset stream contained no non-empty stories")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    content = ("\n\n".join(selected) + "\n").encode("utf-8")
    output.write_bytes(content)

    manifest: dict[str, Any] = {
        "dataset_id": dataset_id,
        "revision": revision,
        "split": split,
        "license": license_name,
        "selection": {
            "method": "streaming shuffle followed by a fixed-size sample",
            "seed": seed,
            "max_stories": max_stories,
        },
        "story_count": len(selected),
        "byte_count": len(content),
        "sha256_utf8": hashlib.sha256(content).hexdigest(),
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Stream a small reproducible TinyStories training subset"
    )
    parser.add_argument(
        "--output",
        default="data/tinystories-pilot.txt",
        help="Output UTF-8 text file (default: data/tinystories-pilot.txt)",
    )
    parser.add_argument(
        "--stories",
        type=int,
        default=50_000,
        help="Number of training stories to sample (default: 50000)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    args = parser.parse_args(argv)

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            'Install the optional data dependencies with: pip install -e ".[data]"'
        ) from error

    stream = load_dataset(
        DATASET_ID,
        split="train",
        streaming=True,
        revision=DATASET_REVISION,
    )
    stream = stream.shuffle(
        seed=args.seed,
        buffer_size=args.shuffle_buffer,
    )
    manifest = write_story_subset(
        stream,
        args.output,
        max_stories=args.stories,
        seed=args.seed,
        dataset_id=DATASET_ID,
        revision=DATASET_REVISION,
        split="train",
        license_name=DATASET_LICENSE,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
