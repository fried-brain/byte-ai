# BYTE

**BYTE** is a small, reproducible research project exploring text modeling directly from raw UTF-8 bytes.

## Research question

Can a model learn useful linguistic structure and predictive capabilities directly from raw bytes, using learned hierarchical representations and adaptive compression rather than a predefined subword vocabulary?

Byte-level modeling, hierarchical processing, and adaptive grouping already have substantial prior work. BYTE's contribution must therefore be established through a clearly specified mechanism and controlled experiments—not by claiming that byte-level modeling itself is new.

## Current status

**Phase 1 — tiny neural foundation implemented; benchmark training not yet established.**

The repository currently includes:
- Raw UTF-8 byte encoding and decoding (IDs 0–255).
- Byte embeddings and a local causal Transformer.
- Learned four-slot adaptive compression with sigmoid gates and a compression-budget loss.
- A blockwise causal BYTE model with a global encoder over individual prior-block latent slots.
- Three controlled baselines: a full-context byte Transformer, a non-hierarchical byte GRU, and fixed two-slot hierarchical compression.
- Inference-time physical selection of active latent slots using a configurable threshold.
- Deterministic corpus splitting and next-byte batch generation.
- Tiny training and bits-per-byte evaluation utilities.

The model has not yet been trained on a selected public benchmark corpus. The synthetic smoke test verifies pipeline mechanics only; it is not a research result.

## Run a smoke test

Install the training dependencies and development tools:

```bash
pip install -e ".[dev,train]"
```

Then run:

```bash
python -m byte.train --config configs/tiny.yaml --smoke-test
```

The smoke test writes a checkpoint and metadata under `runs/tiny/`. It uses generated synthetic text and must not be treated as benchmark evidence.

## Real corpus training

Before real training, edit `configs/tiny.yaml` and supply:
- `data.path`: path to a UTF-8 text corpus.
- `data.source`: source URL or precise source description.
- `data.license`: verified license.

The pipeline records the corpus SHA-256, byte count, split sizes, model configuration, seed, software/hardware details, and validation bits per byte.

After training, evaluate the held-out test split with:

```bash
python -m byte.evaluate --config configs/tiny.yaml --checkpoint runs/tiny/checkpoint.pt
```

The evaluation command verifies that the corpus checksum matches the checkpoint metadata.

## Research baselines

BYTE is intended to be compared against:
1. A standard byte-level Transformer.
2. A non-hierarchical byte model.
3. Fixed-size hierarchical compression.
4. BYTE adaptive compression.

All four model variants share the same batch generator, contiguous data splits, training step, checkpoint format, and bits-per-byte evaluator. Benchmark comparisons have not yet been run.

## Evaluation

Planned metrics include bits per byte, next-byte accuracy, effective sequence-length reduction, memory use, throughput, and training/inference cost. Dataset provenance, licensing, checksums, and split boundaries must be recorded. The test set must not be used for tuning.

## Prior work

- [ByT5 (2022)](https://aclanthology.org/2022.tacl-1.17/)
- [CANINE (2022)](https://aclanthology.org/2022.tacl-1.5/)
- [MEGABYTE (2023)](https://papers.neurips.cc/paper_files/paper/2023/hash/f8f78f8043f35890181a824e53a57134-Abstract-Conference.html)
- [Byte Latent Transformer / BLT (2025)](https://aclanthology.org/2025.acl-long.453/)

## License

Apache-2.0. See [LICENSE](LICENSE).
