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
pip install -e ".[dev,train,data]"
```

Then run:

```bash
python -m byte.train --config configs/tiny.yaml --smoke-test
```

The smoke test overrides the training budget to two steps and uses generated synthetic text. It does not download or use TinyStories.

The smoke test writes a checkpoint and metadata under `runs/tiny/`. It uses generated synthetic text and must not be treated as benchmark evidence.

## Prepare the first public pilot corpus

BYTE uses a small, revision-pinned sample from the English TinyStories training split for its first real-data pilot. TinyStories is a synthetic short-story dataset generated with GPT-3.5/GPT-4; it is useful for a tiny-model pipeline but is not a broad natural-language benchmark.

Prepare 50,000 shuffled training stories:

```bash
python -m byte.prepare_tinystories --output data/tinystories-pilot.txt --stories 50000 --seed 42
```

The preparation command streams the pinned dataset revision, writes the UTF-8 corpus and creates a manifest containing its revision, license, story count, byte count, and SHA-256 checksum. The `data/` directory is ignored by Git; do not commit the raw corpus.

The dataset card identifies English text and the CDLA-Sharing-1.0 agreement. If you redistribute the raw subset or modified data, preserve attribution and publish it under the agreement. The agreement treats computational outputs as “Results” only when they do not contain more than a de minimis portion of the source data; it places no obligations on qualifying Results. We therefore make no blanket legal claim about weights that might memorize substantial story text. See the [dataset card](https://huggingface.co/datasets/roneneldan/TinyStories), [TinyStories paper](https://arxiv.org/abs/2305.07759), and [CDLA Sharing 1.0](https://cdla.dev/sharing-1-0/).

## Real corpus training

The default configuration points to the generated pilot corpus and runs 1,000 tiny training steps. Adjust the training budget only after the smoke test succeeds.

The pipeline records the corpus SHA-256, byte count, split sizes, model variant, parameter count, model configuration, seed, software/hardware details, and validation bits per byte. The initial pilot split is a deterministic contiguous 90/5/5 split of the sampled training stories; it is not the dataset’s official validation/test benchmark.

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
