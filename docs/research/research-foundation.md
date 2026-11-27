# BYTE research foundation

## Mission

Investigate whether a compact neural model can learn useful linguistic regularities directly from raw UTF-8 bytes, using learned hierarchical representations and adaptive compression.

## What BYTE is not claiming

Byte-level language modeling is established research. BYTE does not claim to invent raw-byte input, hierarchical byte processing, or adaptive grouping.

## Relevant prior work

| Work | Relevance |
|---|---|
| ByT5 (2022) | Text-to-text modeling directly over UTF-8 bytes |
| CANINE (2022) | Tokenization-free character encoder with downsampling; character-level, not raw UTF-8 byte-level |
| MEGABYTE (2023) | Multiscale local/global processing for byte sequences |
| BLT (2025) | Dynamic byte patches informed by next-byte entropy |

## Research hypotheses

- H1: A model trained directly on bytes can learn predictive linguistic regularities without a subword vocabulary.
- H2: Learned grouping can reduce effective sequence length while preserving next-byte prediction quality.
- H3: Local/global processing may improve the quality-cost trade-off.
- H4: Predictive representations may generalize beyond memorized training examples.
- H5: Compression utility depends on balancing sequence reduction against information loss.

These are hypotheses to test, not established outcomes.

## Baselines

1. Standard byte-level Transformer.
2. Non-hierarchical byte model.
3. BYTE hybrid model with its proposed learned adaptive compression mechanism.

Use identical dataset partitions and comparable parameter budgets where practical.

## Evaluation protocol

Record:
- Bits per byte (primary language-modeling metric).
- Next-byte accuracy (secondary metric).
- Compression ratio / effective sequence-length reduction.
- Peak memory, throughput, and wall-clock training/inference cost.
- Parameter count and compute settings.

Record dataset source, license, version, checksum, preprocessing, and split boundaries. Do not tune on the test set.

## Initial scope

The first milestone is only a correct UTF-8 byte interface. No claim about intelligence, reasoning, or model capability follows from implementing a codec.
