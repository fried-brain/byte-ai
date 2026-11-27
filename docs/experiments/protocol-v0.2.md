# BYTE v0.2.0 — Experiment Protocol

## Status

The initial training and evaluation pipeline supports synthetic smoke tests and user-supplied UTF-8 text files. No public benchmark corpus has yet been selected or approved for BYTE results.

A smoke test checks that batching, optimization, checkpointing, and validation metrics execute. It does not measure language-model quality and must never be reported as a benchmark.

## Dataset admission requirements

Before a corpus is used for research results, record and verify:

- Dataset title and source URL.
- Dataset version or retrieval date.
- License and whether it permits the intended research and redistribution.
- Original source provenance where the corpus aggregates third-party text.
- SHA-256 checksum of the exact UTF-8 source file.
- Byte count and preprocessing procedure.
- Language coverage and known limitations.
- Exact contiguous train, validation, and test boundaries.

Do not proceed with a corpus whose license or provenance is unclear.

## Splitting and batching

The initial default split is contiguous:

- Training: 90%.
- Validation: 5%.
- Test: 5%.

The split helper returns three non-overlapping byte regions. The batch generator receives one split at a time and samples windows only within that region, so a window cannot cross a train/validation/test boundary.

For a configured prediction window of L bytes, each sampled input window contains L+1 bytes. The target sequence is the one-byte shift of that window. The model's prediction loss uses the same alignment.

## Training

- Objective: self-supervised next-byte prediction.
- Auxiliary objective: average gate-sum budget penalty targeting two slots per valid block.
- Optimizer: AdamW.
- Record random seed, model configuration, learning rate, batch size, sequence length, number of steps, hardware, and software versions.
- Use validation data for model selection and threshold calibration only.

## Evaluation

Primary metric: bits per byte, calculated from summed next-byte negative log likelihood divided by the number of valid predictions and by ln(2).

The test split is evaluated only after model and threshold choices are frozen. Never tune on test results.

The evaluation command verifies that the test corpus file has the same SHA-256 checksum as the corpus used to create the checkpoint.

## Required comparisons

When baseline implementations are ready, compare:

1. Standard byte-level Transformer.
2. Non-hierarchical byte model.
3. Fixed-size hierarchical compression.
4. BYTE adaptive compression.

Use identical corpus splits and comparable parameter budgets where practical. Report parameter counts and hardware if exact matching is not possible.

## Required reporting

Report bits per byte, next-byte accuracy, mean gate sum, active slots at inference, actual global sequence lengths, peak memory, throughput, wall-clock cost, parameter count, corpus metadata, and configuration.

Do not claim adaptive compression is beneficial unless predictive quality and computational cost jointly support that conclusion.
