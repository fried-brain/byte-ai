# BYTE v0.2.0 — Experiment Protocol

## Status

The initial pilot corpus is a revision-pinned 50,000-story sample from the English TinyStories training split. TinyStories is synthetic short-story text generated with GPT-3.5/GPT-4. It is suitable for testing a tiny language-model pipeline, but results must not be generalized to natural web text or treated as a broad benchmark.

The dataset repository identifies the license as CDLA-Sharing-1.0. If raw or modified data is published, the agreement requires publication under the same agreement and preservation of attribution. Section 3.5 imposes no obligations on qualifying Results, while the definition of Results excludes outputs containing more than a de minimis portion of the source data. Do not assume model weights qualify if they memorize substantial story text. The raw subset is kept outside Git and its SHA-256 is recorded in the generated manifest and training metadata.

A smoke test checks that batching, optimization, checkpointing, and validation metrics execute. It does not measure language-model quality and must never be reported as a benchmark.

## Selected pilot dataset\n\n- Dataset: [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)\n- Pinned revision: `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`\n- Split sampled: `train` only\n- License: CDLA-Sharing-1.0\n- Sample: 50,000 stories, streaming shuffle with seed 42 and buffer size 10,000\n- Paper: [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759)\n\nThe preparation utility writes the exact text subset and a manifest with story count, byte count, revision, license, seed, and SHA-256. The training pipeline then creates a contiguous 90/5/5 split within this sampled training subset. These are BYTE pilot splits, not the official TinyStories validation/test splits.\n\n## Dataset admission requirements

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
4. BYTE adaptive compression.\n\nSet `model.variant` to `standard_transformer`, `byte_gru`, `fixed_hierarchical`, or `byte` in the YAML configuration. The fixed hierarchical ablation uses exactly two learned slots per valid block and no adaptive gate network.

Use identical corpus splits and comparable parameter budgets where practical. Report parameter counts and hardware if exact matching is not possible.

## Required reporting

Report bits per byte, next-byte accuracy, mean gate sum, active slots at inference, actual global sequence lengths, peak memory, throughput, wall-clock cost, parameter count, corpus metadata, and configuration.

Do not claim adaptive compression is beneficial unless predictive quality and computational cost jointly support that conclusion.
