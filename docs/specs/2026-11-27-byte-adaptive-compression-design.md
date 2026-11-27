# BYTE v0.2.0 — Adaptive Compression Architecture

**Document:** Architecture Design Specification  
**Project:** BYTE  
**Version:** 0.2.0  
**Status:** Approved design; implementation not yet started  
**Date:** 27 November 2026  
**Modality:** Text  
**Primary objective:** Self-supervised next-byte prediction

## 1. Purpose

BYTE investigates whether a compact neural architecture can learn useful linguistic representations directly from raw UTF-8 bytes through hierarchical processing and learned adaptive compression.

The central research question is:

> Can content-dependent compression reduce the effective sequence length of byte-level language modeling while preserving predictive information?

BYTE is an experimental research project. Its proposed architecture is not assumed to be novel or superior. Those claims require comparison with established work and reproducible experiments.

## 2. Design principles

1. Input consists of raw UTF-8 bytes.
2. Every byte is represented by an integer from 0 to 255.
3. No BPE, WordPiece, SentencePiece, or subword vocabulary is used.
4. Local processing captures nearby byte relationships.
5. Compression is learned rather than determined by a fixed linguistic vocabulary.
6. Global processing captures relationships across completed groups.
7. Causal information flow must be preserved.
8. Model quality and computational efficiency must be evaluated separately.

## 3. Architecture overview

The model consists of six principal components:

1. Raw byte input.
2. Byte embedding.
3. Local causal encoder.
4. Adaptive compression module.
5. Global causal encoder.
6. Next-byte prediction head.

### 3.1 Raw byte input

For a text sequence x, UTF-8 encoding produces:

`B = [b_1,b_2,\ldots,b_T]`

where each `b_i` belongs to the integer range 0 through 255.

The sequence is divided into groups of 16 bytes.

Incomplete final groups are supported through padding and an explicit validity mask. Padding positions must not contribute to losses or compression statistics.

### 3.2 Byte embedding

Each byte is mapped to a learned vector:

`e_i = E[b_i] + P[i]`

Initial embedding dimension: 128.

The positional representation must preserve the byte's position within its group. The model must also receive sufficient positional information to distinguish groups and their ordering.

### 3.3 Local causal encoder

Initial configuration:

- Two Transformer layers.
- Four attention heads.
- Hidden dimension: 128.
- Causal attention mask.
- Group size: 16 bytes.

The local encoder processes one group at a time. Its output is a sequence of 16 contextual vectors. Each position may attend only to itself and earlier positions in the same group.

### 3.4 Adaptive compression module

This is BYTE's primary research component.

For each completed group, the compression module creates four candidate latent representations using four learned latent queries:

`z_{j,k} = Attention(q_k,H_j,H_j)`

A learned gate determines each latent's contribution:

`g_{j,k} = sigmoid(f_theta(z_{j,k},s_j))`

where:

- `z_{j,k}` is a candidate latent.
- `s_j` is a learned group summary.
- `f_theta` is a small learned gating network.
- Each gate is between zero and one.

The gated representation is:

`z_tilde_{j,k} = g_{j,k} z_{j,k}`

The module has a maximum of four slots per group. The initial target is an average of two active slots per valid group.

During training, gates remain differentiable. At inference, a specified threshold determines active slots. Inactive slots are masked out of subsequent global processing.

The threshold, gate distribution, and resulting compute reduction must be measured.

### 3.5 Global causal encoder

Initial configuration:

- Two Transformer layers.
- Four attention heads.
- Hidden dimension: 128.
- Causal attention across completed groups.

The global encoder processes compressed representations of earlier completed groups. Representations from the current incomplete group must not enter the global context used to predict bytes within that group.

### 3.6 Next-byte prediction head

The prediction head produces:

`p(b_{t+1} | b_{<=t})`

with 256 output classes.

The model is trained using categorical cross-entropy. Padding positions are excluded from the loss.



## 3.7 Global latent sequence and positional information

Each retained latent slot is supplied to the global encoder as an individual sequence element; BYTE does not collapse all slots from a block into one fixed summary token.

- During training, all candidate slots are present and each latent vector is scaled by its differentiable gate.
- During evaluation/inference, slots below the configured gate threshold are physically omitted from the global encoder's input sequence.
- Every latent receives a deterministic sinusoidal block-position encoding and a learned slot-position embedding.
- A learned global query token is appended after the prior-block latent sequence. Its output is used as context for the current block.
- The current block is compressed only after its own byte predictions have been produced.
- The first implementation recomputes the global context from the prior latent sequence for each block; it does not yet implement a key/value cache.

This design makes the variable number of active latent slots visible to the global model and allows inference-time sequence length to vary with learned gates.

## 4. Causal information-flow requirements

For a byte position t, its prediction must not depend on bytes after t.

The processing sequence is:

1. Process the current group with a causal local encoder.
2. Predict bytes within that group using previous local bytes and global representations from earlier completed groups.
3. After the group has been fully observed, compress it.
4. Add its compressed representations to the global context.
5. Continue to the next group.

The compression operation may inspect all positions of a completed group because that group is entirely in the past when predicting subsequent groups.

No prediction may use a compressed representation containing future bytes.

## 5. Training objective

The initial objective combines predictive quality and compression-budget control:

`L_total = L_prediction + lambda L_budget`

The budget term penalizes deviations from the desired average active-slot count:

`L_budget = (mean_valid_gates - 2)^2`

The coefficient lambda controls the trade-off. Its value will be selected using training and validation data only.

A separate experiment will measure the difference between soft-gated training and thresholded inference.

## 6. Initial model configuration

| Parameter | Initial value |
|---|---:|
| Byte vocabulary | 256 |
| Embedding dimension | 128 |
| Group size | 16 bytes |
| Local Transformer layers | 2 |
| Global Transformer layers | 2 |
| Attention heads | 4 |
| Candidate latent slots | 4 |
| Target active slots | 2 |
| Prediction classes | 256 |
| Learning objective | Next-byte prediction |

These are starting hyperparameters, not proven optimal values.

## 7. Experimental baselines

- **A — Standard byte Transformer:** conventional causal Transformer operating directly on byte sequences.
- **B — Non-hierarchical byte model:** byte-level model without the proposed hierarchical compression module.
- **C — Fixed-size hierarchical compression:** hierarchical model with a fixed number of retained representations.
- **D — BYTE adaptive compression:** proposed model with learned content-dependent gates.

Comparisons must use identical data partitions and comparable parameter budgets wherever practical.

## 8. Evaluation metrics

Primary metric: bits per byte.

Additional metrics:

- Next-byte accuracy.
- Average active slots per group.
- Effective sequence-length reduction.
- Compression ratio.
- Peak memory use.
- Training throughput.
- Inference throughput.
- Wall-clock training time.
- Parameter count.

A lower number of active slots is not automatically an improvement. Compression is useful only if predictive quality is preserved at a meaningful computational benefit.

## 9. Dataset and reproducibility requirements

Training data must be publicly accessible and legally usable.

For each dataset, record its name and source, license, version, retrieval date, checksum, preprocessing operations, and train/validation/test split boundaries.

Test data must never be used for model selection or hyperparameter tuning.

All experiments must record random seeds, configuration files, software versions, and relevant hardware information.

## 10. Testing requirements

Before model training, the implementation must pass:

1. Byte embedding shape tests.
2. Local causal-mask tests.
3. Compression latent shape tests.
4. Gate range tests.
5. Padding-mask tests.
6. Compression-budget loss tests.
7. Global causal-mask tests.
8. Future-information leakage tests.
9. Next-byte target alignment tests.
10. Forward and backward gradient tests.

A particularly important test will alter future bytes while holding the preceding context constant. Predictions for earlier positions must remain unchanged.

## 11. Scope exclusions

BYTE v0.2.0 will not attempt multimodal processing, distributed training, large-scale pretraining, external memory systems, autonomous agents, or claims of general intelligence, superior reasoning, or research novelty without evidence.

## 12. Acceptance criteria

The architecture is ready for its first training experiment when:

- All components have defined interfaces.
- Causal information flow is verified.
- Adaptive gates are differentiable during training.
- Inference-time slot masking is implemented.
- The budget penalty is tested.
- Baseline configurations are documented.
- A small public dataset and reproducible splits are selected.
- The complete training pipeline passes a tiny end-to-end smoke test.

## 13. Known research risks

- **Information loss:** compression may discard information needed for prediction.
- **Gate collapse:** the model may activate too many or too few slots.
- **Inference mismatch:** hard thresholding may reduce quality compared with soft-gated training.
- **Compute overhead:** compression may cost more than it saves.
- **Baseline fairness:** improvements may result from parameter count or training differences rather than adaptive compression.

Each risk requires an explicit experiment.

## 14. Prior research

- [ByT5 (2022)](https://aclanthology.org/2022.tacl-1.17/)
- [CANINE (2022)](https://aclanthology.org/2022.tacl-1.5/)
- [MEGABYTE (2023)](https://papers.neurips.cc/paper_files/paper/2023/hash/f8f78f8043f35890181a824e53a57134-Abstract-Conference.html)
- [Byte Latent Transformer (2025)](https://aclanthology.org/2025.acl-long.453/)

BYTE must identify a precise distinction from this prior work before making any novelty claim.

## 15. Final design statement

BYTE v0.2.0 is a tiny, text-only, raw-byte neural architecture combining causal local processing, learned adaptive latent compression, causal global processing, and self-supervised next-byte prediction.

Its central experimental hypothesis is that content-dependent compression can reduce effective sequence length while retaining useful predictive information.

The hypothesis remains unproven until the model is implemented, trained, and compared against controlled baselines.
