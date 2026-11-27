# BYTE v0.2.0 Adaptive Compression Implementation Plan

> **For agentic workers:** Use the host's available task-by-task implementation workflow. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and validate BYTE's tiny causal raw-byte model with content-dependent latent gating and next-byte prediction.

**Architecture:** Use PyTorch modules for byte embeddings, within-block causal processing, four learned latent queries, differentiable gates, and a causal global encoder over completed blocks. Preserve the block-causal contract: predictions in block j may use local prefix information from block j and compressed representations only from blocks before j.

**Tech Stack:** Python 3.10+, PyTorch, pytest, YAML configuration, GitHub, and a free hosted notebook environment such as Google Colab for training.

## Global Constraints

- Text-only input represented as raw UTF-8 byte IDs 0–255; no subword tokenizer.
- Initial block size is 16 bytes.
- Embedding and hidden dimension are 128.
- Local and global encoders each have two Transformer layers and four attention heads.
- Each completed block produces at most four candidate latent slots.
- Training uses differentiable sigmoid gates with a target mean gate activation of two slots per valid block.
- Inference applies a configurable threshold and must physically omit/mask inactive slots from global attention computation; multiplying by zero while retaining dense attention is not evidence of compute reduction.
- Next-byte prediction uses 256 output classes and categorical cross-entropy.
- No prediction at position t may depend on any byte after t.
- Incomplete final blocks use padding plus a validity mask. Padding must not affect losses or gate-budget statistics.
- Compare standard byte Transformer, non-hierarchical byte model, fixed-size hierarchical compression, and adaptive BYTE under matched data splits and comparable parameter budgets where practical.
- Record dataset provenance, license, version, checksum, preprocessing, splits, random seed, software versions, parameter count, and hardware.
- Never use test data for model or threshold selection.
- Do not claim novelty, intelligence, or superiority without evidence.
- Keep training tiny and reproducible; no large-scale or distributed training in this milestone.

---

### Task 1: Establish PyTorch model primitives and causal block contract

**Files:**
- Modify: `pyproject.toml`
- Create: `src/byte/modeling.py`
- Create: `src/byte/masks.py`
- Create: `tests/test_masks.py`
- Create: `tests/test_modeling.py`

**Interfaces:**
- `build_block_causal_mask(num_blocks: int, device=None) -> torch.Tensor`: boolean mask where a block may attend only to earlier completed blocks; the current block's global context is excluded.
- `ByteEmbedding(vocab_size: int = 256, d_model: int = 128)`: accepts integer tensor shaped `[batch, sequence]` and returns `[batch, sequence, d_model]`.
- `LocalCausalEncoder(d_model: int = 128, nhead: int = 4, num_layers: int = 2)`: accepts block tensors `[batch, blocks, block_size, d_model]` and a valid-byte mask; returns same-shaped causal contextual states.
- Add a `train` optional dependency group containing a supported PyTorch version. Keep pytest in the `dev` group.

- [ ] **Step 1: Add focused failing tests**

Test that:
- The block mask for three blocks allows block 0 to see no prior block, block 1 to see block 0, and block 2 to see blocks 0 and 1.
- Embedding input shape `[2, 7]` produces `[2, 7, 128]`.
- Byte ID 255 is accepted; ID 256 fails through the embedding's documented index boundary.
- Changing a later byte in a block cannot change local encoder outputs at earlier positions.
- Padded positions do not become valid attention keys.

- [ ] **Step 2: Verify the relevant failure**

Run: `pytest -q tests/test_masks.py tests/test_modeling.py`

Expected: collection or assertion failures specifically because the new mask and model interfaces do not exist yet. Resolve import/setup errors rather than treating them as a valid red result.

- [ ] **Step 3: Implement the minimum primitives**

Implement the boolean block visibility mask, embedding, and local causal Transformer. Use explicit key-padding masks. Document tensor dimensions in docstrings. Do not implement compression in this task.

- [ ] **Step 4: Verify the focused pass**

Run: `pytest -q tests/test_masks.py tests/test_modeling.py`

Expected: all focused mask and local-causality tests pass.

- [ ] **Step 5: Run the existing integration suite**

Run: `pytest -q`

Expected: existing codec tests and new model primitive tests pass.

- [ ] **Step 6: Commit the passing deliverable**

```bash
git add pyproject.toml src/byte/modeling.py src/byte/masks.py tests/test_masks.py tests/test_modeling.py
git commit -m "feat: add causal byte model primitives"
```

### Task 2: Implement adaptive latent compression and budget regularization

**Files:**
- Create: `src/byte/compression.py`
- Create: `tests/test_compression.py`

**Interfaces:**
- `AdaptiveCompressor(d_model: int = 128, num_slots: int = 4)`
- Forward input: completed block states `[batch, blocks, block_size, d_model]` plus valid-byte mask.
- Training output: latent tensor `[batch, blocks, 4, d_model]` and gate tensor `[batch, blocks, 4]`.
- `compression_budget_loss(gates, valid_blocks, target_active_slots: float = 2.0) -> torch.Tensor`.
- Inference output includes packed active latents and their block indices so global attention can skip inactive slots.

- [ ] **Step 1: Add focused failing tests**

Test that:
- Output latent and gate shapes match the interfaces.
- Every gate is finite and lies in [0, 1].
- Padded bytes do not influence the compression output or budget statistic for a valid block.
- The budget loss is zero when the mean valid gate sum is exactly two, and positive when it differs.
- Backpropagation produces finite gradients for latent queries and gate-network parameters.
- A configurable inference threshold returns only slots whose gates meet or exceed that threshold, preserving block indices and stable order.
- An all-inactive batch produces a correctly shaped empty packed representation without NaNs.

- [ ] **Step 2: Verify the relevant failure**

Run: `pytest -q tests/test_compression.py`

Expected: failures because the compressor and budget loss do not yet exist.

- [ ] **Step 3: Implement the minimum compressor**

Use four learned latent queries attending over each completed block's local states. Produce a learned group summary and per-slot sigmoid gates. Apply gates to latent vectors. Compute the budget penalty from valid blocks only. Implement thresholded packing for inference; do not claim compute savings from dense zero-masked tensors.

- [ ] **Step 4: Verify the focused pass**

Run: `pytest -q tests/test_compression.py`

Expected: all shape, masking, threshold, budget, and gradient tests pass.

- [ ] **Step 5: Run affected integration tests**

Run: `pytest -q tests/test_masks.py tests/test_modeling.py tests/test_compression.py`

Expected: all primitive and compressor tests pass together.

- [ ] **Step 6: Commit the passing deliverable**

```bash
git add src/byte/compression.py tests/test_compression.py
git commit -m "feat: add adaptive latent compression"
```

### Task 3: Integrate the causal BYTE language model

**Files:**
- Create: `src/byte/model.py`
- Create: `tests/test_model.py`
- Modify: `src/byte/__init__.py`

**Interfaces:**
- `BYTEConfig`: immutable configuration for vocabulary size, dimensions, block size, local/global layer counts, attention heads, slot count, target active slots, and inference gate threshold.
- `BYTEModel(config: BYTEConfig)`
- Training forward input: `input_ids [batch, sequence]` and optional `attention_mask`.
- Training output: next-byte logits `[batch, sequence, 256]`, prediction loss, compression-budget loss, and gate statistics.
- Inference method: incremental block processing that exposes only compressed representations of fully completed prior blocks.

- [ ] **Step 1: Add focused failing tests**

Test that:
- Input `[2, 32]` produces logits `[2, 32, 256]`.
- Next-byte targets are shifted by exactly one byte and the final unavailable target is excluded.
- Padding positions do not contribute to prediction or compression losses.
- Predictions for positions in an earlier block are unchanged when only later blocks are modified.
- Predictions earlier within the same block are unchanged when only later bytes in that block are modified.
- The current block's compressed latents are not used to predict bytes inside that same block.
- A short sequence and a non-multiple-of-16 sequence run without shape errors.
- Total loss equals prediction loss plus configured weighted budget loss.

- [ ] **Step 2: Verify the relevant failure**

Run: `pytest -q tests/test_model.py`

Expected: failures because the integrated BYTE model and forward contract do not yet exist.

- [ ] **Step 3: Implement the minimum integrated model**

Combine embedding, local causal encoder, completed-block compression, causal global context, and 256-class prediction head. For each block, use only prior completed-block summaries as global context. Keep same-block prediction on local causal states plus prior global context. Implement loss masking and explicit train/inference paths.

- [ ] **Step 4: Verify the focused pass**

Run: `pytest -q tests/test_model.py`

Expected: all shape, target alignment, padding, loss, and future-information isolation tests pass.

- [ ] **Step 5: Run the full unit suite**

Run: `pytest -q`

Expected: codec, mask, primitive, compressor, and integrated model tests all pass.

- [ ] **Step 6: Commit the passing deliverable**

```bash
git add src/byte/model.py src/byte/__init__.py tests/test_model.py
git commit -m "feat: integrate BYTE next-byte model"
```

### Task 4: Add reproducible tiny training and baseline evaluation

**Files:**
- Create: `src/byte/data.py`
- Create: `src/byte/train.py`
- Create: `src/byte/evaluate.py`
- Create: `configs/tiny.yaml` (replace the current placeholder)
- Create: `tests/test_data.py`
- Create: `tests/test_training.py`
- Create: `docs/experiments/protocol-v0.2.md`
- Modify: `README.md`

**Interfaces:**
- `encode_corpus(text: str) -> list[int]`: UTF-8 byte conversion using the existing codec.
- `make_next_byte_batches(byte_ids, block_size, sequence_length, batch_size, seed)`: deterministic next-byte batches with target alignment.
- `train_step(model, batch, optimizer) -> dict[str, float]`: one optimization step with prediction and budget losses.
- `evaluate_bits_per_byte(model, batches) -> float`: cross-entropy converted to bits per byte, excluding padding.
- CLI entry points: `python -m byte.train --config configs/tiny.yaml` and `python -m byte.evaluate --config configs/tiny.yaml --checkpoint PATH`.

- [ ] **Step 1: Add focused failing tests**

Test that:
- A known UTF-8 string becomes exactly its raw byte sequence.
- Batch construction is deterministic for a fixed seed.
- Input/target batches are shifted by one byte.
- No batch crosses a declared dataset split boundary.
- Training on a tiny synthetic corpus produces finite loss and finite gradients.
- One optimizer step changes at least one trainable parameter.
- Bits-per-byte evaluation matches a hand-computed small example and excludes padding.

- [ ] **Step 2: Verify the relevant failure**

Run: `pytest -q tests/test_data.py tests/test_training.py`

Expected: failures because the data, training, and evaluation interfaces do not yet exist.

- [ ] **Step 3: Implement the minimum reproducible pipeline**

Use a tiny in-memory corpus fixture for tests. Keep real dataset download separate from core model code. Record source, license, version, checksum, preprocessing, and split boundaries in the experiment protocol before selecting a public corpus. Seed Python and PyTorch. Save model configuration and checkpoint metadata. Implement bits-per-byte from summed negative log likelihood divided by the number of valid predicted bytes and by ln(2).

Implement baseline variants behind a shared configuration/interface so the evaluation loop uses identical batches and splits. Include fixed-slot hierarchical compression as an ablation. Match parameter budgets as closely as practical and report parameter counts when exact matching is impossible.

- [ ] **Step 4: Verify the focused pass**

Run: `pytest -q tests/test_data.py tests/test_training.py`

Expected: deterministic batches, correct shifts, finite tiny training, parameter updates, and correct bits-per-byte calculation all pass.

- [ ] **Step 5: Run the complete suite and smoke experiment**

Run: `pytest -q`

Expected: all project tests pass.

Run: `python -m byte.train --config configs/tiny.yaml --smoke-test`

Expected: a short synthetic-corpus run completes, reports finite losses, and writes a checkpoint plus configuration metadata to the configured run directory.

- [ ] **Step 6: Commit the passing deliverable**

```bash
git add src/byte/data.py src/byte/train.py src/byte/evaluate.py configs/tiny.yaml tests/test_data.py tests/test_training.py docs/experiments/protocol-v0.2.md README.md
git commit -m "feat: add reproducible BYTE training and evaluation"
```

---

## Unresolved product and experiment decisions

These decisions do not block implementing the architecture interfaces, but must be settled before reporting a benchmark result:

1. **Inference gate threshold:** expose as configuration; initial engineering recommendation is 0.5, then select or calibrate using validation data only. Never tune it on test data.
2. **Public training corpus:** choose a small corpus only after confirming license, source provenance, language coverage, and a stable version/checksum.
3. **Training budget:** choose a tiny step/token budget suitable for the available free accelerator; report actual hardware and runtime.
4. **Baseline parameter matching:** exact equality may be impractical across different architectures; report all parameter counts and define a reasonable comparison tolerance before experiments.
5. **Global positional encoding:** choose a deterministic group-position scheme and test long-sequence behavior before training; the architecture requires ordered group identity but does not mandate a particular positional encoding family.
