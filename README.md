# BYTE

**BYTE** is a small, reproducible research project exploring text modeling directly from raw UTF-8 bytes.

## Research question

Can a model learn useful linguistic structure and predictive capabilities directly from raw bytes, using learned hierarchical representations and adaptive compression rather than a predefined subword vocabulary?

Byte-level modeling and hierarchical byte processing already have substantial prior work. BYTE's contribution must therefore be established through a clearly specified mechanism and controlled experiments—not by claiming that byte-level modeling itself is new.

## Current status

**Phase 1 — research foundation and minimal byte interface.**

The first implementation milestone is deliberately small:
- Convert Unicode text to UTF-8 byte IDs in the range 0–255.
- Decode byte sequences back to text with explicit error handling.
- Test round-trip behavior for ASCII, multilingual text, emoji, and empty input.
- Establish reproducible experiment conventions before training models.

No trained language model is included yet.

## Research baselines

BYTE will eventually be compared against:
1. A standard byte-level Transformer.
2. A non-hierarchical byte model.
3. BYTE's proposed hybrid architecture with learned adaptive compression.

All comparisons should use the same data splits and comparable parameter budgets.

## Evaluation

Planned metrics include bits per byte, next-byte accuracy, effective sequence-length reduction, memory use, throughput, and training/inference cost. Dataset provenance, licensing, checksums, and split boundaries must be recorded. The test set must not be used for tuning.

## Prior work

- [ByT5 (2022)](https://aclanthology.org/2022.tacl-1.17/)
- [CANINE (2022)](https://aclanthology.org/2022.tacl-1.5/)
- [MEGABYTE (2023)](https://papers.neurips.cc/paper_files/paper/2023/hash/f8f78f8043f35890181a824e53a57134-Abstract-Conference.html)
- [Byte Latent Transformer / BLT (2025)](https://aclanthology.org/2025.acl-long.453/)

## Development

The project is Python-first. The initial code is designed to be lightweight and testable. Model training will be added only after the byte interface and experimental protocol are reliable.

## License

Apache-2.0. See [LICENSE](LICENSE).
