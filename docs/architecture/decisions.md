# BYTE architecture decisions

## Accepted for v0.1.0

- Modality: text only.
- Input representation: raw UTF-8 bytes, integer IDs 0 through 255.
- Tokenizer: no BPE, WordPiece, SentencePiece, or learned subword vocabulary.
- Model direction: hybrid local/global architecture (research target).
- Compression: learned and adaptive (research target; mechanism still to be specified).
- Learning objective: self-supervised next-byte prediction as the initial objective.
- Scale: tiny experiments first.
- Baselines: standard byte-level Transformer and non-hierarchical byte model.
- Data: public datasets only, with license and provenance recorded.

## Not yet implemented

The hybrid model, compression module, data pipeline, trainer, checkpointing, and evaluation harness are not implemented in this initial repository commit. These require explicit specifications and controlled tests before training.

## Design constraint

Do not describe the planned architecture as novel until a literature comparison identifies a precise distinction and experiments support its value.
