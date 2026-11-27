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


## Implemented v0.2 neural foundation

- Local byte positions use deterministic sinusoidal absolute-position encodings.
- The global sequence contains each retained latent slot as an individual token, not one pooled token per block.
- Global latent tokens receive a sinusoidal block-position encoding and a learned slot-position embedding.
- A learned global query reads only latent slots from completed earlier blocks.
- Training retains all candidate slots with differentiable gate scaling.
- Evaluation/inference physically selects only slots whose gates meet the configured threshold.
- The current block is compressed after its predictions are produced.
- The first implementation recomputes global context per block and does not yet use a KV cache.
