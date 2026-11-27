import torch

from byte.masks import build_block_causal_mask


def test_block_causal_mask_exposes_only_completed_prior_blocks() -> None:
    mask = build_block_causal_mask(3)

    assert mask.dtype == torch.bool
    assert mask.shape == (3, 3)
    # PyTorch attention-mask convention: True means the key is blocked.
    assert mask[0].tolist() == [True, True, True]
    assert mask[1].tolist() == [False, True, True]
    assert mask[2].tolist() == [False, False, True]


def test_zero_blocks_produces_empty_square_mask() -> None:
    mask = build_block_causal_mask(0)
    assert mask.shape == (0, 0)
    assert mask.dtype == torch.bool
