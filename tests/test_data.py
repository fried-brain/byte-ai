import torch
import pytest

from byte.data import encode_corpus, make_next_byte_batches, split_byte_ids


def test_encode_corpus_returns_raw_utf8_bytes() -> None:
    text = "A🙂नमस्ते"
    assert encode_corpus(text) == list(text.encode("utf-8"))


def test_batches_align_next_byte_targets_and_attention_mask() -> None:
    batches = make_next_byte_batches(
        list(range(40)), block_size=4, sequence_length=8, batch_size=3, seed=5
    )
    batch = next(batches)

    assert batch["input_ids"].shape == (3, 9)
    assert batch["target_ids"].shape == (3, 8)
    assert batch["attention_mask"].shape == (3, 9)
    assert torch.equal(batch["target_ids"], batch["input_ids"][:, 1:])
    assert batch["attention_mask"].all()


def test_batch_sampling_is_reproducible_for_a_fixed_seed() -> None:
    values = [index % 256 for index in range(100)]
    first = next(make_next_byte_batches(values, 4, 8, 2, seed=42))
    second = next(make_next_byte_batches(values, 4, 8, 2, seed=42))
    assert torch.equal(first["input_ids"], second["input_ids"])


def test_splits_are_contiguous_and_batches_stay_inside_one_split() -> None:
    corpus = [10] * 40 + [20] * 20 + [30] * 20
    train, validation, test = split_byte_ids(
        corpus, train_fraction=0.5, validation_fraction=0.25
    )
    assert set(train) == {10}
    assert set(validation) == {20}
    assert set(test) == {30}

    train_batch = next(make_next_byte_batches(train, 4, 8, 2, seed=1))
    validation_batch = next(make_next_byte_batches(validation, 4, 8, 2, seed=1))
    test_batch = next(make_next_byte_batches(test, 4, 8, 2, seed=1))
    assert set(train_batch["input_ids"].flatten().tolist()) == {10}
    assert set(validation_batch["input_ids"].flatten().tolist()) == {20}
    assert set(test_batch["input_ids"].flatten().tolist()) == {30}


def test_batch_generator_rejects_insufficient_data() -> None:
    with pytest.raises(ValueError, match="at least sequence_length"):
        next(make_next_byte_batches([1, 2, 3], 2, 4, 1, seed=0))
