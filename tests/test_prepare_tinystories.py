import hashlib
import json

import pytest

from byte.prepare_tinystories import write_story_subset


def test_write_story_subset_creates_utf8_text_and_reproducibility_manifest(tmp_path) -> None:
    output = tmp_path / "tiny-stories.txt"
    records = iter([{"text": "Once upon a time."}, {"text": "नमस्ते, BYTE!"}])

    manifest = write_story_subset(
        records,
        output,
        max_stories=2,
        seed=42,
        dataset_id="roneneldan/TinyStories",
        revision="f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
        split="train",
        license_name="CDLA-Sharing-1.0",
    )

    expected = "Once upon a time.\n\nनमस्ते, BYTE!\n"
    assert output.read_text(encoding="utf-8") == expected
    assert manifest["story_count"] == 2
    assert manifest["byte_count"] == len(expected.encode("utf-8"))
    assert manifest["sha256_utf8"] == hashlib.sha256(expected.encode("utf-8")).hexdigest()
    saved = json.loads(output.with_suffix(".txt.manifest.json").read_text(encoding="utf-8"))
    assert saved == manifest


def test_write_story_subset_rejects_nonpositive_story_limit(tmp_path) -> None:
    with pytest.raises(ValueError, match="max_stories must be positive"):
        write_story_subset(
            [],
            tmp_path / "empty.txt",
            max_stories=0,
            seed=42,
            dataset_id="roneneldan/TinyStories",
            revision="pinned",
            split="train",
            license_name="CDLA-Sharing-1.0",
        )
