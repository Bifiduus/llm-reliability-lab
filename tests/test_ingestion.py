import pytest

from reliability_lab.ingestion import split_text


def test_split_text():
    text = "one two three four five six"

    chunks = split_text(text, chunk_size=3, overlap=1)

    assert chunks == [
        "one two three",
        "three four five",
        "five six",
    ]


def test_short_text():
    text = "one two three"

    chunks = split_text(text, chunk_size=10, overlap=2)

    assert chunks == ["one two three"]


def test_empty_text():
    assert split_text("") == []


def test_invalid_chunk_size():
    with pytest.raises(ValueError):
        split_text("hello world", chunk_size=0)


def test_invalid_overlap():
    with pytest.raises(ValueError):
        split_text("hello world", chunk_size=5, overlap=5)
