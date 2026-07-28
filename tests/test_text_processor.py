"""Tests for the text_processor module."""

from __future__ import annotations

import pytest
from src.config import ChunkingConfig
from src.text_processor import (
    ProcessedDocument,
    TextProcessingError,
    TextProcessor,
)


def test_clean_text_removes_extra_whitespace_and_invisible_chars() -> None:
    """Cleaning should remove invisible chars and repeated blank lines."""
    processor = TextProcessor(ChunkingConfig(enable_normalization=False))

    raw = "  Hello\u200b   world. \n\n\n\nNew line.  "
    cleaned = processor.clean_text(raw)

    assert cleaned == "Hello world.\n\nNew line."


def test_normalize_text_normalizes_quotes_and_dashes() -> None:
    """Normalization should unify quotes and dashes."""
    processor = TextProcessor(ChunkingConfig(enable_cleaning=False))

    raw = "“Hello” — ‘world’"
    normalized = processor.normalize_text(raw)

    assert normalized == "\"Hello\" - 'world'"


def test_process_documents_returns_processed_documents() -> None:
    """Documents should be converted to ProcessedDocument objects."""
    processor = TextProcessor(ChunkingConfig())

    documents = [
        {
            "document_id": "1",
            "text": "  “Hi”  ",
            "metadata": {"title": "T"},
        }
    ]

    processed = processor.process_documents(documents)

    assert isinstance(processed[0], ProcessedDocument)
    assert processed[0].text == '"Hi"'
    assert processed[0].metadata["title"] == "T"


def test_process_documents_raises_when_text_missing() -> None:
    """Missing text should raise TextProcessingError."""
    processor = TextProcessor(ChunkingConfig())

    with pytest.raises(TextProcessingError):
        processor.process_documents([{"document_id": "1"}])


def test_process_text_respects_disabled_normalization() -> None:
    """Disabled normalization should preserve original quotes."""
    processor = TextProcessor(
        ChunkingConfig(
            enable_normalization=False,
            enable_cleaning=True,
        )
    )

    assert processor.process_text("  “Hi”  ") == "“Hi”"
