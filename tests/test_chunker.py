"""Tests for the chunker module."""

from __future__ import annotations

import json

import pytest
from src.chunker import (
    Chunker,
    ChunkExporter,
    ChunkingError,
    ChunkValidationError,
    compute_statistics,
)
from src.config import ChunkingConfig
from src.text_processor import ProcessedDocument


def _metadata() -> dict[str, object]:
    """Return valid sample metadata."""
    return {
        "title": "T",
        "question": "Q",
        "answer": "A",
        "dataset_split": "train",
        "source_dataset": "squad",
    }


def _processed_document(
    text: str,
    document_id: str = "doc_1",
) -> ProcessedDocument:
    """Create a processed document for tests."""
    return ProcessedDocument(
        document_id=document_id,
        text=text,
        metadata=_metadata(),
    )


def test_invalid_overlap_raises_validation_error() -> None:
    """Overlap >= chunk_size should raise validation error."""
    with pytest.raises(ChunkValidationError):
        Chunker({"chunk_size": 100, "chunk_overlap": 100})


def test_recursive_chunking_produces_valid_overlapping_chunks() -> None:
    """Recursive chunks should be valid and overlap when possible."""
    text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
    config = ChunkingConfig(
        chunk_size=30,
        chunk_overlap=5,
        chunk_strategy="recursive_character",
    )

    chunks = Chunker(config).chunk_documents([_processed_document(text)])

    assert chunks

    for chunk in chunks:
        assert chunk.start_index >= 0
        assert chunk.end_index <= len(text)
        assert text[chunk.start_index : chunk.end_index] == chunk.text
        assert chunk.metadata["dataset_split"] == "train"

    if len(chunks) > 1:
        assert chunks[1].start_index < chunks[0].end_index


def test_character_chunking_exact_windows() -> None:
    """Character strategy should produce exact sliding windows."""
    text = "abcdefghijklmnopqrstuvwxyz"
    config = ChunkingConfig(
        chunk_size=10,
        chunk_overlap=2,
        chunk_strategy="character",
    )

    chunks = Chunker(config).chunk_documents([_processed_document(text)])

    assert chunks[0].text == "abcdefghij"
    assert chunks[1].start_index == 8
    assert chunks[1].text == "ijklmnopqr"


def test_sentence_splitter_prefers_sentence_boundary() -> None:
    """Sentence strategy should end on sentence boundary when possible."""
    text = "First sentence. Second sentence. Third sentence."
    config = ChunkingConfig(
        chunk_size=20,
        chunk_overlap=0,
        chunk_strategy="sentence",
    )

    chunks = Chunker(config).chunk_documents([_processed_document(text)])

    assert chunks
    assert chunks[0].text.endswith(".")


def test_semantic_chunking_placeholder_raises_when_disabled() -> None:
    """Semantic chunking should raise when disabled."""
    config = ChunkingConfig(
        chunk_strategy="semantic",
        future_semantic_chunking=False,
    )
    chunker = Chunker(config)

    with pytest.raises(ChunkingError):
        chunker.chunk_documents([_processed_document("Some text.")])


def test_export_json_and_jsonl(tmp_path) -> None:
    """JSON and JSONL export should produce valid files."""
    config = ChunkingConfig(
        chunk_size=20,
        chunk_overlap=0,
    )
    chunks = Chunker(config).chunk_documents(
        [_processed_document("Alpha beta. Gamma delta.")]
    )

    exporter = ChunkExporter(config, output_dir=tmp_path)

    json_path = exporter.export(
        chunks,
        file_name="chunks.json",
        export_format="json",
    )
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(loaded) == len(chunks)

    jsonl_path = exporter.export(
        chunks,
        file_name="chunks.jsonl",
        export_format="jsonl",
    )
    lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(chunks)


def test_export_csv(tmp_path) -> None:
    """CSV export should include header and rows."""
    config = ChunkingConfig(
        chunk_size=20,
        chunk_overlap=0,
    )
    chunks = Chunker(config).chunk_documents(
        [_processed_document("Alpha beta. Gamma delta.")]
    )

    exporter = ChunkExporter(config, output_dir=tmp_path)
    csv_path = exporter.export(
        chunks,
        file_name="chunks.csv",
        export_format="csv",
    )

    content = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert content[0].startswith("chunk_id")
    assert len(content) == len(chunks) + 1


def test_export_parquet(tmp_path) -> None:
    """Parquet export should work when pandas/pyarrow are available."""
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    config = ChunkingConfig(
        chunk_size=20,
        chunk_overlap=0,
    )
    chunks = Chunker(config).chunk_documents(
        [_processed_document("Alpha beta. Gamma delta.")]
    )

    exporter = ChunkExporter(config, output_dir=tmp_path)
    parquet_path = exporter.export(
        chunks,
        file_name="chunks.parquet",
        export_format="parquet",
    )

    assert parquet_path.exists()


def test_compute_statistics() -> None:
    """Statistics should summarize chunks correctly."""
    config = ChunkingConfig(
        chunk_size=20,
        chunk_overlap=0,
    )
    chunks = Chunker(config).chunk_documents(
        [_processed_document("Alpha beta. Gamma delta. Epsilon zeta.")]
    )

    stats = compute_statistics(chunks)

    assert stats.number_of_chunks == len(chunks)
    assert stats.average_chunk_length > 0
    assert stats.estimated_tokens == sum(chunk.token_estimate for chunk in chunks)


def test_empty_document_is_skipped() -> None:
    """Empty documents should not produce chunks."""
    chunks = Chunker(ChunkingConfig()).chunk_documents([_processed_document("   ")])

    assert chunks == []
