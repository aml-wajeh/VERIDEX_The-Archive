"""Tests for the embeddings module."""

from __future__ import annotations

import json

import numpy as np
import pytest
from src.config import EmbeddingConfig
from src.embeddings import (
    EmbeddingError,
    EmbeddingExporter,
    EmbeddingExportError,
    EmbeddingGenerator,
    EmbeddingRecord,
    EmbeddingValidationError,
)


class FakeModel:
    """Deterministic fake embedding model for fast tests."""

    def __init__(
        self,
        dim: int = 8,
        produce_nan: bool = False,
        variable_dim: bool = False,
    ) -> None:
        """Initialize fake model.

        Args:
            dim: Base embedding dimension.
            produce_nan: Whether to inject NaN values.
            variable_dim: Whether returned dim differs from advertised one.
        """
        self.dim = dim
        self.produce_nan = produce_nan
        self.variable_dim = variable_dim
        self.encode_calls = 0

    def encode(
        self,
        sentences: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = False,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        """Return deterministic vectors.

        Args:
            sentences: Input sentences.
            batch_size: Ignored.
            show_progress_bar: Ignored.
            normalize_embeddings: Ignored.
            convert_to_numpy: Ignored.

        Returns:
            NumPy array of fake embeddings.
        """
        self.encode_calls += 1
        vectors: list[np.ndarray] = []

        for sentence in sentences:
            vector_dim = self.dim + (1 if self.variable_dim else 0)
            vector = np.full(vector_dim, float(len(sentence)), dtype=np.float32)

            if self.produce_nan:
                vector[0] = np.nan

            vectors.append(vector)

        return np.vstack(vectors)

    def get_sentence_embedding_dimension(self) -> int:
        """Return the advertised model dimension.

        Returns:
            Base dimension (differs from encoded vectors when variable_dim).
        """
        return self.dim


def _sample_chunks(count: int = 3) -> list[dict[str, object]]:
    """Create sample chunk dictionaries.

    Args:
        count: Number of chunks.

    Returns:
        List of chunk-like dictionaries.
    """
    return [
        {
            "chunk_id": f"chunk_{index}",
            "document_id": f"doc_{index}",
            "text": f"sample text {index}",
            "metadata": {"title": f"title_{index}"},
        }
        for index in range(count)
    ]


def test_load_model_is_lazy_and_cached() -> None:
    """Model loader should be called only once."""
    calls = {"count": 0}
    model = FakeModel()

    def loader() -> FakeModel:
        calls["count"] += 1
        return model

    generator = EmbeddingGenerator(
        EmbeddingConfig(),
        model_loader=loader,
    )

    generator.load_model()
    generator.load_model()

    assert calls["count"] == 1


def test_encode_text_returns_valid_vector() -> None:
    """Single text encoding should return a valid vector."""
    generator = EmbeddingGenerator(
        EmbeddingConfig(),
        model_loader=lambda: FakeModel(dim=8),
    )

    vector = generator.encode_text("hello world")

    assert isinstance(vector, np.ndarray)
    assert vector.shape == (8,)
    assert np.all(np.isfinite(vector))


def test_encode_chunks_returns_records_and_statistics() -> None:
    """Batch encoding should return records and statistics."""
    chunks = _sample_chunks(5)
    generator = EmbeddingGenerator(
        EmbeddingConfig(batch_size=2),
        model_loader=lambda: FakeModel(dim=8),
    )

    records = generator.encode_chunks(chunks, show_progress=False)

    assert len(records) == 5
    assert all(isinstance(record, EmbeddingRecord) for record in records)
    assert all(record.dimension == 8 for record in records)

    stats = generator.statistics
    assert stats is not None
    assert stats.total_embeddings == 5
    assert stats.embedding_dimension == 8
    assert stats.batch_count == 3
    assert stats.memory_usage_bytes == 5 * 8 * 4


def test_encode_text_with_nan_raises_validation_error() -> None:
    """NaN vectors should fail validation."""
    generator = EmbeddingGenerator(
        EmbeddingConfig(),
        model_loader=lambda: FakeModel(produce_nan=True),
    )

    with pytest.raises(EmbeddingValidationError):
        generator.encode_text("invalid vector")


def test_encode_chunks_with_dimension_mismatch_raises() -> None:
    """Inconsistent vector dimensions should fail validation."""
    generator = EmbeddingGenerator(
        EmbeddingConfig(),
        model_loader=lambda: FakeModel(variable_dim=True),
    )

    with pytest.raises(EmbeddingValidationError):
        generator.encode_chunks(_sample_chunks(2), show_progress=False)


def test_empty_chunk_text_raises_error() -> None:
    """Empty chunk text should raise an embedding error."""
    chunks = [
        {
            "chunk_id": "empty",
            "document_id": "doc",
            "text": "   ",
            "metadata": {},
        }
    ]

    generator = EmbeddingGenerator(
        EmbeddingConfig(),
        model_loader=lambda: FakeModel(),
    )

    with pytest.raises(EmbeddingError):
        generator.encode_chunks(chunks, show_progress=False)


def test_export_npy(tmp_path) -> None:
    """NumPy export should create vector and metadata files."""
    generator = EmbeddingGenerator(
        EmbeddingConfig(),
        model_loader=lambda: FakeModel(dim=8),
    )
    records = generator.encode_chunks(_sample_chunks(3), show_progress=False)

    exporter = EmbeddingExporter(
        EmbeddingConfig(),
        output_dir=tmp_path,
    )
    path = exporter.export(records, file_name="embeddings.npy", export_format="npy")

    assert path.exists()

    loaded = np.load(path)
    assert loaded.shape == (3, 8)

    metadata_path = path.with_suffix(".json")
    assert metadata_path.exists()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert len(metadata) == 3


def test_export_pickle(tmp_path) -> None:
    """Pickle export should create a loadable file."""
    generator = EmbeddingGenerator(
        EmbeddingConfig(),
        model_loader=lambda: FakeModel(dim=8),
    )
    records = generator.encode_chunks(_sample_chunks(2), show_progress=False)

    exporter = EmbeddingExporter(
        EmbeddingConfig(),
        output_dir=tmp_path,
    )
    path = exporter.export(
        records, file_name="embeddings.pickle", export_format="pickle"
    )

    assert path.exists()


def test_export_parquet(tmp_path) -> None:
    """Parquet export should work when pandas/pyarrow are available."""
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    generator = EmbeddingGenerator(
        EmbeddingConfig(),
        model_loader=lambda: FakeModel(dim=8),
    )
    records = generator.encode_chunks(_sample_chunks(2), show_progress=False)

    exporter = EmbeddingExporter(
        EmbeddingConfig(),
        output_dir=tmp_path,
    )
    path = exporter.export(
        records, file_name="embeddings.parquet", export_format="parquet"
    )

    assert path.exists()


def test_export_json(tmp_path) -> None:
    """JSON export should include records and embeddings."""
    generator = EmbeddingGenerator(
        EmbeddingConfig(),
        model_loader=lambda: FakeModel(dim=8),
    )
    records = generator.encode_chunks(_sample_chunks(2), show_progress=False)

    exporter = EmbeddingExporter(
        EmbeddingConfig(),
        output_dir=tmp_path,
    )
    path = exporter.export(records, file_name="embeddings.json", export_format="json")

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["total_embeddings"] == 2
    assert len(payload["records"]) == 2
    assert len(payload["records"][0]["embedding"]) == 8


def test_export_empty_records_raises_error(tmp_path) -> None:
    """Exporting no records should raise an export error."""
    exporter = EmbeddingExporter(
        EmbeddingConfig(),
        output_dir=tmp_path,
    )

    with pytest.raises(EmbeddingExportError):
        exporter.export([], export_format="npy")
