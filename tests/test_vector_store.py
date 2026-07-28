"""Tests for the vector store module."""

from __future__ import annotations

import numpy as np
import pytest
from src.config import VectorStoreConfig
from src.embeddings import EmbeddingRecord
from src.vector_store import (
    VectorHit,
    VectorStoreConnectionError,
    VectorStoreError,
    VectorStoreManager,
    VectorStoreValidationError,
)

pytest.importorskip("chromadb")


def _config(
    tmp_path: object = None,
    name: str = "test_col",
) -> VectorStoreConfig:
    """Build an in-memory (or tmp-path) vector store config for tests.

    Args:
        tmp_path: Optional directory for a persistent store; None = ephemeral.
        name: Collection name.

    Returns:
        A `VectorStoreConfig` with a tiny batch size to exercise batching.
    """
    return VectorStoreConfig(
        persist_directory=tmp_path,  # type: ignore[arg-type]
        collection_name=name,
        hnsw_space="cosine",
        add_batch_size=2,
    )


def _make_records(count: int, dim: int = 8, seed: int = 0) -> list[EmbeddingRecord]:
    """Build deterministic, unit-normalised embedding records.

    Args:
        count: Number of records.
        dim: Embedding dimension.
        seed: RNG seed for reproducibility.

    Returns:
        A list of `EmbeddingRecord` objects.
    """
    rng = np.random.default_rng(seed)
    records: list[EmbeddingRecord] = []
    for index in range(count):
        vector = rng.standard_normal(dim).astype(np.float32)
        vector = vector / (np.linalg.norm(vector) + 1e-9)
        records.append(
            EmbeddingRecord(
                chunk_id=f"c{index}",
                document_id=f"d{index}",
                embedding=vector,
                dimension=dim,
                model_name="test-model",
                created_at="2026-01-01T00:00:00+00:00",
                metadata={"idx": index, "title": f"title_{index}"},
            )
        )
    return records


def test_exception_hierarchy() -> None:
    """Custom exceptions inherit from the base vector store error."""
    assert issubclass(VectorStoreValidationError, VectorStoreError)
    assert issubclass(VectorStoreConnectionError, VectorStoreError)


def test_invalid_config_raises_validation_error() -> None:
    """An invalid configuration is rejected at construction time."""
    with pytest.raises(VectorStoreValidationError):
        VectorStoreManager({"collection_name": ""})

    with pytest.raises(VectorStoreValidationError):
        VectorStoreManager({"hnsw_space": "bogus"})


def test_ephemeral_connect_and_count() -> None:
    """An ephemeral store connects without I/O and starts empty."""
    manager = VectorStoreManager(_config(name="eph"))
    manager.connect()

    assert manager.is_connected
    assert manager.count() == 0


def test_cosine_space_is_set_on_collection() -> None:
    """The collection is created with a cosine distance space."""
    manager = VectorStoreManager(_config(name="cos"))
    manager.connect()

    assert manager.collection.metadata.get("hnsw:space") == "cosine"


def test_add_records_and_query_ranking() -> None:
    """Querying with a stored vector ranks that vector first."""
    records = _make_records(3)
    documents = [f"text {i}" for i in range(3)]
    manager = VectorStoreManager(_config(name="rank"))

    added = manager.add_records(records, documents)

    assert added == 3
    assert manager.count() == 3

    hits = manager.query(records[0].embedding, top_k=3)

    assert isinstance(hits[0], VectorHit)
    assert hits[0].id == "c0"
    assert hits[0].document == "text 0"
    assert hits[0].similarity > 0.99
    assert hits[0].metadata["title"] == "title_0"


def test_add_embeddings_direct_with_batching() -> None:
    """Direct vector insertion works and respects the batch size."""
    vectors = [
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
    ]
    manager = VectorStoreManager(_config(name="batch"))

    added = manager.add_embeddings(
        ids=["a", "b", "c"],
        embeddings=vectors,
        documents=["x", "y", "z"],
    )

    assert added == 3
    assert manager.count() == 3


def test_query_top_k_limits_results() -> None:
    """The number of returned hits never exceeds top_k."""
    records = _make_records(3)
    manager = VectorStoreManager(_config(name="topk"))
    manager.add_records(records, [f"t{i}" for i in range(3)])

    hits = manager.query(records[0].embedding, top_k=2)

    assert len(hits) == 2


def test_query_empty_collection_returns_empty() -> None:
    """Querying an empty collection returns no hits instead of raising."""
    manager = VectorStoreManager(_config(name="empty"))
    manager.connect()

    assert manager.query(np.ones(8, dtype=np.float32), top_k=3) == []


def test_query_invalid_top_k_raises() -> None:
    """A non-positive top_k is rejected."""
    manager = VectorStoreManager(_config(name="badk"))
    manager.connect()

    with pytest.raises(VectorStoreValidationError):
        manager.query(np.ones(8, dtype=np.float32), top_k=0)


def test_add_mismatched_lengths_raises() -> None:
    """Mismatched ids / documents lengths are rejected."""
    manager = VectorStoreManager(_config(name="mismatch"))

    with pytest.raises(VectorStoreValidationError):
        manager.add_embeddings(
            ids=["a"],
            embeddings=[np.ones(4, dtype=np.float32)],
            documents=["x", "y"],
        )


def test_metadata_nested_values_are_sanitised() -> None:
    """Nested / None metadata values are coerced to primitives."""
    record = EmbeddingRecord(
        chunk_id="m1",
        document_id="d1",
        embedding=np.ones(4, dtype=np.float32),
        dimension=4,
        model_name="test-model",
        created_at="2026-01-01T00:00:00+00:00",
        metadata={
            "nested": {"a": 1},
            "lst": [1, 2],
            "none": None,
            "ok": "value",
        },
    )
    manager = VectorStoreManager(_config(name="meta"))
    manager.add_records([record], ["the text"])

    hit = manager.query(record.embedding, top_k=1)[0]

    assert hit.metadata["nested"] == '{"a": 1}'
    assert hit.metadata["lst"] == "[1, 2]"
    assert hit.metadata["none"] == ""
    assert hit.metadata["ok"] == "value"


def test_persistent_client_creates_directory(tmp_path) -> None:
    """A persistent store creates its directory on connect."""
    target = tmp_path / "chroma_db"
    manager = VectorStoreManager(_config(tmp_path=target, name="persist"))
    manager.connect()

    assert target.exists()
    assert target.is_dir()


def test_delete_collection_empties_store() -> None:
    """Deleting the collection removes every stored vector."""
    records = _make_records(3)
    manager = VectorStoreManager(_config(name="del"))
    manager.add_records(records, [f"t{i}" for i in range(3)])
    assert manager.count() == 3

    manager.delete_collection()

    assert manager.count() == 0


def test_reset_leaves_empty_collection() -> None:
    """Reset drops the data and recreates an empty collection."""
    records = _make_records(3)
    manager = VectorStoreManager(_config(name="reset"))
    manager.add_records(records, [f"t{i}" for i in range(3)])

    manager.reset()

    assert manager.count() == 0


def test_distance_to_similarity_cosine() -> None:
    """Cosine distances map to similarities via 1 - distance."""
    to_sim = VectorStoreManager._distance_to_similarity

    assert to_sim(0.0, "cosine") == pytest.approx(1.0)
    assert to_sim(0.25, "cosine") == pytest.approx(0.75)
    assert to_sim(0.0, "l2") == pytest.approx(0.0)
    assert to_sim(2.0, "ip") == pytest.approx(-2.0)


def test_add_records_persists_document_id() -> None:
    """add_records stores the parent document id inside the metadata (Phase 7)."""
    record = EmbeddingRecord(
        chunk_id="c0",
        document_id="parent_doc_7",
        embedding=np.ones(4, dtype=np.float32),
        dimension=4,
        model_name="test-model",
        created_at="2026-01-01T00:00:00+00:00",
        metadata={"title": "T"},
    )
    manager = VectorStoreManager(_config(name="docid"))
    manager.add_records([record], ["the text"])

    hit = manager.query(record.embedding, top_k=1)[0]

    assert hit.metadata["document_id"] == "parent_doc_7"
    assert hit.metadata["title"] == "T"
