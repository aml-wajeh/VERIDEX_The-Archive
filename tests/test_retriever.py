"""Tests for the retriever module."""

from __future__ import annotations

import numpy as np
import pytest
from src.config import RetrieverConfig
from src.retriever import (
    RetrievalResult,
    RetrievedChunk,
    Retriever,
    RetrieverSearchError,
    RetrieverValidationError,
)
from src.vector_store import VectorHit


class FakeEmbeddingGenerator:
    """Deterministic stand-in for `EmbeddingGenerator` (no model, no network)."""

    def __init__(self, dim: int = 8) -> None:
        """Initialise the fake encoder.

        Args:
            dim: Dimension of the produced vectors.
        """
        self.dim = dim
        self.calls = 0
        self.last_text: str | None = None

    def encode_text(self, text: str) -> np.ndarray:
        """Return a deterministic vector derived from the text length.

        Args:
            text: Query text.

        Returns:
            A 1-D float32 vector.
        """
        self.calls += 1
        self.last_text = text
        return np.full(self.dim, float(len(text) % 7), dtype=np.float32)


class FakeVectorStore:
    """Stand-in for `VectorStoreManager` that returns pre-built hits."""

    def __init__(self, hits: list[VectorHit]) -> None:
        """Initialise the fake store.

        Args:
            hits: Hits returned (sliced by ``top_k``) on every query.
        """
        self._hits = hits
        self.last_top_k: int | None = None
        self.last_where: object = None
        self.query_calls = 0

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 4,
        where: object = None,
    ) -> list[VectorHit]:
        """Return the pre-built hits, honouring ``top_k``.

        Args:
            query_embedding: Ignored.
            top_k: Maximum hits to return.
            where: Recorded for assertion.

        Returns:
            A slice of the pre-built hits.
        """
        self.query_calls += 1
        self.last_top_k = top_k
        self.last_where = where
        return self._hits[:top_k]


def _hit(
    chunk_id: str,
    similarity: float,
    *,
    document_id: str | None = None,
    text: str = "evidence text",
    title: str = "Title",
) -> VectorHit:
    """Build a `VectorHit` with a consistent distance/similarity pair.

    Args:
        chunk_id: Stored vector id.
        similarity: Similarity score.
        document_id: Optional explicit document id stored in metadata.
        text: Stored chunk text.
        title: Title stored in metadata.

    Returns:
        A `VectorHit`.
    """
    metadata: dict[str, object] = {"title": title}
    if document_id is not None:
        metadata["document_id"] = document_id
    return VectorHit(
        id=chunk_id,
        document=text,
        metadata=metadata,
        distance=round(1.0 - similarity, 4),
        similarity=similarity,
    )


def _retriever(
    hits: list[VectorHit],
    *,
    top_k: int | None = None,
    min_similarity: float | None = None,
    config: RetrieverConfig | None = None,
) -> tuple[Retriever, FakeVectorStore, FakeEmbeddingGenerator]:
    """Build a retriever wired to fakes.

    Args:
        hits: Hits the fake store returns.
        top_k: Optional constructor top-k.
        min_similarity: Optional constructor similarity floor.
        config: Optional explicit config.

    Returns:
        A tuple of (retriever, fake store, fake encoder).
    """
    store = FakeVectorStore(hits)
    encoder = FakeEmbeddingGenerator()
    retriever = Retriever(
        vector_store=store,  # type: ignore[arg-type]
        embedding_generator=encoder,  # type: ignore[arg-type]
        config=config,
        top_k=top_k,
        min_similarity=min_similarity,
    )
    return retriever, store, encoder


def test_retrieve_requires_vector_store() -> None:
    """Building a retriever without a vector store is rejected eagerly."""
    with pytest.raises(RetrieverValidationError):
        Retriever()


def test_retrieve_returns_chunks_ranked_by_similarity() -> None:
    """Hits are re-ranked by similarity and assigned 1-based ranks."""
    hits = [
        _hit("c1", 0.50, document_id="d1"),
        _hit("c2", 0.90, document_id="d2"),
        _hit("c3", 0.70, document_id="d3"),
    ]
    retriever, _, _ = _retriever(hits)

    result = retriever.retrieve("who is beyonce?")

    assert isinstance(result, RetrievalResult)
    assert [chunk.chunk_id for chunk in result] == ["c2", "c3", "c1"]
    assert [chunk.rank for chunk in result] == [1, 2, 3]
    assert result[0].similarity == pytest.approx(0.90)


def test_retrieve_respects_top_k() -> None:
    """The number of returned chunks never exceeds top_k."""
    hits = [_hit(f"c{i}", 0.9 - i * 0.1, document_id=f"d{i}") for i in range(5)]
    retriever, store, _ = _retriever(hits, top_k=2)

    result = retriever.retrieve("query")

    assert len(result) == 2
    assert store.last_top_k == 2


def test_retrieve_top_k_override_wins() -> None:
    """A per-call top_k overrides the constructor default."""
    hits = [_hit(f"c{i}", 0.9 - i * 0.1, document_id=f"d{i}") for i in range(5)]
    retriever, store, _ = _retriever(hits, top_k=2)

    result = retriever.retrieve("query", top_k=4)

    assert len(result) == 4
    assert store.last_top_k == 4


def test_retrieve_applies_min_similarity_filter() -> None:
    """Hits below the similarity floor are dropped and ranks are renumbered."""
    hits = [
        _hit("c1", 0.90, document_id="d1"),
        _hit("c2", 0.70, document_id="d2"),
        _hit("c3", 0.50, document_id="d3"),
        _hit("c4", 0.30, document_id="d4"),
    ]
    retriever, _, _ = _retriever(hits, min_similarity=0.6)

    result = retriever.retrieve("query")

    assert [chunk.chunk_id for chunk in result] == ["c1", "c2"]
    assert [chunk.rank for chunk in result] == [1, 2]


def test_retrieve_empty_query_raises() -> None:
    """An empty or non-string query is rejected before any search."""
    retriever, store, encoder = _retriever([_hit("c1", 0.9)])

    with pytest.raises(RetrieverValidationError):
        retriever.retrieve("   ")

    assert encoder.calls == 0
    assert store.query_calls == 0


def test_retrieve_invalid_top_k_raises() -> None:
    """A non-positive top_k is rejected."""
    retriever, _, _ = _retriever([_hit("c1", 0.9)])

    with pytest.raises(RetrieverValidationError):
        retriever.retrieve("query", top_k=0)


def test_retrieve_forwards_where_filter() -> None:
    """The metadata filter is forwarded verbatim to the vector store."""
    retriever, store, _ = _retriever([_hit("c1", 0.9)])

    retriever.retrieve("query", where={"title": "Beyoncé"})

    assert store.last_where == {"title": "Beyoncé"}


def test_retrieval_result_is_iterable_len_and_indexable() -> None:
    """RetrievalResult behaves like a sequence of chunks."""
    hits = [_hit("c1", 0.9), _hit("c2", 0.8)]
    retriever, _, _ = _retriever(hits)

    result = retriever.retrieve("query")

    assert len(result) == 2
    assert list(result) == [result[0], result[1]]
    assert isinstance(result[0], RetrievedChunk)
    assert result.to_dicts()[0]["chunk_id"] == "c1"


def test_retrieve_records_non_negative_timing() -> None:
    """The result carries a non-negative retrieval time."""
    retriever, _, _ = _retriever([_hit("c1", 0.9)])

    result = retriever.retrieve("query")

    assert result.retrieval_time_sec >= 0.0
    assert result.query == "query"


def test_retrieve_maps_vector_hit_fields() -> None:
    """Hit fields are mapped onto the retrieved chunk, incl. document_id."""
    hits = [_hit("c9", 0.85, document_id="docX", text="the passage", title="T")]
    retriever, _, _ = _retriever(hits)

    chunk = retriever.retrieve("query")[0]

    assert chunk.chunk_id == "c9"
    assert chunk.document_id == "docX"
    assert chunk.text == "the passage"
    assert chunk.metadata["title"] == "T"
    assert chunk.similarity == pytest.approx(0.85)
    assert chunk.distance == pytest.approx(0.15)


def test_retrieve_derives_document_id_from_chunk_id_when_missing() -> None:
    """Without an explicit document_id, it is parsed from the chunk id."""
    hits = [_hit("doc42_chunk_0003", 0.8)]  # no document_id in metadata
    retriever, _, _ = _retriever(hits)

    chunk = retriever.retrieve("query")[0]

    assert chunk.document_id == "doc42"


def test_retrieve_uses_injected_embedding_generator() -> None:
    """The injected encoder is used exactly once per retrieve call."""
    retriever, _, encoder = _retriever([_hit("c1", 0.9)])

    retriever.retrieve("hello world")

    assert encoder.calls == 1
    assert encoder.last_text == "hello world"


def test_default_top_k_from_config() -> None:
    """When no top_k is passed, the config default is applied."""
    hits = [_hit(f"c{i}", 0.9 - i * 0.1) for i in range(5)]
    retriever, store, _ = _retriever(
        hits,
        config=RetrieverConfig(top_k=3),
    )

    result = retriever.retrieve("query")

    assert len(result) == 3
    assert store.last_top_k == 3
    assert retriever.top_k == 3


def test_invalid_config_raises_validation_error() -> None:
    """An invalid configuration is rejected at construction time."""
    store = FakeVectorStore([])
    with pytest.raises(RetrieverValidationError):
        Retriever(
            vector_store=store,  # type: ignore[arg-type]
            config={"top_k": 0},
        )


class _BrokenStore:
    """A vector store whose query always raises, to exercise error wrapping."""

    def query(
        self, query_embedding: np.ndarray, top_k: int = 4, where: object = None
    ) -> list[VectorHit]:
        """Raise to simulate a search failure.

        Returns:
            Never returns.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError("boom")


def test_retrieve_wraps_search_failure() -> None:
    """A failure in the store is surfaced as a RetrieverSearchError."""
    encoder = FakeEmbeddingGenerator()
    retriever = Retriever(
        vector_store=_BrokenStore(),  # type: ignore[arg-type]
        embedding_generator=encoder,  # type: ignore[arg-type]
    )

    with pytest.raises(RetrieverSearchError):
        retriever.retrieve("query")
