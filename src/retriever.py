"""Similarity retrieval (Phase 7).

Title:
    Retriever Module
Description:
    Turns a natural-language question into ranked, scored context chunks by
    composing the embedding generator (Phase 5) with the vector store
    (Phase 6). The retriever owns *policy* only — how many chunks to fetch,
    how to rank them, and whether to drop low-similarity hits — while the
    heavy lifting (encoding, ANN search) stays in the lower layers.

    Every dependency is injected, so the module is fully unit-testable with
    in-memory fakes and never touches the network or a real model in tests.
Responsibilities:
    - Encode the query via an injected ``EmbeddingGenerator``.
    - Query the injected ``VectorStoreManager``.
    - Rank hits by similarity and assign stable 1-based ranks.
    - Optionally filter hits below a similarity floor.
    - Map ``VectorHit`` objects into self-describing ``RetrievedChunk``s.
    - Return an iterable ``RetrievalResult`` that also carries timing.
Author:
    Aml
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from src.config import RetrieverConfig, resolve_retriever_config
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorHit, VectorStoreManager

try:
    from src.logger import get_logger
except ImportError:  # pragma: no cover
    from logging import getLogger as get_logger


class RetrieverError(Exception):
    """Base exception for retriever errors."""


class RetrieverValidationError(RetrieverError):
    """Raised when retriever inputs or configuration are invalid."""


class RetrieverSearchError(RetrieverError):
    """Raised when the underlying embedding or vector-store call fails."""


_CHUNK_MARKER = "_chunk_"


@dataclass(frozen=True)
class RetrievedChunk:
    """A single ranked chunk returned by the retriever.

    Attributes:
        chunk_id: Identifier of the stored vector / source chunk.
        document_id: Identifier of the parent document the chunk came from.
        text: Chunk text used as retrieval evidence.
        metadata: Stored metadata mapping for the chunk.
        similarity: Similarity score between the query and the chunk.
        distance: Raw distance returned by the vector store.
        rank: 1-based position in the ranked result (most similar = 1).
    """

    chunk_id: str
    document_id: str
    text: str
    metadata: dict[str, Any]
    similarity: float
    distance: float
    rank: int

    def to_dict(self) -> dict[str, Any]:
        """Convert the chunk to a plain dictionary.

        Returns:
            Dictionary representation of the chunk.
        """
        return asdict(self)


@dataclass(frozen=True)
class RetrievalResult:
    """The full, timed outcome of a single retrieval call.

    The result behaves like a sequence of :class:`RetrievedChunk` (it supports
    ``iter``, ``len`` and integer indexing) so consumers that only need the
    chunks can treat it as a list, while callers that need diagnostics can
    read ``query``, ``top_k`` and ``retrieval_time_sec``.

    Attributes:
        query: The original query string.
        chunks: Ranked chunks, most similar first.
        top_k: The ``top_k`` value that was actually applied.
        retrieval_time_sec: Wall-clock time spent encoding + searching.
    """

    query: str
    chunks: tuple[RetrievedChunk, ...]
    top_k: int
    retrieval_time_sec: float

    def __iter__(self) -> Iterator[RetrievedChunk]:
        """Iterate over the ranked chunks.

        Yields:
            Each :class:`RetrievedChunk` in rank order.
        """
        return iter(self.chunks)

    def __len__(self) -> int:
        """Return the number of retrieved chunks.

        Returns:
            Chunk count.
        """
        return len(self.chunks)

    def __getitem__(self, index: int) -> RetrievedChunk:
        """Return the chunk at a rank position (0-based).

        Args:
            index: Zero-based position.

        Returns:
            The chunk at ``index``.
        """
        return self.chunks[index]

    def to_dicts(self) -> list[dict[str, Any]]:
        """Convert every chunk to a dictionary.

        Returns:
            A list of chunk dictionaries in rank order.
        """
        return [chunk.to_dict() for chunk in self.chunks]


def _document_id_from_chunk_id(chunk_id: str) -> str:
    """Best-effort parent-document id extracted from a chunk id.

    The chunker (Phase 4) names chunks ``"{document_id}_chunk_NNNN"``; this
    reverses that convention as a safety net for stores whose metadata does
    not carry ``document_id`` explicitly.

    Args:
        chunk_id: Stored vector identifier.

    Returns:
        The inferred document id, or ``chunk_id`` unchanged when the marker
        is absent.
    """
    if _CHUNK_MARKER in chunk_id:
        return chunk_id.rsplit(_CHUNK_MARKER, 1)[0]
    return chunk_id


class Retriever:
    """Retrieves ranked, scored chunks for a query.

    The vector store is mandatory (the retriever has nothing to search
    otherwise) and is validated eagerly. The embedding generator is optional:
    when omitted, a default one is built lazily from the embedding config the
    first time a query runs.

    Attributes:
        config: Resolved retriever configuration.
    """

    def __init__(
        self,
        vector_store: VectorStoreManager | None = None,
        embedding_generator: EmbeddingGenerator | None = None,
        config: RetrieverConfig | Any | None = None,
        top_k: int | None = None,
        min_similarity: float | None = None,
    ) -> None:
        """Initialise the retriever.

        Args:
            vector_store: The vector store to search (required).
            embedding_generator: Optional query encoder; built lazily if None.
            config: Optional retriever configuration.
            top_k: Optional default ``top_k``; falls back to ``config.top_k``.
            min_similarity: Optional default similarity floor; falls back to
                ``config.min_similarity``.

        Raises:
            RetrieverValidationError: If the configuration is invalid or no
                vector store is supplied.
        """
        try:
            self._config = resolve_retriever_config(config)
        except (TypeError, ValueError, AttributeError) as exc:
            raise RetrieverValidationError("Invalid retriever configuration.") from exc

        if vector_store is None:
            raise RetrieverValidationError(
                "A VectorStoreManager instance is required to build a Retriever."
            )

        self._vector_store = vector_store
        self._embedding_generator = embedding_generator
        self._top_k = int(top_k) if top_k is not None else self._config.top_k
        self._min_similarity = (
            float(min_similarity)
            if min_similarity is not None
            else self._config.min_similarity
        )
        self._logger = get_logger("retriever")

    # ------------------------------------------------------------- config
    @property
    def config(self) -> RetrieverConfig:
        """Return the active configuration.

        Returns:
            Active `RetrieverConfig`.
        """
        return self._config

    @property
    def top_k(self) -> int:
        """Return the default ``top_k`` used when none is passed to retrieve.

        Returns:
            Default top-k value.
        """
        return self._top_k

    @property
    def min_similarity(self) -> float | None:
        """Return the default similarity floor (None when disabled).

        Returns:
            Similarity floor or None.
        """
        return self._min_similarity

    @property
    def embedding_generator(self) -> EmbeddingGenerator:
        """Return the query encoder, building a default one lazily.

        Returns:
            The active `EmbeddingGenerator`.
        """
        if self._embedding_generator is None:
            self._embedding_generator = EmbeddingGenerator()
            self._logger.info("Built a default EmbeddingGenerator for the retriever.")
        return self._embedding_generator

    # -------------------------------------------------------------- search
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        where: Mapping[str, Any] | None = None,
        min_similarity: float | None = None,
    ) -> RetrievalResult:
        """Retrieve ranked chunks for a query.

        Args:
            query: Natural-language question or search query.
            top_k: Optional override for the number of chunks to fetch.
            where: Optional metadata filter forwarded to the vector store.
            min_similarity: Optional override for the similarity floor.

        Returns:
            A :class:`RetrievalResult` (iterable over ranked chunks).

        Raises:
            RetrieverValidationError: If the query or ``top_k`` is invalid.
            RetrieverSearchError: If encoding or searching fails.
        """
        if not isinstance(query, str) or not query.strip():
            raise RetrieverValidationError("query must be a non-empty string.")

        resolved_top_k = int(top_k) if top_k is not None else self._top_k
        if resolved_top_k <= 0:
            raise RetrieverValidationError("top_k must be greater than zero.")

        resolved_min = (
            float(min_similarity)
            if min_similarity is not None
            else self._min_similarity
        )

        start = time.perf_counter()
        try:
            generator = self.embedding_generator
            query_vector = generator.encode_text(query)
            hits = self._vector_store.query(
                query_vector,
                top_k=resolved_top_k,
                where=where,
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Retrieval failed while encoding or searching.")
            raise RetrieverSearchError(
                "Retrieval failed while encoding or searching."
            ) from exc
        elapsed = time.perf_counter() - start

        # Defensive re-rank: do not trust the store ordering blindly.
        ordered = sorted(hits, key=lambda hit: hit.similarity, reverse=True)

        if resolved_min is not None:
            ordered = [hit for hit in ordered if hit.similarity >= resolved_min]

        chunks = tuple(
            self._to_retrieved_chunk(hit, rank=index + 1)
            for index, hit in enumerate(ordered)
        )

        self._logger.info(
            "Retrieved %d chunk(s) for query (top_k=%d, min_sim=%s, %.4fs).",
            len(chunks),
            resolved_top_k,
            resolved_min,
            elapsed,
        )

        return RetrievalResult(
            query=query,
            chunks=chunks,
            top_k=resolved_top_k,
            retrieval_time_sec=round(elapsed, 6),
        )

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _to_retrieved_chunk(hit: VectorHit, rank: int) -> RetrievedChunk:
        """Map a vector-store hit into a self-describing retrieved chunk.

        Args:
            hit: Raw similarity hit.
            rank: 1-based rank to assign.

        Returns:
            A :class:`RetrievedChunk`.
        """
        metadata = dict(hit.metadata)
        document_id = str(
            metadata.get("document_id")
            or metadata.get("doc_id")
            or _document_id_from_chunk_id(hit.id)
        )
        return RetrievedChunk(
            chunk_id=hit.id,
            document_id=document_id,
            text=hit.document,
            metadata=metadata,
            similarity=hit.similarity,
            distance=hit.distance,
            rank=rank,
        )
