"""Chroma vector store management (Phase 6).

Title:
    Vector Store Module
Description:
    Owns the ChromaDB lifecycle for the RAG project: creating / loading a
    cosine collection, persisting the embeddings produced by Phase 5, and
    serving similarity queries to the retriever (Phase 7).

    The module talks to ``chromadb`` directly (not via LangChain) so the
    ``src`` package stays framework-agnostic, and so the already-computed
    ``EmbeddingRecord`` vectors are stored as-is instead of being re-encoded.
    ``chromadb`` is imported lazily, therefore importing this module never
    fails even when the optional dependency is absent.
Responsibilities:
    - Create or load a persistent / in-memory Chroma collection.
    - Guarantee a cosine distance space at creation time.
    - Persist pre-computed embeddings in batches.
    - Sanitise metadata to the primitive types Chroma accepts.
    - Run similarity queries and convert distances to similarities.
    - Expose lifecycle operations (count / delete / reset).
Author:
    Aml
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from src.config import VectorStoreConfig, resolve_vector_store_config
from src.embeddings import EmbeddingRecord

try:
    from src.logger import get_logger
except ImportError:  # pragma: no cover
    from logging import getLogger as get_logger


class VectorStoreError(Exception):
    """Base exception for vector store errors."""


class VectorStoreConnectionError(VectorStoreError):
    """Raised when the Chroma client or collection cannot be created."""


class VectorStoreValidationError(VectorStoreError):
    """Raised when inputs to the vector store fail validation."""


@dataclass(frozen=True)
class VectorHit:
    """A single result returned by a similarity query.

    Attributes:
        id: Stored vector identifier (the originating chunk id).
        document: Stored text of the chunk.
        metadata: Stored metadata mapping (primitive values only).
        distance: Raw distance returned by Chroma.
        similarity: Similarity derived from the distance (meaningful for
            the cosine space; an ordering proxy for other spaces).
    """

    id: str
    document: str
    metadata: dict[str, Any]
    distance: float
    similarity: float

    def to_dict(self) -> dict[str, Any]:
        """Convert the hit to a plain dictionary.

        Returns:
            Dictionary representation of the hit.
        """
        return asdict(self)


def _coerce_meta_value(value: Any) -> Any:
    """Coerce a metadata value to a Chroma-compatible primitive.

    Args:
        value: Raw metadata value.

    Returns:
        A Chroma-compatible primitive value.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


class VectorStoreManager:
    """Manages a Chroma collection and the vectors stored inside it.

    Attributes:
        config: Resolved vector store configuration.
    """

    def __init__(
        self,
        config: VectorStoreConfig | Any | None = None,
    ) -> None:
        """Initialise the manager without touching the database.

        Args:
            config: Optional configuration object, mapping or None.

        Raises:
            VectorStoreValidationError: If the configuration is invalid.
        """
        try:
            self._config = resolve_vector_store_config(config)
        except (TypeError, ValueError, AttributeError) as exc:
            raise VectorStoreValidationError(
                "Invalid vector store configuration."
            ) from exc

        self._client: Any | None = None
        self._collection: Any | None = None
        self._logger = get_logger("vector_store")

    @property
    def config(self) -> VectorStoreConfig:
        """Return the active configuration.

        Returns:
            Active `VectorStoreConfig`.
        """
        return self._config

    @property
    def is_connected(self) -> bool:
        """Return whether a Chroma client has been created.

        Returns:
            True once the client exists.
        """
        return self._client is not None

    @property
    def client(self) -> Any:
        """Return the Chroma client, creating it lazily.

        Returns:
            The underlying Chroma client.
        """
        return self._ensure_client()

    @property
    def collection(self) -> Any:
        """Return the Chroma collection, creating it lazily.

        Returns:
            The underlying Chroma collection.
        """
        return self._ensure_collection()

    def connect(self) -> VectorStoreManager:
        """Create the client and collection if they do not exist yet.

        Returns:
            ``self``, for fluent chaining.
        """
        self._ensure_collection()
        return self

    def count(self) -> int:
        """Return the number of vectors currently stored.

        Returns:
            Vector count (0 when the collection is empty).
        """
        return int(self._ensure_collection().count())

    def delete_collection(self) -> None:
        """Delete the managed collection from the client.

        Raises:
            VectorStoreError: If Chroma cannot delete the collection.
        """
        if self._client is None:
            return

        name = self._config.collection_name
        try:
            self._client.delete_collection(name)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Failed to delete collection '{name}'.") from exc

        self._collection = None
        self._logger.info("Deleted collection '%s'.", name)

    def reset(self) -> None:
        """Delete and recreate the collection, leaving it empty.

        Raises:
            VectorStoreError: If the collection cannot be reset.
        """
        self.delete_collection()
        self._ensure_collection()
        self._logger.info("Reset collection '%s'.", self._config.collection_name)

    def add_embeddings(
        self,
        ids: Sequence[str],
        embeddings: Sequence[np.ndarray] | np.ndarray,
        documents: Sequence[str],
        metadatas: Sequence[Mapping[str, Any]] | None = None,
    ) -> int:
        """Store vectors together with their texts and metadata.

        Args:
            ids: Unique identifier per vector (usually chunk ids).
            embeddings: Dense vectors, either a 2-D array or a sequence of
                1-D arrays, all sharing the same dimension.
            documents: Text stored alongside each vector.
            metadatas: Optional metadata mapping per vector.

        Returns:
            The number of vectors added (0 for an empty input).

        Raises:
            VectorStoreValidationError: If the inputs are inconsistent or
                contain non-finite values.
            VectorStoreError: If Chroma rejects a batch.
        """
        count = len(ids)
        if count == 0:
            return 0

        if len(documents) != count:
            raise VectorStoreValidationError(
                "ids, embeddings and documents must share the same length."
            )

        if metadatas is None:
            resolved_metas: list[Mapping[str, Any]] = [{} for _ in range(count)]
        else:
            resolved_metas = list(metadatas)

        if len(resolved_metas) != count:
            raise VectorStoreValidationError(
                "metadatas length must match the number of ids."
            )

        self._validate_ids(ids)
        vector_rows = self._normalize_embeddings(embeddings, expected_count=count)
        clean_metas = [self._sanitize_metadata(m) for m in resolved_metas]

        collection = self._ensure_collection()
        batch_size = self._config.add_batch_size
        added = 0

        for start in range(0, count, batch_size):
            end = min(start + batch_size, count)
            add_kwargs: dict[str, Any] = {
                "ids": list(ids[start:end]),
                "embeddings": vector_rows[start:end],
                "documents": list(documents[start:end]),
            }
            batch_metas = clean_metas[start:end]
            if any(batch_metas):
                add_kwargs["metadatas"] = [
                    meta if meta else None for meta in batch_metas
                ]
            try:
                collection.add(**add_kwargs)
            except Exception as exc:  # noqa: BLE001
                raise VectorStoreError(
                    "Failed to add an embeddings batch to Chroma."
                ) from exc
            added += end - start

        self._logger.info(
            "Added %d vector(s) to collection '%s' (total %d).",
            added,
            self._config.collection_name,
            self.count(),
        )
        return added

    def add_records(
        self,
        records: Sequence[EmbeddingRecord],
        documents: Sequence[str],
    ) -> int:
        """Store embedding records together with their source texts.

        Args:
            records: Embedding records produced by Phase 5.
            documents: Text of each record, in the same order.

        Returns:
            The number of vectors added.

        Raises:
            VectorStoreValidationError: If the two sequences differ in length.
            VectorStoreError: If Chroma rejects a batch.
        """
        if len(records) != len(documents):
            raise VectorStoreValidationError(
                "records and documents must have the same length."
            )
        if not records:
            return 0

        ids = [record.chunk_id for record in records]
        embeddings = [record.embedding for record in records]
        # Phase 7: persist the parent document id alongside the chunk metadata
        # so the retriever / UI / citations can trace a hit back to its source
        # document without parsing the chunk id. ``setdefault`` never overrides
        # a value the caller already provided, so this is purely additive.
        metadatas: list[dict[str, Any]] = []
        for record in records:
            meta = dict(record.metadata)
            meta.setdefault("document_id", record.document_id)
            metadatas.append(meta)
        return self.add_embeddings(
            ids=ids,
            embeddings=embeddings,
            documents=list(documents),
            metadatas=metadatas,
        )

    def query(
        self,
        query_embedding: np.ndarray | Sequence[float],
        top_k: int = 4,
        where: Mapping[str, Any] | None = None,
    ) -> list[VectorHit]:
        """Run a similarity search and return ranked hits.

        Args:
            query_embedding: Dense query vector.
            top_k: Maximum number of hits to return (must be positive).
            where: Optional Chroma metadata filter.

        Returns:
            Hits ordered by similarity (empty when the collection is empty).

        Raises:
            VectorStoreValidationError: If ``top_k`` or the query vector is
                invalid.
            VectorStoreError: If the query fails.
        """
        if top_k <= 0:
            raise VectorStoreValidationError("top_k must be greater than zero.")

        collection = self._ensure_collection()
        if collection.count() == 0:
            return []

        query_vector = np.asarray(query_embedding, dtype=np.float32)
        if query_vector.ndim != 1 or query_vector.size == 0:
            raise VectorStoreValidationError(
                "query_embedding must be a non-empty 1-D vector."
            )
        if not np.all(np.isfinite(query_vector)):
            raise VectorStoreValidationError(
                "query_embedding contains NaN or infinite values."
            )

        kwargs: dict[str, Any] = {
            "query_embeddings": query_vector.reshape(1, -1).tolist(),
            "n_results": int(top_k),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = dict(where)

        try:
            raw = collection.query(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError("Similarity query failed.") from exc

        return self._parse_query_results(raw)

    def _ensure_client(self) -> Any:
        """Create the Chroma client lazily.

        Returns:
            The Chroma client (persistent or ephemeral).

        Raises:
            VectorStoreConnectionError: If chromadb is missing or the client
                cannot be created.
        """
        if self._client is not None:
            return self._client

        try:
            import chromadb
        except ImportError as exc:
            raise VectorStoreConnectionError(
                "chromadb is required for the vector store. "
                "Install the project requirements first."
            ) from exc

        persist = self._config.persist_directory
        try:
            if persist is None:
                self._client = chromadb.EphemeralClient()
            else:
                persist.mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(path=str(persist))
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreConnectionError(
                "Failed to create the Chroma client."
            ) from exc

        mode = "ephemeral" if persist is None else f"persistent:{persist}"
        self._logger.info("Chroma client ready (%s).", mode)
        return self._client

    def _ensure_collection(self) -> Any:
        """Create or load the collection with a cosine distance space.

        Returns:
            The Chroma collection.

        Raises:
            VectorStoreConnectionError: If the collection cannot be obtained.
        """
        if self._collection is not None:
            return self._collection

        client = self._ensure_client()
        try:
            self._collection = client.get_or_create_collection(
                name=self._config.collection_name,
                metadata={"hnsw:space": self._config.hnsw_space},
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreConnectionError(
                f"Failed to get/create collection '{self._config.collection_name}'."
            ) from exc

        self._logger.info(
            "Collection '%s' ready (space=%s, count=%d).",
            self._config.collection_name,
            self._config.hnsw_space,
            self._collection.count(),
        )
        return self._collection

    @staticmethod
    def _validate_ids(ids: Sequence[str]) -> None:
        """Ensure every id is a non-empty string.

        Args:
            ids: Candidate identifiers.

        Raises:
            VectorStoreValidationError: If any id is invalid.
        """
        for index, value in enumerate(ids):
            if not isinstance(value, str) or not value.strip():
                raise VectorStoreValidationError(
                    f"Vector id at index {index} must be a non-empty string."
                )

    @staticmethod
    def _normalize_embeddings(
        embeddings: Sequence[np.ndarray] | np.ndarray,
        expected_count: int,
    ) -> list[list[float]]:
        """Normalise embeddings into a list of equal-length float rows.

        Args:
            embeddings: A 2-D array or a sequence of 1-D vectors.
            expected_count: Required number of vectors.

        Returns:
            A list of float lists, one per vector.

        Raises:
            VectorStoreValidationError: On shape / count / value problems.
        """
        if isinstance(embeddings, np.ndarray):
            array = embeddings
        else:
            rows = [
                np.asarray(vector, dtype=np.float32).reshape(1, -1)
                for vector in embeddings
            ]
            array = np.vstack(rows) if rows else np.empty((0, 0), dtype=np.float32)

        if array.ndim != 2:
            raise VectorStoreValidationError(
                "Embeddings must form a two-dimensional array."
            )
        if array.shape[0] != expected_count:
            raise VectorStoreValidationError(
                "Embedding count does not match the number of ids."
            )
        if array.size > 0 and not np.all(np.isfinite(array)):
            raise VectorStoreValidationError(
                "Embeddings contain NaN or infinite values."
            )

        return array.astype(np.float32, copy=False).tolist()

    @staticmethod
    def _sanitize_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
        """Flatten metadata to Chroma-compatible primitive values.

        Args:
            metadata: Raw metadata mapping.

        Returns:
            A sanitised mapping with string keys and primitive values.
        """
        if not isinstance(metadata, Mapping):
            return {}

        clean: dict[str, Any] = {}
        for key, value in metadata.items():
            if not isinstance(key, str) or not key:
                continue
            clean[key] = _coerce_meta_value(value)
        return clean

    @staticmethod
    def _distance_to_similarity(distance: float, space: str) -> float:
        """Convert a Chroma distance into a similarity score.

        Args:
            distance: Raw distance value.
            space: Distance space name.

        Returns:
            A similarity score.
        """
        value = float(distance)
        if space == "cosine":
            similarity = 1.0 - value
            return float(max(-1.0, min(1.0, similarity)))
        return float(-value)

    def _parse_query_results(self, raw: Mapping[str, Any]) -> list[VectorHit]:
        """Parse a raw Chroma query response into `VectorHit` objects.

        Args:
            raw: Mapping returned by ``collection.query``.

        Returns:
            A list of hits for the single issued query.
        """
        ids = raw.get("ids") or [[]]
        documents = raw.get("documents") or [[]]
        metadatas = raw.get("metadatas") or [[]]
        distances = raw.get("distances") or [[]]

        row_ids = ids[0] if ids else []
        row_docs = documents[0] if documents else []
        row_metas = metadatas[0] if metadatas else []
        row_dists = distances[0] if distances else []

        hits: list[VectorHit] = []
        for index, hit_id in enumerate(row_ids):
            raw_distance = (
                row_dists[index]
                if index < len(row_dists) and row_dists[index] is not None
                else 0.0
            )
            distance = float(raw_distance)
            raw_meta = (
                row_metas[index]
                if index < len(row_metas) and row_metas[index] is not None
                else {}
            )
            raw_doc = (
                row_docs[index]
                if index < len(row_docs) and row_docs[index] is not None
                else ""
            )
            hits.append(
                VectorHit(
                    id=str(hit_id),
                    document=str(raw_doc),
                    metadata=dict(raw_meta),
                    distance=distance,
                    similarity=self._distance_to_similarity(
                        distance, self._config.hnsw_space
                    ),
                )
            )
        return hits
