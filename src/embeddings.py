"""Embedding generation module using Sentence Transformers."""

from __future__ import annotations

import json
import math
import pickle
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.config import EmbeddingConfig, resolve_embedding_config

try:
    from src.logger import get_logger
except ImportError:  # pragma: no cover
    from logging import getLogger as get_logger


class EmbeddingError(Exception):
    """Base exception for embedding errors."""


class EmbeddingValidationError(EmbeddingError):
    """Raised when embedding validation fails."""


class EmbeddingExportError(EmbeddingError):
    """Raised when embedding export fails."""


_MODEL_CACHE: dict[tuple[str, str, str | None], Any] = {}

_TEXT_KEYS = (
    "text",
    "page_content",
    "content",
    "passage",
)
_CHUNK_ID_KEYS = (
    "chunk_id",
    "chunk_identifier",
    "id",
)
_DOCUMENT_ID_KEYS = (
    "document_id",
    "doc_id",
)
_METADATA_KEYS = (
    "metadata",
    "meta",
)


def _utc_timestamp() -> str:
    """Return current UTC timestamp in ISO format.

    Returns:
        ISO timestamp string.
    """
    return datetime.now(UTC).isoformat()


def _resolve_device(device: str | None) -> str:
    """Resolve the target device.

    Args:
        device: Requested device. If None or "auto", detect automatically.

    Returns:
        Resolved device name.
    """
    if device is None or device.strip() == "" or device.strip().lower() == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            return "cpu"

    return device


def _default_model_loader(
    model_name: str,
    device: str,
    cache_folder: Path | None,
) -> Any:
    """Load a Sentence Transformers model.

    Args:
        model_name: HuggingFace model identifier.
        device: Torch device.
        cache_folder: Optional cache directory.

    Returns:
        Loaded SentenceTransformer model.

    Raises:
        EmbeddingError: If sentence-transformers is unavailable or loading fails.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise EmbeddingError(
            "sentence-transformers is required for embedding generation."
        ) from exc

    try:
        if cache_folder is not None:
            cache_folder.mkdir(parents=True, exist_ok=True)

        return SentenceTransformer(
            model_name,
            device=device,
            cache_folder=str(cache_folder) if cache_folder else None,
        )
    except Exception as exc:  # noqa: BLE001
        raise EmbeddingError("Failed to load embedding model.") from exc


def _json_default(value: Any) -> Any:
    """Serialize NumPy values for JSON export.

    Args:
        value: Value to serialize.

    Returns:
        JSON-serializable representation.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.integer):
        return int(value)

    return str(value)


@dataclass(frozen=True)
class _ChunkParts:
    """Internal representation of extracted chunk fields.

    Attributes:
        chunk_id: Chunk identifier.
        document_id: Document identifier.
        text: Chunk text.
        metadata: Chunk metadata.
    """

    chunk_id: str
    document_id: str
    text: str
    metadata: dict[str, Any]


@dataclass
class EmbeddingRecord:
    """A single embedding record.

    Attributes:
        chunk_id: Chunk identifier.
        document_id: Parent document identifier.
        embedding: Dense vector as NumPy array.
        dimension: Embedding dimension.
        model_name: Model used to generate the embedding.
        created_at: Creation timestamp.
        metadata: Additional metadata.
    """

    chunk_id: str
    document_id: str
    embedding: np.ndarray
    dimension: int
    model_name: str
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_embedding: bool = True) -> dict[str, Any]:
        """Convert record to dictionary.

        Args:
            include_embedding: Whether to include the embedding vector.

        Returns:
            Dictionary representation.
        """
        data: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "dimension": self.dimension,
            "model_name": self.model_name,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

        if include_embedding:
            data["embedding"] = self.embedding.tolist()

        return data


@dataclass(frozen=True)
class EmbeddingStatistics:
    """Statistics for embedding generation.

    Attributes:
        total_embeddings: Number of generated embeddings.
        embedding_dimension: Embedding vector dimension.
        model_name: Model used for encoding.
        total_encoding_time_sec: Total encoding time in seconds.
        average_encoding_time_sec: Average encoding time per embedding.
        batch_count: Number of batches processed.
        memory_usage_bytes: Estimated memory usage for vectors in bytes.
    """

    total_embeddings: int
    embedding_dimension: int
    model_name: str
    total_encoding_time_sec: float
    average_encoding_time_sec: float
    batch_count: int
    memory_usage_bytes: int

    def to_dict(self) -> dict[str, Any]:
        """Convert statistics to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "total_embeddings": self.total_embeddings,
            "embedding_dimension": self.embedding_dimension,
            "model_name": self.model_name,
            "total_encoding_time_sec": self.total_encoding_time_sec,
            "average_encoding_time_sec": self.average_encoding_time_sec,
            "batch_count": self.batch_count,
            "memory_usage_bytes": self.memory_usage_bytes,
        }


def _extract_value(source: Any, keys: Sequence[str]) -> Any | None:
    """Extract the first available value from a mapping or object.

    Args:
        source: Source object.
        keys: Candidate keys or attribute names.

    Returns:
        First found value or None.
    """
    if isinstance(source, Mapping):
        for key in keys:
            if key in source:
                return source[key]
        return None

    for key in keys:
        if hasattr(source, key):
            return getattr(source, key)

    return None


def _extract_chunk_parts(chunk: Any, index: int) -> _ChunkParts:
    """Extract required fields from a chunk-like object.

    Args:
        chunk: Chunk-like object or mapping.
        index: Position used for fallback identifiers.

    Returns:
        Extracted chunk parts.

    Raises:
        EmbeddingError: If chunk text cannot be extracted.
    """
    chunk_id = _extract_value(chunk, _CHUNK_ID_KEYS)
    document_id = _extract_value(chunk, _DOCUMENT_ID_KEYS)
    text = _extract_value(chunk, _TEXT_KEYS)
    metadata = _extract_value(chunk, _METADATA_KEYS)

    if text is None:
        raise EmbeddingError(
            f"Chunk text could not be extracted for item index {index}."
        )

    if not isinstance(metadata, Mapping):
        metadata = {}

    return _ChunkParts(
        chunk_id=str(chunk_id) if chunk_id is not None else f"chunk_{index + 1}",
        document_id=str(document_id) if document_id is not None else f"doc_{index + 1}",
        text=str(text),
        metadata=dict(metadata),
    )


class EmbeddingGenerator:
    """Generates dense embeddings from chunk text.

    Attributes:
        config: Resolved embedding configuration.
    """

    def __init__(
        self,
        config: EmbeddingConfig | Any | None = None,
        model_loader: Callable[[], Any] | None = None,
    ) -> None:
        """Initialize the embedding generator.

        Args:
            config: Optional embedding configuration.
            model_loader: Optional custom model loader for testing or injection.

        Raises:
            EmbeddingError: If configuration is invalid.
        """
        try:
            self._config = resolve_embedding_config(config)
        except (TypeError, ValueError, AttributeError) as exc:
            raise EmbeddingError("Invalid embedding configuration.") from exc

        self._custom_model_loader = model_loader
        self._model: Any | None = None
        self._dimension: int | None = None
        self._statistics: EmbeddingStatistics | None = None
        self._logger = get_logger("embedding_generator")

    @property
    def config(self) -> EmbeddingConfig:
        """Return active configuration.

        Returns:
            Active `EmbeddingConfig`.
        """
        return self._config

    @property
    def model_name(self) -> str:
        """Return configured model name.

        Returns:
            Embedding model name.
        """
        return self._config.embedding_model

    @property
    def dimension(self) -> int | None:
        """Return known embedding dimension.

        Returns:
            Embedding dimension if known, otherwise None.
        """
        return self._dimension

    @property
    def statistics(self) -> EmbeddingStatistics | None:
        """Return latest encoding statistics.

        Returns:
            Latest statistics if available.
        """
        return self._statistics

    def load_model(self) -> Any:
        """Load the embedding model lazily and cache it.

        Returns:
            Loaded model instance.

        Raises:
            EmbeddingError: If model loading fails.
        """
        if self._model is not None:
            return self._model

        device = _resolve_device(self._config.device)

        if self._custom_model_loader is not None:
            self._logger.info("Loading injected embedding model.")
            model = self._custom_model_loader()
        else:
            cache_key = (
                self._config.embedding_model,
                device,
                (
                    str(self._config.cache_folder)
                    if self._config.cache_folder is not None
                    else None
                ),
            )

            cached_model = _MODEL_CACHE.get(cache_key)
            if cached_model is not None:
                self._logger.info(
                    "Reusing cached embedding model '%s' on device '%s'.",
                    self._config.embedding_model,
                    device,
                )
                model = cached_model
            else:
                self._logger.info(
                    "Loading embedding model '%s' on device '%s'.",
                    self._config.embedding_model,
                    device,
                )
                model = _default_model_loader(
                    model_name=self._config.embedding_model,
                    device=device,
                    cache_folder=self._config.cache_folder,
                )
                _MODEL_CACHE[cache_key] = model

        self._model = model

        dimension_getter = getattr(model, "get_embedding_dimension", None) or getattr(
            model, "get_sentence_embedding_dimension", None
        )
        if callable(dimension_getter):
            try:
                inferred_dimension = int(dimension_getter())
                if inferred_dimension > 0:
                    self._dimension = inferred_dimension
            except Exception:  # noqa: BLE001
                self._logger.debug("Could not infer embedding dimension from model.")

        return self._model

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a single text into an embedding vector.

        Args:
            text: Input text.

        Returns:
            Embedding vector.

        Raises:
            EmbeddingValidationError: If input text is invalid.
            EmbeddingError: If encoding fails.
        """
        if not isinstance(text, str):
            raise EmbeddingValidationError("Text must be a string.")

        model = self.load_model()
        vectors = self._encode_with_model(
            model=model,
            texts=[text],
            batch_size=1,
            show_progress=False,
        )

        vector = self._validate_and_prepare_vector(vectors[0])

        if self._dimension is None:
            self._dimension = vector.size

        return vector

    def encode_chunks(
        self,
        chunks: Iterable[Any],
        batch_size: int | None = None,
        show_progress: bool = True,
    ) -> list[EmbeddingRecord]:
        """Encode chunks into embedding records.

        Args:
            chunks: Iterable of chunk-like objects.
            batch_size: Optional batch size override.
            show_progress: Whether to show a tqdm progress bar.

        Returns:
            List of embedding records.

        Raises:
            EmbeddingError: If encoding fails or chunks are invalid.
            EmbeddingValidationError: If batch size or vectors are invalid.
        """
        chunks_list = list(chunks)

        if not chunks_list:
            self._logger.warning("No chunks provided for embedding generation.")
            self._statistics = EmbeddingStatistics(
                total_embeddings=0,
                embedding_dimension=self._dimension or 0,
                model_name=self._config.embedding_model,
                total_encoding_time_sec=0.0,
                average_encoding_time_sec=0.0,
                batch_count=0,
                memory_usage_bytes=0,
            )
            return []

        resolved_batch_size = int(batch_size or self._config.batch_size)
        if resolved_batch_size <= 0:
            raise EmbeddingValidationError("batch_size must be greater than zero.")

        chunk_parts: list[_ChunkParts] = []
        for index, chunk in enumerate(chunks_list):
            parts = _extract_chunk_parts(chunk, index)
            if not parts.text.strip():
                raise EmbeddingError(f"Chunk '{parts.chunk_id}' has empty text.")
            chunk_parts.append(parts)

        texts = [part.text for part in chunk_parts]

        model = self.load_model()

        self._logger.info(
            "Encoding %d chunk(s) with model '%s'.",
            len(texts),
            self._config.embedding_model,
        )

        start_time = time.perf_counter()
        vectors = self._encode_with_model(
            model=model,
            texts=texts,
            batch_size=resolved_batch_size,
            show_progress=show_progress,
        )
        total_time = time.perf_counter() - start_time

        if len(vectors) != len(chunk_parts):
            raise EmbeddingError(
                "Model returned an unexpected number of embedding vectors."
            )

        records: list[EmbeddingRecord] = []
        dimension = self._dimension

        for parts, vector in zip(chunk_parts, vectors, strict=True):
            validated_vector = self._validate_and_prepare_vector(
                vector,
                expected_dimension=dimension,
            )

            if dimension is None:
                dimension = validated_vector.size
                self._dimension = dimension

            records.append(
                EmbeddingRecord(
                    chunk_id=parts.chunk_id,
                    document_id=parts.document_id,
                    embedding=validated_vector,
                    dimension=validated_vector.size,
                    model_name=self._config.embedding_model,
                    created_at=_utc_timestamp(),
                    metadata=dict(parts.metadata),
                )
            )

        batch_count = math.ceil(len(texts) / resolved_batch_size)
        memory_usage_bytes = len(records) * (dimension or 0) * 4
        average_time = total_time / len(records) if records else 0.0

        self._statistics = EmbeddingStatistics(
            total_embeddings=len(records),
            embedding_dimension=dimension or 0,
            model_name=self._config.embedding_model,
            total_encoding_time_sec=round(total_time, 6),
            average_encoding_time_sec=round(average_time, 6),
            batch_count=batch_count,
            memory_usage_bytes=memory_usage_bytes,
        )

        self._logger.info(
            "Generated %d embedding(s) in %s second(s).",
            len(records),
            self._statistics.total_encoding_time_sec,
        )

        return records

    def validate_vector(
        self,
        vector: np.ndarray,
        expected_dimension: int | None = None,
    ) -> np.ndarray:
        """Validate an embedding vector.

        Args:
            vector: Candidate vector.
            expected_dimension: Optional expected dimension.

        Returns:
            Validated vector as float32 NumPy array.

        Raises:
            EmbeddingValidationError: If validation fails.
        """
        return self._validate_and_prepare_vector(
            vector=vector,
            expected_dimension=expected_dimension,
        )

    def _encode_with_model(
        self,
        model: Any,
        texts: list[str],
        batch_size: int,
        show_progress: bool,
    ) -> np.ndarray:
        """Encode texts using the loaded model.

        Args:
            model: Embedding model.
            texts: Input texts.
            batch_size: Batch size.
            show_progress: Whether to show progress bar.

        Returns:
            Two-dimensional float32 array of vectors.

        Raises:
            EmbeddingError: If encoding fails or output is invalid.
        """
        try:
            result = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=self._config.normalize_embeddings,
                convert_to_numpy=True,
            )
        except TypeError:
            try:
                result = model.encode(texts)
            except Exception as exc:  # noqa: BLE001
                raise EmbeddingError("Model encoding failed.") from exc
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError("Model encoding failed.") from exc

        if hasattr(result, "detach"):
            result = result.detach().cpu().numpy()

        try:
            array = np.asarray(result, dtype=np.float32)
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(
                "Model output could not be converted to a NumPy array."
            ) from exc

        if array.ndim == 1:
            if len(texts) == 1:
                array = array.reshape(1, -1)
            else:
                raise EmbeddingError(
                    "Model returned a one-dimensional array for batch encoding."
                )

        if array.ndim != 2:
            raise EmbeddingError("Model output must be a two-dimensional array.")

        if array.shape[0] != len(texts):
            raise EmbeddingError(
                "Model returned a number of vectors different from input texts."
            )

        return array

    def _validate_and_prepare_vector(
        self,
        vector: Any,
        expected_dimension: int | None = None,
    ) -> np.ndarray:
        """Validate and normalize a single vector.

        Args:
            vector: Candidate vector.
            expected_dimension: Optional expected dimension.

        Returns:
            Validated float32 vector.

        Raises:
            EmbeddingValidationError: If validation fails.
        """
        if not isinstance(vector, np.ndarray):
            try:
                vector = np.asarray(vector, dtype=np.float32)
            except Exception as exc:  # noqa: BLE001
                raise EmbeddingValidationError(
                    "Embedding must be convertible to numpy.ndarray."
                ) from exc
        else:
            vector = vector.astype(np.float32, copy=False)

        if vector.ndim != 1:
            if vector.size == 0:
                raise EmbeddingValidationError("Embedding must not be empty.")
            vector = vector.reshape(-1)

        if vector.size == 0:
            raise EmbeddingValidationError("Embedding must not be empty.")

        if not np.all(np.isfinite(vector)):
            raise EmbeddingValidationError("Embedding contains NaN or infinite values.")

        resolved_expected_dimension = expected_dimension or self._dimension

        if (
            resolved_expected_dimension is not None
            and vector.size != resolved_expected_dimension
        ):
            raise EmbeddingValidationError(
                "Embedding dimension mismatch. "
                f"Expected {resolved_expected_dimension}, got {vector.size}."
            )

        return vector


class EmbeddingExporter:
    """Exports embedding records to supported file formats.

    Attributes:
        config: Resolved embedding configuration.
        output_dir: Export destination directory.
    """

    def __init__(
        self,
        config: EmbeddingConfig | Any | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        """Initialize exporter.

        Args:
            config: Optional embedding configuration.
            output_dir: Optional output directory override.

        Raises:
            EmbeddingExportError: If configuration is invalid.
        """
        try:
            self._config = resolve_embedding_config(config)
        except (TypeError, ValueError, AttributeError) as exc:
            raise EmbeddingExportError(
                "Invalid embedding export configuration."
            ) from exc

        self._output_dir = (
            Path(output_dir)
            if output_dir is not None
            else Path(self._config.embeddings_dir)
        )
        self._logger = get_logger("embedding_exporter")

    @property
    def output_dir(self) -> Path:
        """Return output directory.

        Returns:
            Export directory path.
        """
        return self._output_dir

    def export(
        self,
        records: Sequence[EmbeddingRecord],
        file_name: str | None = None,
        export_format: str | None = None,
    ) -> Path:
        """Export embedding records to disk.

        Args:
            records: Embedding records to export.
            file_name: Optional file name.
            export_format: Optional export format override.

        Returns:
            Path to exported file.

        Raises:
            EmbeddingExportError: If export fails.
        """
        if not records:
            raise EmbeddingExportError("No embedding records to export.")

        resolved_format = (export_format or self._config.export_format).lower()

        allowed_formats = {"npy", "pickle", "parquet", "json"}
        if resolved_format not in allowed_formats:
            raise EmbeddingExportError(
                f"Unsupported embedding export format: {resolved_format}"
            )

        try:
            self._output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise EmbeddingExportError(
                "Could not create embedding export directory."
            ) from exc

        if file_name is None:
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            file_name = f"embeddings_{timestamp}.{resolved_format}"

        path = self._output_dir / file_name

        if path.suffix.lower() != f".{resolved_format}":
            path = path.with_suffix(f".{resolved_format}")

        self._logger.info(
            "Exporting %d embedding record(s) to '%s'.",
            len(records),
            path,
        )

        try:
            if resolved_format == "npy":
                exported_path = self._export_npy(records, path)
            elif resolved_format == "pickle":
                exported_path = self._export_pickle(records, path)
            elif resolved_format == "parquet":
                exported_path = self._export_parquet(records, path)
            else:
                exported_path = self._export_json(records, path)
        except EmbeddingExportError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingExportError("Embedding export failed.") from exc

        self._logger.info(
            "Embeddings exported successfully to '%s'.",
            exported_path,
        )

        return exported_path

    def _export_npy(
        self,
        records: Sequence[EmbeddingRecord],
        path: Path,
    ) -> Path:
        """Export embeddings as NumPy array plus metadata JSON.

        Args:
            records: Embedding records.
            path: Destination path for `.npy` file.

        Returns:
            Path to exported `.npy` file.
        """
        embeddings = np.vstack(
            [record.embedding.reshape(1, -1) for record in records]
        ).astype(np.float32)

        np.save(path, embeddings)

        metadata_path = path.with_suffix(".json")
        metadata_payload = [
            record.to_dict(include_embedding=False) for record in records
        ]

        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(
                metadata_payload,
                file,
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            )

        return path

    def _export_pickle(
        self,
        records: Sequence[EmbeddingRecord],
        path: Path,
    ) -> Path:
        """Export embeddings as pickle.

        Args:
            records: Embedding records.
            path: Destination path.

        Returns:
            Path to exported pickle file.
        """
        with path.open("wb") as file:
            pickle.dump(list(records), file)

        return path

    def _export_parquet(
        self,
        records: Sequence[EmbeddingRecord],
        path: Path,
    ) -> Path:
        """Export embeddings as Parquet.

        Args:
            records: Embedding records.
            path: Destination path.

        Returns:
            Path to exported Parquet file.

        Raises:
            EmbeddingExportError: If pandas or parquet engine is unavailable.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise EmbeddingExportError(
                "pandas is required for parquet export."
            ) from exc

        rows: list[dict[str, Any]] = []

        for record in records:
            row = record.to_dict(include_embedding=False)
            row["embedding"] = record.embedding.tolist()
            row["metadata"] = json.dumps(
                record.metadata,
                ensure_ascii=False,
                default=_json_default,
            )
            rows.append(row)

        dataframe = pd.DataFrame(rows)

        try:
            dataframe.to_parquet(path, index=False)
        except ImportError as exc:
            raise EmbeddingExportError(
                "A parquet engine such as pyarrow is required."
            ) from exc

        return path

    def _export_json(
        self,
        records: Sequence[EmbeddingRecord],
        path: Path,
    ) -> Path:
        """Export embeddings and metadata as JSON.

        Args:
            records: Embedding records.
            path: Destination path.

        Returns:
            Path to exported JSON file.
        """
        payload = {
            "model_name": records[0].model_name,
            "dimension": records[0].dimension,
            "exported_at": _utc_timestamp(),
            "total_embeddings": len(records),
            "records": [record.to_dict(include_embedding=True) for record in records],
        }

        with path.open("w", encoding="utf-8") as file:
            json.dump(
                payload,
                file,
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            )

        return path
