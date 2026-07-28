"""Chunking module for converting processed documents into chunks."""

from __future__ import annotations

import csv
import json
import re
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import ChunkingConfig, resolve_chunking_config
from src.text_processor import (
    extract_document_id,
    extract_document_metadata,
    extract_document_text,
)

try:
    from src.logger import get_logger
except ImportError:  # pragma: no cover
    from logging import getLogger as get_logger


class ChunkingError(Exception):
    """Base exception for chunking errors."""


class ChunkValidationError(ChunkingError):
    """Raised when chunk validation fails."""


class ChunkExportError(ChunkingError):
    """Raised when chunk export fails."""


_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

_REQUIRED_METADATA_KEYS = frozenset(
    {
        "dataset_split",
        "title",
        "question",
        "has_answer",
        "source_dataset",
        "chunk_size",
        "chunk_overlap",
        "processing_timestamp",
    }
)

_CHUNK_FIELDS = (
    "chunk_id",
    "document_id",
    "text",
    "metadata",
    "start_index",
    "end_index",
    "chunk_number",
    "token_estimate",
    "character_count",
)

_CSV_FIELDNAMES = (
    "chunk_id",
    "document_id",
    "chunk_number",
    "start_index",
    "end_index",
    "token_estimate",
    "character_count",
    "text",
    "metadata_json",
)


@dataclass(frozen=True)
class Chunk:
    """A single chunk produced from a processed document.

    Attributes:
        chunk_id: Unique chunk identifier.
        document_id: Parent document identifier.
        text: Chunk text.
        metadata: Chunk metadata.
        start_index: Start character index in the processed document.
        end_index: End character index in the processed document.
        chunk_number: Sequential chunk number within the document.
        token_estimate: Lightweight whitespace-based token estimate.
        character_count: Number of characters in the chunk.
    """

    chunk_id: str
    document_id: str
    text: str
    metadata: dict[str, Any]
    start_index: int
    end_index: int
    chunk_number: int
    token_estimate: int
    character_count: int

    def to_dict(self) -> dict[str, Any]:
        """Convert chunk to dictionary.

        Returns:
            Dictionary representation of the chunk.
        """
        return asdict(self)


@dataclass(frozen=True)
class ChunkStatistics:
    """Statistics for a collection of chunks.

    Attributes:
        number_of_chunks: Total chunk count.
        average_chunk_length: Average character count.
        maximum_chunk_length: Maximum character count.
        minimum_chunk_length: Minimum character count.
        average_chunks_per_document: Average chunks per document.
        estimated_tokens: Total estimated tokens.
    """

    number_of_chunks: int
    average_chunk_length: float
    maximum_chunk_length: int
    minimum_chunk_length: int
    average_chunks_per_document: float
    estimated_tokens: int

    def to_dict(self) -> dict[str, Any]:
        """Convert statistics to dictionary.

        Returns:
            Dictionary representation of statistics.
        """
        return asdict(self)


def _utc_timestamp() -> str:
    """Return current UTC timestamp in ISO format.

    Returns:
        ISO timestamp string.
    """
    return datetime.now(UTC).isoformat()


def _estimate_tokens(text: str) -> int:
    """Estimate token count using whitespace splitting.

    TODO:
        Replace with a real tokenizer in a future phase.

    Args:
        text: Input text.

    Returns:
        Estimated token count.
    """
    if not text:
        return 0
    return len(text.split())


def _find_separator_boundary(
    text: str,
    start: int,
    desired_end: int,
    separators: Sequence[str],
) -> int:
    """Find the best separator-based boundary before `desired_end`.

    Args:
        text: Full source text.
        start: Window start index.
        desired_end: Desired window end index.
        separators: Candidate separators.

    Returns:
        Best boundary index.
    """
    best = start

    for separator in separators:
        if not separator:
            continue

        index = text.rfind(separator, start, desired_end)
        if index == -1:
            continue

        boundary = index + len(separator)
        if start < boundary <= desired_end and boundary > best:
            best = boundary

    return best


def _character_window_spans(
    text: str,
    config: ChunkingConfig,
) -> list[tuple[int, int]]:
    """Generate fixed-size character spans with overlap.

    Args:
        text: Input text.
        config: Chunking configuration.

    Returns:
        List of start/end spans.
    """
    step = config.chunk_size - config.chunk_overlap
    text_length = len(text)
    start = 0
    spans: list[tuple[int, int]] = []

    while start < text_length:
        end = min(start + config.chunk_size, text_length)
        spans.append((start, end))

        if end >= text_length:
            break

        start += step

    return spans


def _recursive_window_spans(
    text: str,
    config: ChunkingConfig,
) -> list[tuple[int, int]]:
    """Generate recursive character-style spans with overlap.

    Args:
        text: Input text.
        config: Chunking configuration.

    Returns:
        List of start/end spans.
    """
    step = config.chunk_size - config.chunk_overlap
    separators = [separator for separator in config.recursive_separators if separator]
    text_length = len(text)
    start = 0
    spans: list[tuple[int, int]] = []

    while start < text_length:
        desired_end = min(start + config.chunk_size, text_length)

        if desired_end >= text_length:
            end = text_length
        else:
            end = _find_separator_boundary(
                text=text,
                start=start,
                desired_end=desired_end,
                separators=separators,
            )
            if end <= start:
                end = desired_end

        if end - start < config.min_chunk_length and end < text_length:
            end = desired_end

        spans.append((start, end))

        if end >= text_length:
            break

        next_start = end - config.chunk_overlap
        if next_start <= start:
            next_start = start + step

        start = min(next_start, text_length)

    return spans


def _sentence_window_spans(
    text: str,
    config: ChunkingConfig,
) -> list[tuple[int, int]]:
    """Generate sentence-aware spans with overlap.

    Args:
        text: Input text.
        config: Chunking configuration.

    Returns:
        List of start/end spans.
    """
    step = config.chunk_size - config.chunk_overlap
    escaped_markers = "".join(re.escape(marker) for marker in config.sentence_endings)
    pattern = re.compile(rf"(?<=[{escaped_markers}])\s+")

    text_length = len(text)
    start = 0
    spans: list[tuple[int, int]] = []

    while start < text_length:
        desired_end = min(start + config.chunk_size, text_length)

        if desired_end >= text_length:
            end = text_length
        else:
            end = start
            for match in pattern.finditer(text, start, desired_end):
                if match.end() <= desired_end and match.end() > end:
                    end = match.end()

            if end <= start:
                end = _find_separator_boundary(
                    text=text,
                    start=start,
                    desired_end=desired_end,
                    separators=["\n\n", "\n", ". ", " "],
                )
                if end <= start:
                    end = desired_end

        if end - start < config.min_chunk_length and end < text_length:
            end = desired_end

        spans.append((start, end))

        if end >= text_length:
            break

        next_start = end - config.chunk_overlap
        if next_start <= start:
            next_start = start + step

        start = min(next_start, text_length)

    return spans


def compute_statistics(chunks: Sequence[Chunk]) -> ChunkStatistics:
    """Compute statistics for chunks.

    Args:
        chunks: Sequence of chunks.

    Returns:
        Chunk statistics.
    """
    if not chunks:
        return ChunkStatistics(
            number_of_chunks=0,
            average_chunk_length=0.0,
            maximum_chunk_length=0,
            minimum_chunk_length=0,
            average_chunks_per_document=0.0,
            estimated_tokens=0,
        )

    lengths = [chunk.character_count for chunk in chunks]
    document_ids = {chunk.document_id for chunk in chunks}
    document_count = len(document_ids)

    return ChunkStatistics(
        number_of_chunks=len(chunks),
        average_chunk_length=round(statistics.mean(lengths), 2),
        maximum_chunk_length=max(lengths),
        minimum_chunk_length=min(lengths),
        average_chunks_per_document=(
            round(len(chunks) / document_count, 2) if document_count else 0.0
        ),
        estimated_tokens=sum(chunk.token_estimate for chunk in chunks),
    )


class Chunker:
    """Chunks processed documents into validated `Chunk` objects.

    Attributes:
        config: Resolved chunking configuration.
    """

    def __init__(
        self,
        config: ChunkingConfig | Any | None = None,
    ) -> None:
        """Initialize the chunker.

        Args:
            config: Optional chunking configuration.

        Raises:
            ChunkValidationError: If configuration is invalid.
        """
        try:
            self._config = resolve_chunking_config(config)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ChunkValidationError("Invalid chunking configuration.") from exc

        self._logger = get_logger("chunker")
        self._logger.debug(
            "Chunker initialized with strategy '%s'.",
            self._config.chunk_strategy,
        )

    @property
    def config(self) -> ChunkingConfig:
        """Return active configuration.

        Returns:
            Active `ChunkingConfig`.
        """
        return self._config

    def chunk_documents(
        self,
        documents: Iterable[Any],
    ) -> list[Chunk]:
        """Chunk a collection of documents.

        Args:
            documents: Iterable of processed or raw document-like objects.

        Returns:
            List of chunks.

        Raises:
            ChunkingError: If chunking fails.
        """
        documents_list = list(documents)

        if not documents_list:
            self._logger.warning("No documents provided for chunking.")
            return []

        self._logger.info(
            "Starting chunking for %d document(s).",
            len(documents_list),
        )

        chunks: list[Chunk] = []

        for index, document in enumerate(documents_list):
            document_id = extract_document_id(document, index)
            text = extract_document_text(document)
            metadata = extract_document_metadata(document)

            if not text.strip():
                self._logger.warning(
                    "Document '%s' has empty text and was skipped.",
                    document_id,
                )
                continue

            document_chunks = self._chunk_single_document(
                document_id=document_id,
                text=text,
                metadata=metadata,
            )
            chunks.extend(document_chunks)

        stats = compute_statistics(chunks)

        self._logger.info(
            "Generated %d chunk(s) from %d document(s).",
            stats.number_of_chunks,
            len(documents_list),
        )
        self._logger.debug(
            "Chunk statistics: %s",
            stats.to_dict(),
        )

        return chunks

    def _chunk_single_document(
        self,
        document_id: str,
        text: str,
        metadata: dict[str, Any],
    ) -> list[Chunk]:
        """Chunk one document.

        Args:
            document_id: Document identifier.
            text: Processed document text.
            metadata: Document metadata.

        Returns:
            List of chunks for the document.

        Raises:
            ChunkingError: If chunk generation fails.
        """
        base_metadata = self._build_chunk_metadata(metadata)
        spans = self._generate_spans(text)
        document_chunks: list[Chunk] = []

        for start, end in spans:
            raw_chunk_text = text[start:end]
            stripped_text = raw_chunk_text.strip()

            if not stripped_text:
                self._logger.debug(
                    "Skipping empty chunk for document '%s'.",
                    document_id,
                )
                continue

            leading_offset = len(raw_chunk_text) - len(raw_chunk_text.lstrip())
            trailing_offset = len(raw_chunk_text) - len(raw_chunk_text.rstrip())

            adjusted_start = start + leading_offset
            adjusted_end = end - trailing_offset

            if adjusted_end <= adjusted_start:
                continue

            chunk_text = text[adjusted_start:adjusted_end]
            if not chunk_text:
                continue

            chunk_number = len(document_chunks) + 1

            chunk = Chunk(
                chunk_id=f"{document_id}_chunk_{chunk_number:04d}",
                document_id=document_id,
                text=chunk_text,
                metadata=dict(base_metadata),
                start_index=adjusted_start,
                end_index=adjusted_end,
                chunk_number=chunk_number,
                token_estimate=_estimate_tokens(chunk_text),
                character_count=len(chunk_text),
            )

            self._validate_chunk(chunk, len(text))
            document_chunks.append(chunk)

        self._logger.debug(
            "Document '%s' produced %d chunk(s).",
            document_id,
            len(document_chunks),
        )

        return document_chunks

    def _build_chunk_metadata(
        self,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Build required chunk metadata while preserving future fields.

        Args:
            metadata: Source document metadata.

        Returns:
            Enriched metadata dictionary.
        """
        meta = dict(metadata or {})

        meta.setdefault(
            "dataset_split",
            meta.get("split", "unknown"),
        )
        meta.setdefault("title", meta.get("title", ""))
        meta.setdefault("question", meta.get("question", ""))

        if "has_answer" not in meta:
            answer = meta.get("answer", meta.get("answers"))
            if isinstance(answer, (list, tuple, set)):
                meta["has_answer"] = len(answer) > 0
            else:
                meta["has_answer"] = bool(answer)

        meta.setdefault(
            "source_dataset",
            meta.get("dataset", meta.get("source", "squad_v2")),
        )

        meta["chunk_size"] = self._config.chunk_size
        meta["chunk_overlap"] = self._config.chunk_overlap
        meta["processing_timestamp"] = _utc_timestamp()

        return meta

    def _generate_spans(self, text: str) -> list[tuple[int, int]]:
        """Generate text spans according to configured strategy.

        Args:
            text: Input text.

        Returns:
            List of start/end spans.

        Raises:
            ChunkingError: If semantic chunking is disabled.
            ChunkValidationError: If strategy is unsupported.
        """
        strategy = self._config.chunk_strategy

        if strategy == "semantic":
            if not self._config.future_semantic_chunking:
                raise ChunkingError(
                    "Semantic chunking is disabled and reserved for future use."
                )

            self._logger.warning(
                "Semantic chunking is not implemented yet; "
                "falling back to recursive_character."
            )
            strategy = "recursive_character"

        if strategy == "recursive_character":
            return _recursive_window_spans(text, self._config)

        if strategy == "character":
            return _character_window_spans(text, self._config)

        if strategy == "sentence":
            return _sentence_window_spans(text, self._config)

        raise ChunkValidationError(f"Unsupported chunk strategy: {strategy}")

    def _validate_chunk(self, chunk: Chunk, text_length: int) -> None:
        """Validate a generated chunk.

        Args:
            chunk: Generated chunk.
            text_length: Length of source text.

        Raises:
            ChunkValidationError: If validation fails.
        """
        if not chunk.text.strip():
            raise ChunkValidationError("Chunk text must not be empty.")

        if chunk.character_count != len(chunk.text):
            raise ChunkValidationError(
                "Chunk character_count does not match chunk text length."
            )

        if chunk.start_index < 0:
            raise ChunkValidationError("Chunk start_index must be >= 0.")

        if chunk.end_index > text_length:
            raise ChunkValidationError("Chunk end_index exceeds source text length.")

        if chunk.start_index >= chunk.end_index:
            raise ChunkValidationError(
                "Chunk start_index must be smaller than end_index."
            )

        if chunk.token_estimate < 0:
            raise ChunkValidationError("Chunk token_estimate must be non-negative.")

        if not isinstance(chunk.metadata, Mapping):
            raise ChunkValidationError("Chunk metadata must be a mapping.")

        missing_keys = _REQUIRED_METADATA_KEYS.difference(chunk.metadata)
        if missing_keys:
            raise ChunkValidationError(
                f"Chunk metadata is missing required keys: {sorted(missing_keys)}"
            )


class ChunkExporter:
    """Exports chunks to supported file formats.

    Attributes:
        config: Resolved chunking configuration.
        output_dir: Export destination directory.
    """

    def __init__(
        self,
        config: ChunkingConfig | Any | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        """Initialize exporter.

        Args:
            config: Optional configuration object.
            output_dir: Optional output directory override.

        Raises:
            ChunkExportError: If configuration is invalid.
        """
        try:
            self._config = resolve_chunking_config(config)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ChunkExportError("Invalid export configuration.") from exc

        self._output_dir = (
            Path(output_dir)
            if output_dir is not None
            else Path(self._config.processed_chunks_dir)
        )
        self._logger = get_logger("chunk_exporter")

    @property
    def output_dir(self) -> Path:
        """Return output directory.

        Returns:
            Export directory path.
        """
        return self._output_dir

    def export(
        self,
        chunks: Sequence[Chunk],
        file_name: str | None = None,
        export_format: str | None = None,
    ) -> Path:
        """Export chunks to disk.

        Args:
            chunks: Chunks to export.
            file_name: Optional file name.
            export_format: Optional export format override.

        Returns:
            Path to exported file.

        Raises:
            ChunkExportError: If export fails.
        """
        resolved_format = (export_format or self._config.export_format).lower()

        allowed_formats = {"json", "jsonl", "csv", "parquet"}
        if resolved_format not in allowed_formats:
            raise ChunkExportError(f"Unsupported export format: {resolved_format}")

        try:
            self._output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ChunkExportError("Could not create export directory.") from exc

        if file_name is None:
            timestamp = datetime.now(UTC).strftime(_TIMESTAMP_FORMAT)
            file_name = f"chunks_{timestamp}.{resolved_format}"

        path = self._output_dir / file_name

        if path.suffix.lower() != f".{resolved_format}":
            path = path.with_suffix(f".{resolved_format}")

        records = [chunk.to_dict() for chunk in chunks]

        self._logger.info(
            "Exporting %d chunk(s) to '%s'.",
            len(records),
            path,
        )

        try:
            if resolved_format == "json":
                self._export_json(records, path)
            elif resolved_format == "jsonl":
                self._export_jsonl(records, path)
            elif resolved_format == "csv":
                self._export_csv(records, path)
            else:
                self._export_parquet(records, path)
        except ChunkExportError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ChunkExportError("Chunk export failed.") from exc

        self._logger.info(
            "Chunks exported successfully to '%s'.",
            path,
        )

        return path

    def _export_json(
        self,
        records: list[dict[str, Any]],
        path: Path,
    ) -> None:
        """Export records as JSON.

        Args:
            records: Chunk records.
            path: Destination path.
        """
        with path.open("w", encoding="utf-8") as file:
            json.dump(
                records,
                file,
                ensure_ascii=False,
                indent=2,
                default=str,
            )

    def _export_jsonl(
        self,
        records: list[dict[str, Any]],
        path: Path,
    ) -> None:
        """Export records as JSON Lines.

        Args:
            records: Chunk records.
            path: Destination path.
        """
        with path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        default=str,
                    )
                )
                file.write("\n")

    def _export_csv(
        self,
        records: list[dict[str, Any]],
        path: Path,
    ) -> None:
        """Export records as CSV.

        Args:
            records: Chunk records.
            path: Destination path.
        """
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=_CSV_FIELDNAMES,
            )
            writer.writeheader()

            for record in records:
                row = {
                    key: record.get(key, "")
                    for key in _CSV_FIELDNAMES
                    if key != "metadata_json"
                }
                row["metadata_json"] = json.dumps(
                    record.get("metadata", {}),
                    ensure_ascii=False,
                    default=str,
                )
                writer.writerow(row)

    def _export_parquet(
        self,
        records: list[dict[str, Any]],
        path: Path,
    ) -> None:
        """Export records as Parquet.

        Args:
            records: Chunk records.
            path: Destination path.

        Raises:
            ChunkExportError: If pandas or parquet engine is unavailable.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ChunkExportError("pandas is required for parquet export.") from exc

        rows: list[dict[str, Any]] = []

        for record in records:
            row = dict(record)
            row["metadata"] = json.dumps(
                row.get("metadata", {}),
                ensure_ascii=False,
                default=str,
            )
            rows.append(row)

        dataframe = (
            pd.DataFrame(rows, columns=list(_CHUNK_FIELDS))
            if rows
            else pd.DataFrame(columns=list(_CHUNK_FIELDS))
        )

        try:
            dataframe.to_parquet(path, index=False)
        except ImportError as exc:
            raise ChunkExportError(
                "A parquet engine such as pyarrow is required."
            ) from exc
