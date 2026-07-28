"""Text processing module for cleaning and normalization."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from src.config import ChunkingConfig, resolve_chunking_config

try:
    from src.logger import get_logger
except ImportError:  # pragma: no cover
    from logging import getLogger as get_logger


class TextProcessingError(Exception):
    """Raised when text processing fails."""


@dataclass(frozen=True)
class ProcessedDocument:
    """A cleaned and normalized document ready for chunking.

    Attributes:
        document_id: Unique document identifier.
        text: Processed text content.
        metadata: Document metadata.
    """

    document_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


_TEXT_KEYS = (
    "text",
    "context",
    "page_content",
    "content",
    "passage",
)
_ID_KEYS = (
    "document_id",
    "id",
    "doc_id",
    "question_id",
)
_METADATA_KEYS = (
    "metadata",
    "meta",
)

_INVISIBLE_RE = re.compile(r"[\u00ad\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
_WHITESPACE_RE = re.compile(r"\s+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")

_QUOTE_TRANSLATION = str.maketrans(
    {
        0x2018: "'",
        0x2019: "'",
        0x201A: "'",
        0x201B: "'",
        0x2039: "'",
        0x203A: "'",
        0x201C: '"',
        0x201D: '"',
        0x201E: '"',
        0x201F: '"',
        0x00AB: '"',
        0x00BB: '"',
    }
)

_DASH_TRANSLATION = str.maketrans(
    {
        0x2010: "-",
        0x2011: "-",
        0x2012: "-",
        0x2013: "-",
        0x2014: "-",
        0x2015: "-",
        0x2212: "-",
    }
)


def _normalize_line_whitespace(text: str) -> str:
    """Normalize whitespace while preserving line breaks.

    Args:
        text: Raw input text.

    Returns:
        Text with normalized spaces and stripped lines.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    normalized_lines = [_WHITESPACE_RE.sub(" ", line).strip() for line in lines]
    return "\n".join(normalized_lines)


def _extract_value(document: Any, keys: tuple[str, ...]) -> Any | None:
    """Extract the first available value from a document.

    Args:
        document: Document-like object or mapping.
        keys: Candidate keys or attribute names.

    Returns:
        The first found value, or None.
    """
    if isinstance(document, ProcessedDocument):
        for key in keys:
            if hasattr(document, key):
                return getattr(document, key)

    if isinstance(document, Mapping):
        for key in keys:
            if key in document:
                return document[key]
        return None

    for key in keys:
        if hasattr(document, key):
            return getattr(document, key)

    return None


def extract_document_text(document: Any) -> str:
    """Extract text content from a document-like object.

    Args:
        document: Document-like object or mapping.

    Returns:
        Extracted text as string.

    Raises:
        TextProcessingError: If no text field can be found.
    """
    value = _extract_value(document, _TEXT_KEYS)
    if value is None:
        raise TextProcessingError("Document text could not be extracted.")
    return str(value)


def extract_document_id(document: Any, index: int = 0) -> str:
    """Extract a document identifier or generate a fallback.

    Args:
        document: Document-like object or mapping.
        index: Document position used for fallback ID.

    Returns:
        Document identifier.
    """
    value = _extract_value(document, _ID_KEYS)
    if value is None or str(value).strip() == "":
        return f"doc_{index + 1}"
    return str(value)


def extract_document_metadata(document: Any) -> dict[str, Any]:
    """Extract metadata from a document-like object.

    Args:
        document: Document-like object or mapping.

    Returns:
        Metadata dictionary.
    """
    if isinstance(document, ProcessedDocument):
        return dict(document.metadata)

    if isinstance(document, Mapping):
        metadata = document.get("metadata") or document.get("meta")
        if isinstance(metadata, Mapping):
            return dict(metadata)

        return {
            key: value
            for key, value in document.items()
            if key not in _TEXT_KEYS
            and key not in _ID_KEYS
            and key not in _METADATA_KEYS
        }

    metadata = getattr(document, "metadata", None) or getattr(
        document,
        "meta",
        None,
    )
    if isinstance(metadata, Mapping):
        return dict(metadata)

    return {}


class TextProcessor:
    """Cleans and normalizes document text.

    Attributes:
        config: Resolved chunking/text-processing configuration.
    """

    def __init__(
        self,
        config: ChunkingConfig | Any | None = None,
    ) -> None:
        """Initialize the text processor.

        Args:
            config: Optional configuration object.

        Raises:
            TextProcessingError: If configuration is invalid.
        """
        try:
            self._config = resolve_chunking_config(config)
        except (TypeError, ValueError, AttributeError) as exc:
            raise TextProcessingError("Invalid text-processing configuration.") from exc

        self._logger = get_logger("text_processor")

    @property
    def config(self) -> ChunkingConfig:
        """Return the resolved configuration.

        Returns:
            Active `ChunkingConfig`.
        """
        return self._config

    def clean_text(self, text: str) -> str:
        """Clean text without changing semantic meaning.

        Args:
            text: Raw text.

        Returns:
            Cleaned text.

        Raises:
            TextProcessingError: If input is not a string.
        """
        if not isinstance(text, str):
            raise TextProcessingError("Text must be a string.")

        if not text:
            return ""

        cleaned = _INVISIBLE_RE.sub("", text)
        cleaned = _normalize_line_whitespace(cleaned)
        cleaned = _BLANK_LINES_RE.sub("\n\n", cleaned)
        cleaned = cleaned.strip()

        self._logger.debug("Text cleaned successfully.")
        return cleaned

    def normalize_text(self, text: str) -> str:
        """Normalize unicode, quotes, dashes, and whitespace.

        Args:
            text: Raw text.

        Returns:
            Normalized text.

        Raises:
            TextProcessingError: If input is not a string.
        """
        if not isinstance(text, str):
            raise TextProcessingError("Text must be a string.")

        if not text:
            return ""

        normalized = unicodedata.normalize("NFC", text)
        normalized = normalized.translate(_QUOTE_TRANSLATION)
        normalized = normalized.translate(_DASH_TRANSLATION)
        normalized = _normalize_line_whitespace(normalized)
        normalized = _BLANK_LINES_RE.sub("\n\n", normalized)
        normalized = normalized.strip()

        self._logger.debug("Text normalized successfully.")
        return normalized

    def process_text(self, text: str) -> str:
        """Apply normalization and cleaning according to config.

        Args:
            text: Raw text.

        Returns:
            Processed text.

        Raises:
            TextProcessingError: If input is not a string.
        """
        if not isinstance(text, str):
            raise TextProcessingError("Text must be a string.")

        result = text

        if self._config.enable_normalization:
            result = self.normalize_text(result)

        if self._config.enable_cleaning:
            result = self.clean_text(result)

        return result

    def process_documents(
        self,
        documents: Iterable[Any],
    ) -> list[ProcessedDocument]:
        """Process a collection of documents.

        Args:
            documents: Iterable of document-like objects.

        Returns:
            List of processed documents.

        Raises:
            TextProcessingError: If a document cannot be processed.
        """
        documents_list = list(documents)
        self._logger.info(
            "Starting text processing for %d document(s).",
            len(documents_list),
        )

        processed_documents: list[ProcessedDocument] = []

        for index, document in enumerate(documents_list):
            document_id = extract_document_id(document, index)
            raw_text = extract_document_text(document)
            metadata = extract_document_metadata(document)

            processed_text = self.process_text(raw_text)

            processed_documents.append(
                ProcessedDocument(
                    document_id=document_id,
                    text=processed_text,
                    metadata=metadata,
                )
            )

            self._logger.debug(
                "Processed document '%s'.",
                document_id,
            )

        non_empty_count = sum(1 for doc in processed_documents if doc.text.strip())

        self._logger.info(
            "Finished text processing: %d document(s), %d non-empty.",
            len(processed_documents),
            non_empty_count,
        )

        return processed_documents
