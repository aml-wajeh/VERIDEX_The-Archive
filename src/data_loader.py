"""Dataset loading and validation layer.

Title:
    Dataset Loading & Validation Layer

Description:
    This module is the single source of data for every later RAG phase. It
    downloads SQuAD v2 from Hugging Face ``datasets`` (with automatic caching),
    validates schema and content, converts every example into a structured
    :class:`Document`, exposes aggregated :class:`DatasetStatistics`, and
    exports the cleaned data to ``data/processed/`` in JSON / JSONL / CSV /
    Parquet with progress bars.

    It intentionally performs **only** basic whitespace cleaning. Chunking,
    embedding, retrieval, prompting and generation all belong to later modules.

Responsibilities:
    - Load SQuAD v2 ``train`` and ``validation`` splits from Hugging Face.
    - Validate schema (required columns) and content (no empty text).
    - Convert examples into immutable :class:`Document` objects with rich,
      extensible metadata.
    - Compute aggregated :class:`DatasetStatistics`.
    - Export processed splits to disk in configurable formats with ``tqdm``.

Author:
    Author Placeholder
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict
from tqdm.auto import tqdm

from .config import DatasetConfig, PathConfig
from .logger import get_logger
from .utils import ensure_directory, normalize_whitespace, safe_filename

REQUIRED_COLUMNS: tuple[str, ...] = (
    "id",
    "title",
    "context",
    "question",
    "answers",
)
SUPPORTED_FORMATS: tuple[str, ...] = ("json", "jsonl", "csv", "parquet")
REQUIRED_METADATA_KEYS: tuple[str, ...] = (
    "dataset_split",
    "question_id",
    "has_answer",
    "context_length",
    "question_length",
    "answer_count",
    "source_dataset",
    "created_at",
)
_MAX_REPORTED_ERRORS: int = 50
_DEFAULT_VERSION: str = "v2"


class DatasetLoadingError(Exception):
    """Raised when the dataset cannot be loaded or accessed."""


class DatasetValidationError(Exception):
    """Raised when the dataset fails structural or content validation."""


class DatasetExportError(Exception):
    """Raised when an export operation cannot be completed."""


@dataclass(frozen=True)
class Document:
    """A single, cleaned question/context example.

    The ``metadata`` mapping is intentionally open: the eight required keys are
    always present, and extra keys (for example ``answer_starts``) are kept so
    that future phases can add information without changing this class.

    Attributes:
        id: Stable identifier of the example.
        title: Article / passage title (may be empty).
        context: Cleaned passage text the question refers to.
        question: Cleaned question text.
        answers: Cleaned answer texts (empty for unanswerable questions).
        metadata: Extensible bag of derived facts.

    Example:
        >>> doc = Document(id="1", title="T", context="c", question="q",
        ...                answers=["a"])
        >>> doc.to_dict()["id"]
        '1'
    """

    id: str
    title: str
    context: str
    question: str
    answers: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the document into a plain dictionary.

        Returns:
            A deep-ish dict copy (nested ``metadata`` is shallow-copied).
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Document:
        """Reconstruct a document from a dictionary (inverse of :meth:`to_dict`).

        Args:
            data: Mapping produced by :meth:`to_dict` or compatible schema.

        Returns:
            A new :class:`Document` instance.
        """
        return cls(
            id=str(data["id"]),
            title=str(data.get("title", "")),
            context=str(data.get("context", "")),
            question=str(data.get("question", "")),
            answers=list(data.get("answers", [])),
            metadata=dict(data.get("metadata", {})),
        )

    def __repr__(self) -> str:
        """Return a compact, readable representation (context is truncated)."""
        question = self.question
        if len(question) > 40:
            question = question[:37] + "..."
        return (
            f"Document(id={self.id!r}, question={question!r}, "
            f"context_len={len(self.context)}, answers={len(self.answers)})"
        )


@dataclass(frozen=True)
class DatasetStatistics:
    """Aggregated statistics across every loaded split.

    Attributes:
        total_samples: Total number of documents across all splits.
        train_samples: Number of documents in the ``train`` split (0 if absent).
        validation_samples: Number of documents in ``validation`` (0 if absent).
        average_context_length: Mean context length over all documents.
        average_question_length: Mean question length over all documents.
        average_answer_count: Mean answer count over all documents.
        maximum_context_length: Longest context across all documents.
        minimum_context_length: Shortest context across all documents.
        dataset_name: Hugging Face dataset identifier.
        dataset_version: Dataset version / revision (defaults to ``v2``).

    Example:
        >>> DatasetStatistics(0, 0, 0, 0.0, 0.0, 0.0, 0, 0, "n", "v2").total_samples
        0
    """

    total_samples: int
    train_samples: int
    validation_samples: int
    average_context_length: float
    average_question_length: float
    average_answer_count: float
    maximum_context_length: int
    minimum_context_length: int
    dataset_name: str
    dataset_version: str


class DataLoader:
    """Loads, validates, analyses and exports the SQuAD v2 dataset.

    The constructor performs **no I/O**; data is materialised by :meth:`load`
    (Hugging Face) or :meth:`load_from_records` (in-memory, used by tests and
    local overrides). This makes the class fully unit-testable without network
    access via dependency injection.

    Attributes:
        config: Dataset-related configuration.
        paths: Filesystem layout (used for the export directory and cache).

    Example:
        >>> from src.config import get_settings  # doctest: +SKIP
        >>> loader = DataLoader(get_settings().dataset, get_settings().paths)
    """

    def __init__(self, config: DatasetConfig, paths: PathConfig) -> None:
        self.config = config
        self.paths = paths
        self._logger = get_logger(__name__)
        self._dataset: DatasetDict | None = None
        self._documents: dict[str, list[Document]] = {}

    # ------------------------------------------------------------------ load

    def load(self) -> DataLoader:
        """Load the dataset from Hugging Face and validate it (fail-fast).

        Returns:
            ``self``, for fluent chaining.

        Raises:
            DatasetLoadingError: If the ``datasets`` library is missing or the
                remote load fails for any reason.
            DatasetValidationError: If the loaded data fails validation.
        """
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise DatasetLoadingError(
                "The 'datasets' library is required to load from Hugging Face. "
                "Install the project requirements first."
            ) from exc

        cache_dir = self._resolve_cache_dir()
        self._logger.info(
            "Loading dataset '%s' (cache=%s) ...",
            self.config.dataset_name,
            cache_dir,
        )

        kwargs: dict[str, Any] = {
            "path": self.config.dataset_name,
            "cache_dir": str(cache_dir),
        }
        if self.config.dataset_revision:
            kwargs["revision"] = self.config.dataset_revision

        try:
            dataset = load_dataset(**kwargs)
        except Exception as exc:  # network / hub errors are not our domain
            raise DatasetLoadingError(
                f"Failed to load dataset '{self.config.dataset_name}': {exc}"
            ) from exc

        if isinstance(dataset, Dataset):
            dataset = DatasetDict({"train": dataset})

        self._dataset = dataset
        self._validate_schema(dataset)
        self._build_all(validate=True)
        self.validate()
        self._logger.info("Loaded and validated splits: %s", self.splits())
        return self

    def load_from_records(
        self,
        records: Mapping[str, Iterable[Mapping[str, Any]]],
        *,
        validate: bool = True,
    ) -> DataLoader:
        """Build the loader from in-memory SQuAD-style records.

        This is the dependency-injection entry point used by tests and local
        overrides; it never touches the network.

        Args:
            records: Mapping of ``split_name`` to an iterable of row dicts that
                follow the SQuAD schema.
            validate: When ``True``, malformed rows raise immediately and a
                final content validation runs. When ``False``, malformed rows
                are skipped with a warning and content validation is deferred.

        Returns:
            ``self``, for fluent chaining.

        Raises:
            DatasetValidationError: If ``validate`` is ``True`` and any row is
                malformed or any content check fails.
        """
        self._dataset = None
        self._documents = {}
        for split, rows in tqdm(records.items(), desc="building splits"):
            self._documents[split] = self._build_split(rows, split, validate=validate)
        if validate:
            self.validate()
        self._logger.info(
            "Built %d split(s) from in-memory records: %s",
            len(self._documents),
            self.splits(),
        )
        return self

    # ------------------------------------------------------------ validation

    def validate(self) -> None:
        """Validate content of every loaded document.

        Empty contexts or questions are treated as corrupted data and raise.
        Missing answers are **not** an error (SQuAD v2 unanswerable questions).

        Returns:
            ``None``.

        Raises:
            DatasetLoadingError: If no data has been loaded yet.
            DatasetValidationError: If any content check fails.
        """
        if self._dataset is not None:
            self._validate_schema(self._dataset)
        if not self._documents:
            raise DatasetLoadingError(
                "No data loaded; call load() or load_from_records() first."
            )

        total = sum(len(docs) for docs in self._documents.values())
        errors: list[str] = []
        for split, docs in self._documents.items():
            for doc in tqdm(docs, desc=f"validate {split}", total=len(docs)):
                if not doc.context:
                    errors.append(f"split '{split}', id '{doc.id}': empty context")
                if not doc.question:
                    errors.append(f"split '{split}', id '{doc.id}': empty question")
        _ = total

        if errors:
            raise DatasetValidationError(
                "Validation failed with "
                f"{len(errors)} content error(s). First "
                f"{min(len(errors), _MAX_REPORTED_ERRORS)}:\n"
                + "\n".join(errors[:_MAX_REPORTED_ERRORS])
            )
        self._logger.info("Validation passed for splits: %s", self.splits())

    def _validate_schema(self, dataset: DatasetDict) -> None:
        """Verify every split exposes the required SQuAD columns.

        Args:
            dataset: The raw Hugging Face ``DatasetDict``.

        Raises:
            DatasetValidationError: If any split is missing required columns.
        """
        errors: list[str] = []
        for split, split_ds in dataset.items():
            missing = [c for c in REQUIRED_COLUMNS if c not in split_ds.column_names]
            if missing:
                errors.append(f"split '{split}' missing columns: {missing}")
        if errors:
            raise DatasetValidationError(
                "Schema validation failed:\n" + "\n".join(errors)
            )

    # --------------------------------------------------------------- access

    def documents(self, split: str) -> list[Document]:
        """Return the cleaned documents for a loaded split.

        Args:
            split: Split name (for example ``"train"``).

        Returns:
            The list of :class:`Document` objects for that split.

        Raises:
            DatasetLoadingError: If the split has not been loaded.
        """
        if split not in self._documents:
            raise DatasetLoadingError(
                f"Split '{split}' is not loaded. "
                f"Available splits: {self.splits()}"
            )
        return self._documents[split]

    def splits(self) -> list[str]:
        """Return the names of the loaded splits, in load order.

        Returns:
            A list of split names.
        """
        return list(self._documents.keys())

    # ------------------------------------------------------------ statistics

    def num_samples(self, split: str) -> int:
        """Return the number of documents in a split.

        Args:
            split: Split name.

        Returns:
            The document count.
        """
        return len(self.documents(split))

    def split_sizes(self) -> dict[str, int]:
        """Return the document count per loaded split.

        Returns:
            Mapping ``{split: count}``.
        """
        return {split: len(docs) for split, docs in self._documents.items()}

    def compute_statistics(self) -> DatasetStatistics:
        """Compute aggregated statistics across every loaded split.

        Returns:
            A :class:`DatasetStatistics` instance.

        Raises:
            DatasetLoadingError: If no data has been loaded yet.
        """
        if not self._documents:
            raise DatasetLoadingError(
                "No data loaded; call load() or load_from_records() first."
            )

        all_docs = [doc for docs in self._documents.values() for doc in docs]
        total = len(all_docs)
        train_samples = len(self._documents.get("train", []))
        validation_samples = len(self._documents.get("validation", []))

        if total == 0:
            return DatasetStatistics(
                total_samples=0,
                train_samples=train_samples,
                validation_samples=validation_samples,
                average_context_length=0.0,
                average_question_length=0.0,
                average_answer_count=0.0,
                maximum_context_length=0,
                minimum_context_length=0,
                dataset_name=self.config.dataset_name,
                dataset_version=self._version(),
            )

        context_lens = [len(d.context) for d in all_docs]
        question_lens = [len(d.question) for d in all_docs]
        answer_counts = [len(d.answers) for d in all_docs]

        return DatasetStatistics(
            total_samples=total,
            train_samples=train_samples,
            validation_samples=validation_samples,
            average_context_length=sum(context_lens) / total,
            average_question_length=sum(question_lens) / total,
            average_answer_count=sum(answer_counts) / total,
            maximum_context_length=max(context_lens),
            minimum_context_length=min(context_lens),
            dataset_name=self.config.dataset_name,
            dataset_version=self._version(),
        )

    def statistics(self, split: str | None = None) -> dict[str, Any]:
        """Return per-split statistics (one split or all splits).

        Args:
            split: A split name, or ``None`` for every loaded split.

        Returns:
            A statistics dict, or ``{split: stats}`` when ``split`` is ``None``.
        """
        if split is not None:
            return self._stats_for(split)
        return {s: self._stats_for(s) for s in self._documents}

    def answer_availability(self) -> dict[str, dict[str, Any]]:
        """Return answerable / unanswerable counts per split.

        Returns:
            Mapping ``{split: {"answerable", "unanswerable", "ratio"}}``.
        """
        result: dict[str, dict[str, Any]] = {}
        for split, docs in self._documents.items():
            answerable = sum(1 for d in docs if d.metadata.get("has_answer", False))
            total = len(docs)
            result[split] = {
                "answerable": answerable,
                "unanswerable": total - answerable,
                "ratio": (answerable / total) if total else 0.0,
            }
        return result

    def _stats_for(self, split: str) -> dict[str, Any]:
        """Compute the per-split statistics block.

        Args:
            split: Split name.

        Returns:
            A dict with sample counts and length statistics.
        """
        docs = self.documents(split)
        total = len(docs)
        if total == 0:
            return {
                "num_samples": 0,
                "avg_context_length": 0.0,
                "avg_question_length": 0.0,
                "max_context_length": 0,
                "min_context_length": 0,
            }

        context_lens = [len(d.context) for d in docs]
        question_lens = [len(d.question) for d in docs]
        answerable = sum(1 for d in docs if d.metadata.get("has_answer", False))
        return {
            "num_samples": total,
            "avg_context_length": sum(context_lens) / total,
            "avg_question_length": sum(question_lens) / total,
            "max_context_length": max(context_lens),
            "min_context_length": min(context_lens),
            "answerable": answerable,
            "unanswerable": total - answerable,
            "answerable_ratio": answerable / total,
        }

    def _version(self) -> str:
        """Return the dataset version string (revision or default).

        Returns:
            The configured revision, or ``_DEFAULT_VERSION`` when empty.
        """
        return self.config.dataset_revision or _DEFAULT_VERSION

    # ---------------------------------------------------------------- export

    def export(
        self,
        split: str,
        fmt: str = "jsonl",
        output_dir: Path | None = None,
    ) -> Path:
        """Export a cleaned split to disk.

        Args:
            split: Split name to export.
            fmt: One of ``"json"``, ``"jsonl"``, ``"csv"``, ``"parquet"``.
            output_dir: Destination directory; defaults to
                ``paths.processed_dir``.

        Returns:
            The ``Path`` of the written file.

        Raises:
            DatasetExportError: If the format is unsupported or a required
                optional dependency (Parquet) is missing.
        """
        if fmt not in SUPPORTED_FORMATS:
            raise DatasetExportError(
                f"Unsupported export format '{fmt}'. "
                f"Supported: {list(SUPPORTED_FORMATS)}"
            )

        docs = self.documents(split)
        directory = (
            Path(output_dir) if output_dir is not None else self.paths.processed_dir
        )
        ensure_directory(directory)

        prefix = safe_filename(self.config.dataset_name)
        path = directory / f"{prefix}_{split}.{fmt}"
        self._logger.info(
            "Exporting %d document(s) of split '%s' to %s (%s)",
            len(docs),
            split,
            path,
            fmt,
        )

        if fmt == "json":
            self._write_json(docs, path)
        elif fmt == "jsonl":
            self._write_jsonl(docs, path)
        elif fmt == "csv":
            self._write_csv(docs, path)
        else:
            self._write_parquet(docs, path)

        return path

    # --------------------------------------------------------- export writers

    def _write_json(self, docs: list[Document], path: Path) -> None:
        """Write documents as a single JSON array.

        Args:
            docs: Documents to write.
            path: Destination file path.
        """
        rows = [doc.to_dict() for doc in tqdm(docs, desc=f"serialise {path.name}")]
        with path.open("w", encoding="utf-8") as handle:
            json.dump(rows, handle, ensure_ascii=False, indent=2)

    def _write_jsonl(self, docs: list[Document], path: Path) -> None:
        """Write documents as one JSON object per line.

        Args:
            docs: Documents to write.
            path: Destination file path.
        """
        with path.open("w", encoding="utf-8") as handle:
            for doc in tqdm(docs, desc=f"write {path.name}"):
                handle.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")

    def _write_csv(self, docs: list[Document], path: Path) -> None:
        """Write documents as CSV with answers serialised as JSON.

        Args:
            docs: Documents to write.
            path: Destination file path.
        """
        fieldnames = [
            "id",
            "title",
            "context",
            "question",
            "answers",
            "has_answer",
            "dataset_split",
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for doc in tqdm(docs, desc=f"write {path.name}"):
                writer.writerow(
                    {
                        "id": doc.id,
                        "title": doc.title,
                        "context": doc.context,
                        "question": doc.question,
                        "answers": json.dumps(doc.answers, ensure_ascii=False),
                        "has_answer": doc.metadata.get("has_answer", False),
                        "dataset_split": doc.metadata.get("dataset_split", ""),
                    }
                )

    def _write_parquet(self, docs: list[Document], path: Path) -> None:
        """Write documents as Parquet (requires pandas + a parquet engine).

        Args:
            docs: Documents to write.
            path: Destination file path.

        Raises:
            DatasetExportError: If pandas or a parquet engine is unavailable.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise DatasetExportError(
                "Parquet export requires 'pandas'. Install it first."
            ) from exc

        rows = [doc.to_dict() for doc in tqdm(docs, desc=f"serialise {path.name}")]
        dataframe = pd.DataFrame(rows)
        try:
            dataframe.to_parquet(path, index=False)
        except (ImportError, ValueError) as exc:
            raise DatasetExportError(
                "Parquet export failed; ensure 'pyarrow' or 'fastparquet' "
                "is installed."
            ) from exc

    # --------------------------------------------------------- internal build

    def _build_all(self, *, validate: bool) -> None:
        """Build documents for every split in the raw dataset.

        Args:
            validate: Forwarded to :meth:`_build_split`.
        """
        assert self._dataset is not None  # guarded by load()
        items = list(self._dataset.items())
        for split, split_ds in tqdm(items, desc="loading splits"):
            self._documents[split] = self._build_split(
                split_ds, split, validate=validate
            )

    def _build_split(
        self,
        rows: Iterable[Mapping[str, Any]],
        split: str,
        *,
        validate: bool,
    ) -> list[Document]:
        """Convert raw rows into documents, handling malformed rows.

        Args:
            rows: Iterable of SQuAD-style row dicts.
            split: Split name (recorded in metadata).
            validate: When ``True``, any malformed row raises. When ``False``,
                malformed rows are skipped with a warning.

        Returns:
            The list of successfully built documents.

        Raises:
            DatasetValidationError: If ``validate`` is ``True`` and any row is
                malformed.
        """
        docs: list[Document] = []
        errors: list[str] = []
        total = len(rows) if hasattr(rows, "__len__") else None
        for row in tqdm(rows, total=total, desc=f"build {split}"):
            try:
                docs.append(self._row_to_document(row, split))
            except ValueError as exc:
                errors.append(str(exc))

        if errors:
            if validate:
                raise DatasetValidationError(
                    f"Failed to build split '{split}': {len(errors)} malformed "
                    f"row(s). First {min(len(errors), _MAX_REPORTED_ERRORS)}:\n"
                    + "\n".join(errors[:_MAX_REPORTED_ERRORS])
                )
            self._logger.warning(
                "Skipped %d malformed sample(s) in split '%s'.",
                len(errors),
                split,
            )
        return docs

    def _row_to_document(self, row: Mapping[str, Any], split: str) -> Document:
        """Convert one raw row into a cleaned :class:`Document`.

        Type / structural problems raise ``ValueError``; empty content does
        **not** raise here (it is enforced later by :meth:`validate`), and
        missing answers are recorded safely via the ``has_answer`` flag.

        Args:
            row: A SQuAD-style row dict.
            split: Split name stored in metadata.

        Returns:
            The cleaned document.

        Raises:
            ValueError: On missing id or wrong field types.
        """
        doc_id = row.get("id")
        if doc_id is None:
            raise ValueError("sample missing 'id'")

        context_raw = row.get("context")
        if not isinstance(context_raw, str):
            raise ValueError(f"sample {doc_id}: 'context' must be a string")

        question_raw = row.get("question")
        if not isinstance(question_raw, str):
            raise ValueError(f"sample {doc_id}: 'question' must be a string")

        answers_raw = row.get("answers")
        if not isinstance(answers_raw, Mapping):
            raise ValueError(f"sample {doc_id}: 'answers' must be a mapping")

        answer_texts = answers_raw.get("text", []) or []
        answer_starts = answers_raw.get("answer_start", []) or []
        if not isinstance(answer_texts, list):
            raise ValueError(f"sample {doc_id}: 'answers.text' must be a list")

        clean_context = normalize_whitespace(context_raw)
        clean_question = normalize_whitespace(question_raw)
        clean_answers = [
            normalize_whitespace(a) for a in answer_texts if isinstance(a, str)
        ]
        clean_starts = [int(s) for s in answer_starts if isinstance(s, int)]

        metadata: dict[str, Any] = {
            "dataset_split": split,
            "question_id": str(doc_id),
            "has_answer": len(clean_answers) > 0,
            "context_length": len(clean_context),
            "question_length": len(clean_question),
            "answer_count": len(clean_answers),
            "source_dataset": self.config.dataset_name,
            "created_at": datetime.now(UTC).isoformat(),
            "answer_starts": clean_starts,
        }

        return Document(
            id=str(doc_id),
            title=normalize_whitespace(str(row.get("title", ""))),
            context=clean_context,
            question=clean_question,
            answers=clean_answers,
            metadata=metadata,
        )

    # ------------------------------------------------------------------ utils

    def _resolve_cache_dir(self) -> Path:
        """Resolve and ensure the dataset cache directory.

        Returns:
            The cache directory ``Path``.
        """
        if self.config.cache_dir:
            cache_dir = Path(self.config.cache_dir)
        else:
            cache_dir = self.paths.cache_dir / "datasets"
        return ensure_directory(cache_dir)