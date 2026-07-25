"""Unit tests for the data loading layer.

Title:
    Data Loader Unit Tests

Description:
    These tests exercise :class:`src.data_loader.DataLoader` using in-memory
    synthetic SQuAD-style records, so they never touch the network and run in
    milliseconds. A single integration test that hits Hugging Face is included
    but skipped unless ``RUN_INTEGRATION_TESTS=1`` is set in the environment.

Responsibilities:
    - Verify record loading, cleaning and metadata keys.
    - Verify ``Document`` (de)serialisation and representation.
    - Verify :class:`DatasetStatistics` aggregation.
    - Verify validation (empty content, malformed rows, missing answers).
    - Verify JSON / JSONL / CSV / Parquet export and failure cases.

Author:
    Author Placeholder
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from src.config import DatasetConfig, PathConfig
from src.data_loader import (
    REQUIRED_METADATA_KEYS,
    DataLoader,
    DatasetExportError,
    DatasetLoadingError,
    DatasetStatistics,
    DatasetValidationError,
    Document,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_paths(tmp_path: Path) -> PathConfig:
    """Build a PathConfig rooted at a temporary directory.

    Args:
        tmp_path: Pytest-provided temporary directory.

    Returns:
        A ``PathConfig`` whose writable dirs point under ``tmp_path``.
    """
    return PathConfig(
        project_root=tmp_path,
        assets_dir=tmp_path / "assets",
        data_dir=tmp_path / "data",
        raw_dir=tmp_path / "data" / "raw",
        processed_dir=tmp_path / "data" / "processed",
        embeddings_dir=tmp_path / "data" / "embeddings",
        logs_dir=tmp_path / "logs",
        tmp_dir=tmp_path / "tmp",
        cache_dir=tmp_path / "cache",
        chroma_dir=tmp_path / "chroma_db",
    )


def _make_config() -> DatasetConfig:
    """Build a DatasetConfig with deterministic test defaults.

    Returns:
        A ``DatasetConfig`` instance.
    """
    return DatasetConfig(
        dataset_name="test/squad_v2",
        dataset_revision="",
        cache_dir="",
        export_format="jsonl",
        batch_size=100,
    )


def _clean_records() -> dict[str, list[dict]]:
    """Return well-formed records across train and validation.

    Returns:
        A mapping with a two-sample ``train`` split and a one-sample
        ``validation`` split (one unanswerable sample included).
    """
    return {
        "train": [
            {
                "id": "1",
                "title": "  Title  ",
                "context": "  Hello   world  ",
                "question": "  What?  ",
                "answers": {"text": ["world"], "answer_start": [6]},
            },
            {
                "id": "2",
                "title": "Title",
                "context": "Some context here.",
                "question": "Unanswerable?",
                "answers": {"text": [], "answer_start": []},
            },
        ],
        "validation": [
            {
                "id": "v1",
                "title": "Title",
                "context": "Val context.",
                "question": "V?",
                "answers": {"text": ["Val"], "answer_start": [0]},
            }
        ],
    }


@pytest.fixture()
def loader(tmp_path: Path) -> DataLoader:
    """Provide a DataLoader wired to a temporary directory.

    Args:
        tmp_path: Pytest-provided temporary directory.

    Returns:
        A fresh ``DataLoader`` (no data loaded yet).
    """
    return DataLoader(_make_config(), _make_paths(tmp_path))


# ---------------------------------------------------------------------------
# Loading & cleaning
# ---------------------------------------------------------------------------


def test_load_from_records_builds_and_cleans(loader: DataLoader) -> None:
    """Records are converted to documents and whitespace is normalised.

    Args:
        loader: The fixture-provided loader.

    Returns:
        None.
    """
    loader.load_from_records(_clean_records(), validate=False)

    docs = loader.documents("train")
    assert len(docs) == 2
    assert docs[0].context == "Hello world"
    assert docs[0].question == "What?"
    assert docs[0].title == "Title"
    assert docs[0].answers == ["world"]


def test_splits_and_num_samples(loader: DataLoader) -> None:
    """Split listing and counts reflect the loaded records.

    Args:
        loader: The fixture-provided loader.

    Returns:
        None.
    """
    loader.load_from_records(_clean_records(), validate=False)
    assert loader.splits() == ["train", "validation"]
    assert loader.num_samples("train") == 2
    assert loader.num_samples("validation") == 1
    assert loader.split_sizes() == {"train": 2, "validation": 1}


# ---------------------------------------------------------------------------
# Document (de)serialisation & repr
# ---------------------------------------------------------------------------


def test_document_to_from_dict_roundtrip(loader: DataLoader) -> None:
    """to_dict and from_dict are inverse operations.

    Args:
        loader: The fixture-provided loader.

    Returns:
        None.
    """
    loader.load_from_records(_clean_records(), validate=False)
    original = loader.documents("train")[0]

    rebuilt = Document.from_dict(original.to_dict())
    assert rebuilt.to_dict() == original.to_dict()


def test_document_repr_contains_id(loader: DataLoader) -> None:
    """The representation contains the id and is compact.

    Args:
        loader: The fixture-provided loader.

    Returns:
        None.
    """
    loader.load_from_records(_clean_records(), validate=False)
    text = repr(loader.documents("train")[0])
    assert "Document(" in text
    assert "'1'" in text
    assert "context_len=" in text


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_metadata_required_keys_present(loader: DataLoader) -> None:
    """Every document carries the eight required metadata keys.

    Args:
        loader: The fixture-provided loader.

    Returns:
        None.
    """
    loader.load_from_records(_clean_records(), validate=False)
    for split in loader.splits():
        for doc in loader.documents(split):
            assert set(REQUIRED_METADATA_KEYS) <= set(doc.metadata)
            assert doc.metadata["dataset_split"] == split
            assert doc.metadata["question_id"] == doc.id
            assert doc.metadata["context_length"] == len(doc.context)
            assert doc.metadata["question_length"] == len(doc.question)
            assert doc.metadata["source_dataset"] == "test/squad_v2"
            assert isinstance(doc.metadata["created_at"], str)


def test_missing_answers_has_answer_false(loader: DataLoader) -> None:
    """Unanswerable samples expose has_answer=False and zero answer count.

    Args:
        loader: The fixture-provided loader.

    Returns:
        None.
    """
    loader.load_from_records(_clean_records(), validate=False)
    unanswerable = loader.documents("train")[1]
    assert unanswerable.answers == []
    assert unanswerable.metadata["has_answer"] is False
    assert unanswerable.metadata["answer_count"] == 0


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def test_compute_statistics_returns_dataclass(loader: DataLoader) -> None:
    """compute_statistics returns a DatasetStatistics with correct values.

    Args:
        loader: The fixture-provided loader.

    Returns:
        None.
    """
    loader.load_from_records(_clean_records(), validate=False)
    stats = loader.compute_statistics()

    assert isinstance(stats, DatasetStatistics)
    assert stats.total_samples == 3
    assert stats.train_samples == 2
    assert stats.validation_samples == 1
    # context lengths: 11, 18, 12
    assert stats.maximum_context_length == 18
    assert stats.minimum_context_length == 11
    assert stats.average_context_length == pytest.approx((11 + 18 + 12) / 3)
    # question lengths: 5, 13, 2
    assert stats.average_question_length == pytest.approx((5 + 13 + 2) / 3)
    # answer counts: 1, 0, 1
    assert stats.average_answer_count == pytest.approx((1 + 0 + 1) / 3)
    assert stats.dataset_name == "test/squad_v2"
    assert stats.dataset_version == "v2"


def test_compute_statistics_before_load_raises(loader: DataLoader) -> None:
    """Computing statistics with no data raises a descriptive error.

    Args:
        loader: The fixture-provided loader.

    Returns:
        None.
    """
    with pytest.raises(DatasetLoadingError):
        loader.compute_statistics()


def test_statistics_per_split(loader: DataLoader) -> None:
    """statistics(split) returns a per-split dict.

    Args:
        loader: The fixture-provided loader.

    Returns:
        None.
    """
    loader.load_from_records(_clean_records(), validate=False)
    stats = loader.statistics("train")
    assert stats["num_samples"] == 2
    assert stats["max_context_length"] == 18
    assert stats["min_context_length"] == 11


def test_answer_availability(loader: DataLoader) -> None:
    """Answer availability reports per-split ratios.

    Args:
        loader: The fixture-provided loader.

    Returns:
        None.
    """
    loader.load_from_records(_clean_records(), validate=False)
    avail = loader.answer_availability()
    assert avail["train"]["answerable"] == 1
    assert avail["train"]["ratio"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validate_passes_on_clean_data(loader: DataLoader) -> None:
    """Clean data passes validation without raising.

    Args:
        loader: The fixture-provided loader.

    Returns:
        None.
    """
    loader.load_from_records(_clean_records(), validate=False)
    loader.validate()


def test_validate_raises_on_empty_context(loader: DataLoader) -> None:
    """An empty context (after cleaning) is rejected by validate().

    Args:
        loader: The fixture-provided loader.

    Returns:
        None.
    """
    records = {
        "train": [
            {
                "id": "bad",
                "title": "T",
                "context": "   ",
                "question": "q?",
                "answers": {"text": ["a"], "answer_start": [0]},
            }
        ]
    }
    loader.load_from_records(records, validate=False)
    with pytest.raises(DatasetValidationError):
        loader.validate()


def test_validate_raises_on_empty_question(loader: DataLoader) -> None:
    """An empty question (after cleaning) is rejected by validate().

    Args:
        loader: The fixture-provided loader.

    Returns:
        None.
    """
    records = {
        "train": [
            {
                "id": "bad",
                "title": "T",
                "context": "ctx",
                "question": "  ",
                "answers": {"text": ["a"], "answer_start": [0]},
            }
        ]
    }
    loader.load_from_records(records, validate=False)
    with pytest.raises(DatasetValidationError):
        loader.validate()


def test_malformed_row_raises_when_validate_true(loader: DataLoader) -> None:
    """A wrong field type raises immediately when validate=True.

    Args:
        loader: The fixture-provided loader.

    Returns:
        None.
    """
    records = {
        "train": [
            {
                "id": "x",
                "title": "T",
                "context": "c",
                "question": "q",
                "answers": "not-a-mapping",
            }
        ]
    }
    with pytest.raises(DatasetValidationError):
        loader.load_from_records(records, validate=True)


def test_malformed_row_skipped_when_validate_false(loader: DataLoader) -> None:
    """A wrong field type is skipped (not raised) when validate=False.

    Args:
        loader: The fixture-provided loader.

    Returns:
        None.
    """
    records = {
        "train": [
            {
                "id": "good",
                "title": "T",
                "context": "c",
                "question": "q",
                "answers": {"text": ["a"], "answer_start": [0]},
            },
            {
                "id": "bad",
                "title": "T",
                "context": "c",
                "question": "q",
                "answers": "not-a-mapping",
            },
        ]
    }
    loader.load_from_records(records, validate=False)
    assert loader.num_samples("train") == 1


def test_validate_before_load_raises(loader: DataLoader) -> None:
    """Validating with no loaded data raises a descriptive error.

    Args:
        loader: The fixture-provided loader.

    Returns:
        None.
    """
    with pytest.raises(DatasetLoadingError):
        loader.validate()


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def test_export_json(loader: DataLoader, tmp_path: Path) -> None:
    """JSON export writes a readable array of documents.

    Args:
        loader: The fixture-provided loader.
        tmp_path: Pytest-provided temporary directory.

    Returns:
        None.
    """
    loader.load_from_records(_clean_records(), validate=False)
    out = loader.export("train", fmt="json", output_dir=tmp_path)

    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["id"] == "1"
    assert data[0]["context"] == "Hello world"


def test_export_jsonl(loader: DataLoader, tmp_path: Path) -> None:
    """JSONL export writes one object per line.

    Args:
        loader: The fixture-provided loader.
        tmp_path: Pytest-provided temporary directory.

    Returns:
        None.
    """
    loader.load_from_records(_clean_records(), validate=False)
    out = loader.export("train", fmt="jsonl", output_dir=tmp_path)

    assert out.exists()
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["id"] == "1"


def test_export_csv(loader: DataLoader, tmp_path: Path) -> None:
    """CSV export writes a header and one row per document.

    Args:
        loader: The fixture-provided loader.
        tmp_path: Pytest-provided temporary directory.

    Returns:
        None.
    """
    loader.load_from_records(_clean_records(), validate=False)
    out = loader.export("train", fmt="csv", output_dir=tmp_path)

    assert out.exists()
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("id,title,context")
    assert len(lines) == 3


def test_export_parquet_skips_without_engine(
    loader: DataLoader, tmp_path: Path
) -> None:
    """Parquet export succeeds when an engine is present, else fails cleanly.

    Args:
        loader: The fixture-provided loader.
        tmp_path: Pytest-provided temporary directory.

    Returns:
        None.
    """
    loader.load_from_records(_clean_records(), validate=False)
    try:
        out = loader.export("train", fmt="parquet", output_dir=tmp_path)
    except DatasetExportError:
        return
    assert out.exists()
    assert out.stat().st_size > 0


def test_export_unsupported_format_raises(loader: DataLoader, tmp_path: Path) -> None:
    """An unknown format raises DatasetExportError.

    Args:
        loader: The fixture-provided loader.
        tmp_path: Pytest-provided temporary directory.

    Returns:
        None.
    """
    loader.load_from_records(_clean_records(), validate=False)
    with pytest.raises(DatasetExportError):
        loader.export("train", fmt="xml", output_dir=tmp_path)


def test_documents_unknown_split_raises(loader: DataLoader) -> None:
    """Requesting an unloaded split raises a descriptive error.

    Args:
        loader: The fixture-provided loader.

    Returns:
        None.
    """
    loader.load_from_records(_clean_records(), validate=False)
    with pytest.raises(DatasetLoadingError):
        loader.documents("test")


# ---------------------------------------------------------------------------
# Integration (network) - skipped by default
# ---------------------------------------------------------------------------


def test_load_from_huggingface_integration(loader: DataLoader) -> None:
    """Integration: real Hugging Face load (skipped unless opted in).

    Set ``RUN_INTEGRATION_TESTS=1`` to execute this against the network.

    Args:
        loader: The fixture-provided loader.

    Returns:
        None.
    """
    if os.getenv("RUN_INTEGRATION_TESTS") != "1":
        pytest.skip("integration tests disabled; set RUN_INTEGRATION_TESTS=1")

    real_loader = DataLoader(
        DatasetConfig(
            dataset_name="rajpurkar/squad_v2",
            dataset_revision="",
            cache_dir="",
            export_format="jsonl",
            batch_size=100,
        ),
        loader.paths,
    )
    try:
        real_loader.load()
    except DatasetLoadingError:
        pytest.skip("network / hub unavailable during integration test")

    assert "train" in real_loader.splits()
    assert real_loader.num_samples("train") > 0
    assert isinstance(real_loader.compute_statistics(), DatasetStatistics)
