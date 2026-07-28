"""Application configuration for the RAG pipeline."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Base paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_CHUNKS_DIR = PROCESSED_DATA_DIR / "chunks"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
MODEL_CACHE_DIR = DATA_DIR / "models" / "sentence_transformers"

ASSETS_DIR = BASE_DIR / "assets"
RAW_DIR = DATA_DIR / "raw"
LOGS_DIR = BASE_DIR / "logs"
TMP_DIR = BASE_DIR / "tmp"
CACHE_DIR = DATA_DIR / "cache"
CHROMA_DIR = BASE_DIR / "chroma_db"

CHROMA_DB_DIR: Path = CHROMA_DIR
DEFAULT_COLLECTION_NAME: str = "squad_v2_rag"
DEFAULT_TOP_K: int = 4

DEFAULT_MODEL_NAME: str = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE: float = 0.0
DEFAULT_MAX_TOKENS: int = 512
GROQ_API_KEY_ENV: str = "GROQ_API_KEY"

# ---------------------------------------------------------------------------
# Literal types
# ---------------------------------------------------------------------------
ChunkStrategy = Literal[
    "recursive_character",
    "character",
    "sentence",
    "semantic",
]
ExportFormat = Literal[
    "json",
    "jsonl",
    "csv",
    "parquet",
]
EmbeddingExportFormat = Literal[
    "npy",
    "pickle",
    "parquet",
    "json",
]

_ALLOWED_CHUNK_STRATEGIES = frozenset(
    {
        "recursive_character",
        "character",
        "sentence",
        "semantic",
    }
)
_ALLOWED_EXPORT_FORMATS = frozenset(
    {
        "json",
        "jsonl",
        "csv",
        "parquet",
    }
)
_ALLOWED_EMBEDDING_EXPORT_FORMATS = frozenset(
    {
        "npy",
        "pickle",
        "parquet",
        "json",
    }
)
_ALLOWED_HNSW_SPACES = frozenset({"cosine", "l2", "ip"})


# ---------------------------------------------------------------------------
# Chunking configuration (Phase 4)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ChunkingConfig:
    """Configuration for text processing, chunking, and export.

    Attributes:
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.
        chunk_strategy: Chunk splitting strategy.
        export_format: Default export format.
        enable_cleaning: Whether to apply text cleaning.
        enable_normalization: Whether to apply text normalization.
        future_semantic_chunking: Reserved flag for future semantic chunking.
        processed_chunks_dir: Output directory for exported chunks.
        recursive_separators: Separators used by recursive character chunking.
        sentence_endings: Single-character sentence ending markers.
        min_chunk_length: Minimum accepted chunk length after splitting.
    """

    chunk_size: int = 500
    chunk_overlap: int = 100
    chunk_strategy: ChunkStrategy = "recursive_character"
    export_format: ExportFormat = "jsonl"
    enable_cleaning: bool = True
    enable_normalization: bool = True
    future_semantic_chunking: bool = False
    processed_chunks_dir: Path = PROCESSED_CHUNKS_DIR
    recursive_separators: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")
    sentence_endings: tuple[str, ...] = (".", "?", "!")
    min_chunk_length: int = 1

    def __post_init__(self) -> None:
        """Validate and normalize configuration values."""
        if isinstance(self.processed_chunks_dir, str):
            object.__setattr__(
                self,
                "processed_chunks_dir",
                Path(self.processed_chunks_dir),
            )

        if not isinstance(self.processed_chunks_dir, Path):
            object.__setattr__(
                self,
                "processed_chunks_dir",
                Path(str(self.processed_chunks_dir)),
            )

        if isinstance(self.recursive_separators, str):
            object.__setattr__(
                self,
                "recursive_separators",
                (self.recursive_separators,),
            )

        if not isinstance(self.recursive_separators, tuple):
            object.__setattr__(
                self,
                "recursive_separators",
                tuple(self.recursive_separators),
            )

        if isinstance(self.sentence_endings, str):
            object.__setattr__(
                self,
                "sentence_endings",
                (self.sentence_endings,),
            )

        if not isinstance(self.sentence_endings, tuple):
            object.__setattr__(
                self,
                "sentence_endings",
                tuple(self.sentence_endings),
            )

        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be zero or positive.")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        if self.chunk_strategy not in _ALLOWED_CHUNK_STRATEGIES:
            raise ValueError("Unsupported chunk_strategy.")

        if self.export_format not in _ALLOWED_EXPORT_FORMATS:
            raise ValueError("Unsupported export_format.")

        if self.min_chunk_length <= 0 or self.min_chunk_length > self.chunk_size:
            raise ValueError("min_chunk_length must be between 1 and chunk_size.")

        if not self.recursive_separators:
            raise ValueError("recursive_separators must not be empty.")

        if not self.sentence_endings:
            raise ValueError("sentence_endings must not be empty.")

        if any(len(marker) != 1 for marker in self.sentence_endings):
            raise ValueError("sentence_endings must contain single-character markers.")


# ---------------------------------------------------------------------------
# Embedding configuration (Phase 5)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for embedding generation and export.

    Attributes:
        embedding_model: Sentence Transformers model name.
        batch_size: Number of texts encoded per batch.
        normalize_embeddings: Whether to L2-normalize embeddings.
        device: Torch device. If None or "auto", CUDA is used when available.
        cache_folder: Optional cache directory for model artifacts.
        export_format: Default embedding export format.
        future_quantization: Reserved flag for future quantization support.
        embeddings_dir: Output directory for exported embeddings.
    """

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    batch_size: int = 32
    normalize_embeddings: bool = True
    device: str | None = None
    cache_folder: Path | None = MODEL_CACHE_DIR
    export_format: EmbeddingExportFormat = "npy"
    future_quantization: bool = False
    embeddings_dir: Path = EMBEDDINGS_DIR

    def __post_init__(self) -> None:
        """Validate and normalize embedding configuration values."""
        if isinstance(self.cache_folder, str):
            object.__setattr__(
                self,
                "cache_folder",
                Path(self.cache_folder),
            )

        if self.cache_folder is not None and not isinstance(
            self.cache_folder,
            Path,
        ):
            object.__setattr__(
                self,
                "cache_folder",
                Path(str(self.cache_folder)),
            )

        if isinstance(self.embeddings_dir, str):
            object.__setattr__(
                self,
                "embeddings_dir",
                Path(self.embeddings_dir),
            )

        if not isinstance(self.embeddings_dir, Path):
            object.__setattr__(
                self,
                "embeddings_dir",
                Path(str(self.embeddings_dir)),
            )

        if self.device is not None and not isinstance(self.device, str):
            object.__setattr__(self, "device", str(self.device))

        if self.device == "":
            object.__setattr__(self, "device", None)

        if not self.embedding_model.strip():
            raise ValueError("embedding_model must not be empty.")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        if self.export_format not in _ALLOWED_EMBEDDING_EXPORT_FORMATS:
            raise ValueError("Unsupported embedding export_format.")


# ---------------------------------------------------------------------------
# Vector store configuration (Phase 6)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VectorStoreConfig:
    """Configuration for the Chroma vector store.

    Attributes:
        persist_directory: Directory for a persistent store. ``None`` selects
            an in-memory (ephemeral) store, which is ideal for tests.
        collection_name: Chroma collection name.
        hnsw_space: Distance space baked into the collection at creation time.
        add_batch_size: Number of vectors written per ``collection.add`` call.
    """

    persist_directory: Path | None = CHROMA_DB_DIR
    collection_name: str = DEFAULT_COLLECTION_NAME
    hnsw_space: str = "cosine"
    add_batch_size: int = 1000

    def __post_init__(self) -> None:
        """Validate and normalize vector store configuration values."""
        if isinstance(self.persist_directory, str):
            object.__setattr__(
                self,
                "persist_directory",
                Path(self.persist_directory),
            )

        if self.persist_directory is not None and not isinstance(
            self.persist_directory,
            Path,
        ):
            object.__setattr__(
                self,
                "persist_directory",
                Path(str(self.persist_directory)),
            )

        if not self.collection_name.strip():
            raise ValueError("collection_name must not be empty.")

        if self.hnsw_space not in _ALLOWED_HNSW_SPACES:
            raise ValueError("Unsupported hnsw_space.")

        if self.add_batch_size <= 0:
            raise ValueError("add_batch_size must be greater than zero.")


# ---------------------------------------------------------------------------
# Retriever configuration (Phase 7)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RetrieverConfig:
    """Configuration for similarity retrieval.

    Attributes:
        top_k: Default number of chunks to retrieve per query.
        min_similarity: Optional similarity floor; hits below it are dropped.
            ``None`` disables filtering. Valid range for the cosine space is
            ``[-1, 1]``.
    """

    top_k: int = DEFAULT_TOP_K
    min_similarity: float | None = None

    def __post_init__(self) -> None:
        """Validate retriever configuration values."""
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if self.min_similarity is not None and not (-1.0 <= self.min_similarity <= 1.0):
            raise ValueError("min_similarity must be within [-1, 1] or None.")


# ---------------------------------------------------------------------------
# LLM configuration (Phase 8)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LLMConfig:
    """Configuration for the generation (LLM) step.

    The ``src`` package talks to Groq through the official ``groq`` SDK
    directly (no LangChain), keeping the package framework-agnostic. The API
    key is never stored here; it is read from the environment variable named
    by ``api_key_env`` (or injected explicitly at client-construction time).

    Attributes:
        model_name: Groq model identifier.
        temperature: Sampling temperature (0.0 = deterministic).
        max_tokens: Optional maximum number of tokens to generate.
        api_key_env: Name of the environment variable holding the API key.
        request_timeout: Optional per-request timeout in seconds.
    """

    model_name: str = DEFAULT_MODEL_NAME
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int | None = DEFAULT_MAX_TOKENS
    api_key_env: str = GROQ_API_KEY_ENV
    request_timeout: float | None = None

    def __post_init__(self) -> None:
        """Validate LLM configuration values."""
        if not self.model_name.strip():
            raise ValueError("model_name must not be empty.")

        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be within [0, 2].")

        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero or None.")

        if not self.api_key_env.strip():
            raise ValueError("api_key_env must not be empty.")

        if self.request_timeout is not None and self.request_timeout <= 0:
            raise ValueError("request_timeout must be greater than zero or None.")


# ---------------------------------------------------------------------------
# Evaluation configuration (Phase 9)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for the evaluation engine.

    Attributes:
        top_k: Number of chunks the pipeline retrieves per evaluated question.
        max_questions: Optional cap on how many questions are evaluated (the
            first ``max_questions`` items are taken; ``None`` evaluates all).
            The caller is responsible for any random sampling before passing
            the questions in, so evaluation stays deterministic.
        include_unanswerable: Whether to evaluate unanswerable questions (the
            SQuAD v2 "refusal" cases). When ``False`` they are skipped.
    """

    top_k: int = DEFAULT_TOP_K
    max_questions: int | None = None
    include_unanswerable: bool = True

    def __post_init__(self) -> None:
        """Validate evaluation configuration values."""
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if self.max_questions is not None and self.max_questions <= 0:
            raise ValueError("max_questions must be greater than zero or None.")


# ---------------------------------------------------------------------------
# Path configuration (data-loading layer)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PathConfig:
    """Centralised filesystem layout for the whole project.

    Attributes:
        project_root: Repository / project root directory.
        assets_dir: Static assets directory.
        data_dir: Top-level data directory.
        raw_dir: Raw, untouched dataset dumps.
        processed_dir: Cleaned / processed artefacts.
        embeddings_dir: Persisted embedding vectors.
        logs_dir: Log files.
        tmp_dir: Scratch / temporary files.
        cache_dir: Generic cache (datasets, models, ...).
        chroma_dir: Persistent ChromaDB vector-store directory.
    """

    project_root: Path = BASE_DIR
    assets_dir: Path = ASSETS_DIR
    data_dir: Path = DATA_DIR
    raw_dir: Path = RAW_DIR
    processed_dir: Path = PROCESSED_DATA_DIR
    embeddings_dir: Path = EMBEDDINGS_DIR
    logs_dir: Path = LOGS_DIR
    tmp_dir: Path = TMP_DIR
    cache_dir: Path = CACHE_DIR
    chroma_dir: Path = CHROMA_DIR

    def __post_init__(self) -> None:
        """Coerce any string path fields to ``Path`` instances."""
        for field_def in dataclasses.fields(self):
            value = getattr(self, field_def.name)
            if isinstance(value, str):
                object.__setattr__(self, field_def.name, Path(value))


# ---------------------------------------------------------------------------
# Dataset configuration (data-loading layer)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for dataset ingestion.

    Attributes:
        dataset_name: HuggingFace dataset identifier (e.g. ``rajpurkar/squad_v2``).
        dataset_revision: Optional dataset revision / commit / branch.
        cache_dir: Optional local cache directory for the downloaded dataset.
        export_format: Default serialisation format for exported documents.
        batch_size: Processing / export batch size.
    """

    dataset_name: str = "rajpurkar/squad_v2"
    dataset_revision: str = ""
    cache_dir: str = ""
    export_format: str = "jsonl"
    batch_size: int = 100

    def __post_init__(self) -> None:
        """Validate dataset configuration values."""
        if not self.dataset_name.strip():
            raise ValueError("dataset_name must not be empty.")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")


# ---------------------------------------------------------------------------
# Top-level application configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration.

    Attributes:
        chunking: Chunking and text-processing configuration.
        embedding: Embedding-generation configuration.
        vector_store: Vector-store configuration.
        retriever: Retriever configuration.
        llm: LLM / generation configuration.
        evaluation: Evaluation configuration.
    """

    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


# ---------------------------------------------------------------------------
# Unified settings facade (used by notebooks & docs)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Settings:
    """Convenience facade exposing every config block from one object.

    Attributes:
        dataset: Dataset-ingestion configuration.
        paths: Filesystem layout configuration.
        chunking: Chunking / text-processing configuration.
        embedding: Embedding-generation configuration.
        vector_store: Vector-store configuration.
        retriever: Retriever configuration.
        llm: LLM / generation configuration.
        evaluation: Evaluation configuration.
    """

    dataset: DatasetConfig
    paths: PathConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------
CONFIG = AppConfig()
CHUNKING_CONFIG = CONFIG.chunking
EMBEDDING_CONFIG = CONFIG.embedding
VECTOR_STORE_CONFIG = CONFIG.vector_store
RETRIEVER_CONFIG = CONFIG.retriever
LLM_CONFIG = CONFIG.llm
EVALUATION_CONFIG = CONFIG.evaluation
DATASET_CONFIG = DatasetConfig()
PATH_CONFIG = PathConfig()


def get_settings() -> Settings:
    """Return a unified `Settings` object built from the module singletons.

    Returns:
        A `Settings` instance wiring together every configuration block.
    """
    return Settings(
        dataset=DATASET_CONFIG,
        paths=PATH_CONFIG,
        chunking=CHUNKING_CONFIG,
        embedding=EMBEDDING_CONFIG,
        vector_store=VECTOR_STORE_CONFIG,
        retriever=RETRIEVER_CONFIG,
        llm=LLM_CONFIG,
        evaluation=EVALUATION_CONFIG,
    )


# ---------------------------------------------------------------------------
# Resolvers
# ---------------------------------------------------------------------------
def resolve_chunking_config(
    config: ChunkingConfig | AppConfig | Any | None = None,
) -> ChunkingConfig:
    """Resolve a user-supplied configuration object to `ChunkingConfig`.

    Args:
        config: A configuration object, mapping, or None.

    Returns:
        A concrete `ChunkingConfig` instance.

    Raises:
        ValueError: If configuration values are invalid.
    """
    if config is None:
        return CHUNKING_CONFIG

    if isinstance(config, ChunkingConfig):
        return config

    if isinstance(config, AppConfig):
        return config.chunking

    nested_config = getattr(config, "chunking", None)
    if isinstance(nested_config, ChunkingConfig):
        return nested_config

    allowed_fields = {f.name for f in dataclasses.fields(ChunkingConfig)}

    if isinstance(config, dict):
        kwargs = {key: value for key, value in config.items() if key in allowed_fields}
    else:
        kwargs = {
            name: getattr(config, name)
            for name in allowed_fields
            if hasattr(config, name)
        }

    if not kwargs:
        return CHUNKING_CONFIG

    return ChunkingConfig(**kwargs)


def resolve_embedding_config(
    config: EmbeddingConfig | AppConfig | Any | None = None,
) -> EmbeddingConfig:
    """Resolve a user-supplied configuration object to `EmbeddingConfig`.

    Args:
        config: A configuration object, mapping, or None.

    Returns:
        A concrete `EmbeddingConfig` instance.

    Raises:
        ValueError: If configuration values are invalid.
    """
    if config is None:
        return EMBEDDING_CONFIG

    if isinstance(config, EmbeddingConfig):
        return config

    if isinstance(config, AppConfig):
        return config.embedding

    nested_config = getattr(config, "embedding", None)
    if isinstance(nested_config, EmbeddingConfig):
        return nested_config

    allowed_fields = {f.name for f in dataclasses.fields(EmbeddingConfig)}

    if isinstance(config, dict):
        kwargs = {key: value for key, value in config.items() if key in allowed_fields}
    else:
        kwargs = {
            name: getattr(config, name)
            for name in allowed_fields
            if hasattr(config, name)
        }

    if not kwargs:
        return EMBEDDING_CONFIG

    return EmbeddingConfig(**kwargs)


def resolve_vector_store_config(
    config: VectorStoreConfig | AppConfig | Any | None = None,
) -> VectorStoreConfig:
    """Resolve a user-supplied configuration object to `VectorStoreConfig`.

    Args:
        config: A configuration object, mapping, or None.

    Returns:
        A concrete `VectorStoreConfig` instance.

    Raises:
        ValueError: If configuration values are invalid.
    """
    if config is None:
        return VECTOR_STORE_CONFIG

    if isinstance(config, VectorStoreConfig):
        return config

    if isinstance(config, AppConfig):
        return config.vector_store

    nested_config = getattr(config, "vector_store", None)
    if isinstance(nested_config, VectorStoreConfig):
        return nested_config

    allowed_fields = {f.name for f in dataclasses.fields(VectorStoreConfig)}

    if isinstance(config, dict):
        kwargs = {key: value for key, value in config.items() if key in allowed_fields}
    else:
        kwargs = {
            name: getattr(config, name)
            for name in allowed_fields
            if hasattr(config, name)
        }

    if not kwargs:
        return VECTOR_STORE_CONFIG

    return VectorStoreConfig(**kwargs)


def resolve_retriever_config(
    config: RetrieverConfig | AppConfig | Any | None = None,
) -> RetrieverConfig:
    """Resolve a user-supplied configuration object to `RetrieverConfig`.

    Args:
        config: A configuration object, mapping, or None.

    Returns:
        A concrete `RetrieverConfig` instance.

    Raises:
        ValueError: If configuration values are invalid.
    """
    if config is None:
        return RETRIEVER_CONFIG

    if isinstance(config, RetrieverConfig):
        return config

    if isinstance(config, AppConfig):
        return config.retriever

    nested_config = getattr(config, "retriever", None)
    if isinstance(nested_config, RetrieverConfig):
        return nested_config

    allowed_fields = {f.name for f in dataclasses.fields(RetrieverConfig)}

    if isinstance(config, dict):
        kwargs = {key: value for key, value in config.items() if key in allowed_fields}
    else:
        kwargs = {
            name: getattr(config, name)
            for name in allowed_fields
            if hasattr(config, name)
        }

    if not kwargs:
        return RETRIEVER_CONFIG

    return RetrieverConfig(**kwargs)


def resolve_llm_config(
    config: LLMConfig | AppConfig | Any | None = None,
) -> LLMConfig:
    """Resolve a user-supplied configuration object to `LLMConfig`.

    Args:
        config: A configuration object, mapping, or None.

    Returns:
        A concrete `LLMConfig` instance.

    Raises:
        ValueError: If configuration values are invalid.
    """
    if config is None:
        return LLM_CONFIG

    if isinstance(config, LLMConfig):
        return config

    if isinstance(config, AppConfig):
        return config.llm

    nested_config = getattr(config, "llm", None)
    if isinstance(nested_config, LLMConfig):
        return nested_config

    allowed_fields = {f.name for f in dataclasses.fields(LLMConfig)}

    if isinstance(config, dict):
        kwargs = {key: value for key, value in config.items() if key in allowed_fields}
    else:
        kwargs = {
            name: getattr(config, name)
            for name in allowed_fields
            if hasattr(config, name)
        }

    if not kwargs:
        return LLM_CONFIG

    return LLMConfig(**kwargs)


def resolve_evaluation_config(
    config: EvaluationConfig | AppConfig | Any | None = None,
) -> EvaluationConfig:
    """Resolve a user-supplied configuration object to `EvaluationConfig`.

    Args:
        config: A configuration object, mapping, or None.

    Returns:
        A concrete `EvaluationConfig` instance.

    Raises:
        ValueError: If configuration values are invalid.
    """
    if config is None:
        return EVALUATION_CONFIG

    if isinstance(config, EvaluationConfig):
        return config

    if isinstance(config, AppConfig):
        return config.evaluation

    nested_config = getattr(config, "evaluation", None)
    if isinstance(nested_config, EvaluationConfig):
        return nested_config

    allowed_fields = {f.name for f in dataclasses.fields(EvaluationConfig)}

    if isinstance(config, dict):
        kwargs = {key: value for key, value in config.items() if key in allowed_fields}
    else:
        kwargs = {
            name: getattr(config, name)
            for name in allowed_fields
            if hasattr(config, name)
        }

    if not kwargs:
        return EVALUATION_CONFIG

    return EvaluationConfig(**kwargs)


def resolve_dataset_config(
    config: DatasetConfig | Any | None = None,
) -> DatasetConfig:
    """Resolve a user-supplied object to `DatasetConfig`.

    Args:
        config: A configuration object, mapping, or None.

    Returns:
        A concrete `DatasetConfig` instance.

    Raises:
        ValueError: If configuration values are invalid.
    """
    if config is None:
        return DATASET_CONFIG

    if isinstance(config, DatasetConfig):
        return config

    allowed_fields = {f.name for f in dataclasses.fields(DatasetConfig)}

    if isinstance(config, dict):
        kwargs = {key: value for key, value in config.items() if key in allowed_fields}
    else:
        kwargs = {
            name: getattr(config, name)
            for name in allowed_fields
            if hasattr(config, name)
        }

    if not kwargs:
        return DATASET_CONFIG

    return DatasetConfig(**kwargs)


def resolve_path_config(
    config: PathConfig | Any | None = None,
) -> PathConfig:
    """Resolve a user-supplied object to `PathConfig`.

    Args:
        config: A configuration object, mapping, or None.

    Returns:
        A concrete `PathConfig` instance.
    """
    if config is None:
        return PATH_CONFIG

    if isinstance(config, PathConfig):
        return config

    allowed_fields = {f.name for f in dataclasses.fields(PathConfig)}

    if isinstance(config, dict):
        kwargs = {key: value for key, value in config.items() if key in allowed_fields}
    else:
        kwargs = {
            name: getattr(config, name)
            for name in allowed_fields
            if hasattr(config, name)
        }

    if not kwargs:
        return PATH_CONFIG

    return PathConfig(**kwargs)
