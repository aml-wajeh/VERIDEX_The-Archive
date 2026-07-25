"""Centralized application configuration.

Title:
    Centralized Configuration

Description:
    This module is the single source of truth for every configurable value in
    the project: filesystem paths, model names, retrieval hyper-parameters,
    dataset options, application flags, and secrets. All values are loaded from
    environment variables (with a ``.env`` file via ``python-dotenv``),
    validated, and exposed through immutable dataclasses.

    Importing this module produces **no side effects**: environment loading,
    validation, and directory creation only happen when
    :func:`get_settings` is called.

Responsibilities:
    - Auto-detect the project root and derive every directory path.
    - Load and validate environment variables with descriptive errors.
    - Centralize LLM and embedding model names (no hardcoding elsewhere).
    - Centralize dataset source, cache and export defaults.
    - Provide a lazy, cached :func:`get_settings` singleton.
    - Create required directories automatically on first settings build.

Author:
    Author Placeholder
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from . import utils

# Named constants (never magic numbers).
DEFAULT_SEED: int = 42
DEFAULT_LLM_NAME: str = "llama-3.1-8b-instant"
DEFAULT_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_COLLECTION_NAME: str = "squad_v2_chunks"
DEFAULT_TEMPERATURE: float = 0.0
DEFAULT_MAX_TOKENS: int = 512
DEFAULT_TOP_K: int = 5
DEFAULT_CHUNK_SIZE: int = 500
DEFAULT_CHUNK_OVERLAP: int = 50
DEFAULT_LOG_LEVEL: str = "INFO"
DEFAULT_DATASET_NAME: str = "rajpurkar/squad_v2"
DEFAULT_EXPORT_FORMAT: str = "jsonl"
DEFAULT_BATCH_SIZE: int = 1000
APP_NAME: str = "Full RAG Pipeline"
APP_VERSION: str = "0.1.0"
_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


class ConfigurationError(Exception):
    """Raised when configuration is missing, invalid, or inconsistent."""


@dataclass(frozen=True)
class PathConfig:
    """Immutable filesystem layout derived from the project root.

    Attributes:
        project_root: Repository root directory (auto-detected).
        assets_dir: Static assets (images, diagrams).
        data_dir: Top-level data directory.
        raw_dir: Raw, untouched datasets.
        processed_dir: Cleaned / transformed datasets.
        embeddings_dir: Cached embedding artifacts.
        logs_dir: Application log files.
        tmp_dir: Ephemeral scratch files.
        cache_dir: Reusable caches.
        chroma_dir: Persisted Chroma vector store.

    Example:
        >>> from src.config import get_settings  # doctest: +SKIP
        >>> get_settings().paths.raw_dir.name  # doctest: +SKIP
        'raw'
    """

    project_root: Path
    assets_dir: Path
    data_dir: Path
    raw_dir: Path
    processed_dir: Path
    embeddings_dir: Path
    logs_dir: Path
    tmp_dir: Path
    cache_dir: Path
    chroma_dir: Path

    @property
    def squad_train_path(self) -> Path:
        """Return the expected SQuAD v2 training file path.

        Returns:
            ``<raw_dir>/train-v2.0.json``.
        """
        return self.raw_dir / "train-v2.0.json"

    def ensure_directories(self) -> list[Path]:
        """Create every managed directory if it does not already exist.

        Returns:
            The list of directory paths that were ensured.
        """
        managed = (
            self.assets_dir,
            self.data_dir,
            self.raw_dir,
            self.processed_dir,
            self.embeddings_dir,
            self.logs_dir,
            self.tmp_dir,
            self.cache_dir,
            self.chroma_dir,
        )
        return [utils.ensure_directory(directory) for directory in managed]


@dataclass(frozen=True)
class ModelConfig:
    """Model names and generation hyper-parameters.

    Attributes:
        llm_name: Groq chat-completion model identifier.
        embedding_model: Sentence-transformers model identifier.
        temperature: Sampling temperature for generation.
        max_tokens: Maximum generated tokens per response.

    Example:
        >>> from src.config import get_settings  # doctest: +SKIP
        >>> get_settings().models.temperature >= 0  # doctest: +SKIP
        True
    """

    llm_name: str
    embedding_model: str
    temperature: float
    max_tokens: int


@dataclass(frozen=True)
class RetrievalConfig:
    """Chunking and retrieval hyper-parameters.

    Attributes:
        top_k: Number of chunks retrieved per query.
        chunk_size: Chunk length in characters / tokens (pipeline-defined unit).
        chunk_overlap: Overlap between consecutive chunks.
        collection_name: Chroma collection name.

    Example:
        >>> from src.config import get_settings  # doctest: +SKIP
        >>> r = get_settings().retrieval  # doctest: +SKIP
        >>> r.chunk_overlap < r.chunk_size  # doctest: +SKIP
        True
    """

    top_k: int
    chunk_size: int
    chunk_overlap: int
    collection_name: str


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset source, caching and export configuration.

    Attributes:
        dataset_name: Hugging Face dataset identifier (e.g. SQuAD v2).
        dataset_revision: Optional dataset revision / branch; empty for default.
        cache_dir: Optional explicit cache dir; empty uses ``cache/datasets``.
        export_format: Default export format (``jsonl`` / ``csv`` / ``parquet``).
        batch_size: Placeholder batch size for future streaming processing.

    Example:
        >>> from src.config import get_settings  # doctest: +SKIP
        >>> get_settings().dataset.export_format  # doctest: +SKIP
        'jsonl'
    """

    dataset_name: str
    dataset_revision: str
    cache_dir: str
    export_format: str
    batch_size: int


@dataclass(frozen=True)
class AppConfig:
    """Application-wide flags.

    Attributes:
        app_name: Human-readable application name.
        app_version: Semantic version string.
        debug: Whether debug mode is enabled.
        log_level: Active log level name.
        seed: Deterministic seed applied at bootstrap.

    Example:
        >>> from src.config import get_settings  # doctest: +SKIP
        >>> isinstance(get_settings().app.debug, bool)  # doctest: +SKIP
        True
    """

    app_name: str
    app_version: str
    debug: bool
    log_level: str
    seed: int


@dataclass(frozen=True)
class Settings:
    """Top-level, immutable settings container.

    Attributes:
        paths: Filesystem layout.
        models: Model names and generation parameters.
        retrieval: Chunking and retrieval parameters.
        app: Application flags.
        dataset: Dataset source, cache and export defaults.
        groq_api_key: Groq API key (empty string when unset).
        hf_token: Hugging Face token (empty string when unset).

    Example:
        >>> from src.config import get_settings  # doctest: +SKIP
        >>> get_settings().device in {"cuda", "mps", "cpu"}  # doctest: +SKIP
        True
    """

    paths: PathConfig
    models: ModelConfig
    retrieval: RetrievalConfig
    app: AppConfig
    dataset: DatasetConfig
    groq_api_key: str = ""
    hf_token: str = ""

    @property
    def device(self) -> str:
        """Return the detected compute device (``cuda``/``mps``/``cpu``).

        Returns:
            The device string from :func:`src.utils.get_device`.
        """
        return utils.get_device()

    def require_secrets(self, *names: str) -> None:
        """Assert that named secrets are present, raising otherwise.

        Use this at the start of any phase that needs a credential (for example
        generation needs ``groq_api_key``). Building settings does *not* require
        secrets so that offline phases (embedding, indexing) still work.

        Args:
            *names: Secret attribute names on this instance, e.g.
                ``"groq_api_key"``.

        Returns:
            ``None``.

        Raises:
            ConfigurationError: If any requested secret is empty, listing all
                missing names in a single descriptive message.

        Example:
            >>> Settings.from_env().require_secrets()  # no-op when no names
        """
        available = {"groq_api_key": self.groq_api_key, "hf_token": self.hf_token}
        unknown = [name for name in names if name not in available]
        if unknown:
            raise ConfigurationError(f"Unknown secret name(s): {unknown}")

        missing = [name for name in names if not available[name].strip()]
        if missing:
            raise ConfigurationError(
                "Missing required secret(s): "
                f"{missing}. Provide them via the .env file or environment "
                "variables (never hardcode them)."
            )

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from environment variables and ``.env``.

        Returns:
            A fully validated :class:`Settings` instance.

        Raises:
            ConfigurationError: If any value fails type or range validation.
        """
        _ensure_env_loaded()

        project_root = Path(__file__).resolve().parents[1]
        paths = PathConfig(
            project_root=project_root,
            assets_dir=project_root / "assets",
            data_dir=project_root / "data",
            raw_dir=project_root / "data" / "raw",
            processed_dir=project_root / "data" / "processed",
            embeddings_dir=project_root / "data" / "embeddings",
            logs_dir=project_root / "logs",
            tmp_dir=project_root / "tmp",
            cache_dir=project_root / "cache",
            chroma_dir=project_root / "chroma_db",
        )

        temperature = _env_float("TEMPERATURE", DEFAULT_TEMPERATURE)
        if not 0.0 <= temperature <= 2.0:
            raise ConfigurationError(
                f"TEMPERATURE must be within [0.0, 2.0], got {temperature}."
            )

        max_tokens = _env_int("MAX_TOKENS", DEFAULT_MAX_TOKENS)
        if max_tokens <= 0:
            raise ConfigurationError(f"MAX_TOKENS must be > 0, got {max_tokens}.")

        top_k = _env_int("TOP_K", DEFAULT_TOP_K)
        if top_k <= 0:
            raise ConfigurationError(f"TOP_K must be > 0, got {top_k}.")

        chunk_size = _env_int("CHUNK_SIZE", DEFAULT_CHUNK_SIZE)
        if chunk_size <= 0:
            raise ConfigurationError(f"CHUNK_SIZE must be > 0, got {chunk_size}.")

        chunk_overlap = _env_int("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP)
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ConfigurationError(
                "CHUNK_OVERLAP must satisfy 0 <= CHUNK_OVERLAP < CHUNK_SIZE, "
                f"got overlap={chunk_overlap}, size={chunk_size}."
            )

        log_level = _env_str("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()
        if log_level not in _VALID_LOG_LEVELS:
            raise ConfigurationError(
                f"LOG_LEVEL must be one of {sorted(_VALID_LOG_LEVELS)}, "
                f"got '{log_level}'."
            )

        debug = _env_bool("DEBUG", False)

        batch_size = _env_int("BATCH_SIZE", DEFAULT_BATCH_SIZE)
        if batch_size <= 0:
            raise ConfigurationError(f"BATCH_SIZE must be > 0, got {batch_size}.")

        models = ModelConfig(
            llm_name=_env_str("MODEL_NAME", DEFAULT_LLM_NAME),
            embedding_model=_env_str("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        retrieval = RetrievalConfig(
            top_k=top_k,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            collection_name=_env_str("COLLECTION_NAME", DEFAULT_COLLECTION_NAME),
        )
        dataset = DatasetConfig(
            dataset_name=_env_str("DATASET_NAME", DEFAULT_DATASET_NAME),
            dataset_revision=_env_str("DATASET_REVISION", ""),
            cache_dir=_env_str("DATASET_CACHE_DIR", ""),
            export_format=_env_str("EXPORT_FORMAT", DEFAULT_EXPORT_FORMAT),
            batch_size=batch_size,
        )
        app = AppConfig(
            app_name=APP_NAME,
            app_version=APP_VERSION,
            debug=debug,
            log_level=log_level,
            seed=DEFAULT_SEED,
        )

        return cls(
            paths=paths,
            models=models,
            retrieval=retrieval,
            app=app,
            dataset=dataset,
            groq_api_key=_env_str("GROQ_API_KEY", ""),
            hf_token=_env_str("HF_TOKEN", ""),
        )


# ---------------------------------------------------------------------------
# Environment parsing helpers (private, no side effects beyond os.getenv).
# ---------------------------------------------------------------------------


def _env_str(name: str, default: str) -> str:
    """Read a string env var, returning ``default`` when unset/blank.

    Args:
        name: Environment variable name.
        default: Fallback value.

    Returns:
        The trimmed value or ``default``.
    """
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip()


def _env_int(name: str, default: int) -> int:
    """Read an integer env var with a descriptive error on bad input.

    Args:
        name: Environment variable name.
        default: Fallback value when unset/blank.

    Returns:
        The parsed integer.

    Raises:
        ConfigurationError: If the value is not a valid integer.
    """
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigurationError(
            f"Environment variable {name} must be an integer, got '{raw}'."
        ) from exc


def _env_float(name: str, default: float) -> float:
    """Read a float env var with a descriptive error on bad input.

    Args:
        name: Environment variable name.
        default: Fallback value when unset/blank.

    Returns:
        The parsed float.

    Raises:
        ConfigurationError: If the value is not a valid float.
    """
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigurationError(
            f"Environment variable {name} must be a number, got '{raw}'."
        ) from exc


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean env var via :func:`src.utils.parse_bool`.

    Args:
        name: Environment variable name.
        default: Fallback value when unset/blank.

    Returns:
        The parsed boolean.

    Raises:
        ConfigurationError: If the value cannot be interpreted as boolean.
    """
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return utils.parse_bool(raw)
    except ValueError as exc:
        raise ConfigurationError(
            f"Environment variable {name} must be boolean-like, got '{raw}'."
        ) from exc


def _ensure_env_loaded() -> None:
    """Load ``.env`` exactly once (idempotent guard).

    The loaded-state flag is stored on the function object to avoid introducing
    mutable module-level globals for configuration state.
    """
    if getattr(_ensure_env_loaded, "_done", False):
        return
    try:
        from dotenv import load_dotenv

        env_path = Path(__file__).resolve().parents[1] / ".env"
        load_dotenv(dotenv_path=env_path, override=False)
    except ImportError:
        pass
    _ensure_env_loaded._done = True  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Lazy singleton accessor (no work happens at import time).
# ---------------------------------------------------------------------------

_settings: Settings | None = None  # singleton cache; not configuration state.


def get_settings(*, reload: bool = False) -> Settings:
    """Return the cached :class:`Settings`, building it on first call.

    The first call loads environment variables, validates them, and creates all
    managed directories. Subsequent calls return the cached instance unless
    ``reload`` is ``True``.

    Args:
        reload: When ``True``, rebuild settings from the environment even if a
            cached instance exists (useful for tests).

    Returns:
        The application :class:`Settings` instance.

    Example:
        >>> get_settings().app.app_name  # doctest: +SKIP
        'Full RAG Pipeline'
    """
    global _settings
    if _settings is None or reload:
        _settings = Settings.from_env()
        _settings.paths.ensure_directories()
    return _settings