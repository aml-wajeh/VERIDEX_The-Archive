"""Centralized enterprise logging configuration.

Title:
    Centralized Logging Configuration

Description:
    This module provides a single, consistent logging setup for the whole
    project. It configures a root logger with a colorized console handler (when
    ``colorlog`` is installed) and a rotating file handler that writes into the
    ``logs/`` directory, which is created automatically.

    Production code must obtain loggers through :func:`get_logger` and must
    never use ``print`` for diagnostics.

Responsibilities:
    - Configure console and rotating-file logging handlers.
    - Resolve the active log level from arguments or the environment.
    - Create the ``logs/`` directory automatically when file logging is on.
    - Provide colorized output when ``colorlog`` is available, plain otherwise.
    - Expose :func:`get_logger` for named module loggers.

Author:
    Author Placeholder
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
COLOR_LOG_FORMAT = (
    "%(log_color)s%(levelname)-8s%(reset)s | %(asctime)s | %(name)s | %(message)s"
)
DEFAULT_LOG_FILENAME = "app.log"
MAX_LOG_BYTES = 5 * 1024 * 1024  # 5 MB per log file before rotation.
LOG_BACKUP_COUNT = 3
_VALID_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def _default_log_dir() -> Path:
    """Return the default ``logs/`` directory next to the project root.

    Returns:
        The ``Path`` to ``<project_root>/logs``.
    """
    return Path(__file__).resolve().parents[1] / "logs"


def _resolve_level(level: str | int | None) -> int:
    """Resolve a log level argument into a numeric ``logging`` level.

    Args:
        level: A level name (case-insensitive), a numeric level, or ``None`` to
            read ``LOG_LEVEL`` from the environment (defaulting to ``INFO``).

    Returns:
        The numeric logging level.

    Raises:
        ValueError: If a string level name is not one of the valid levels.
    """
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")

    if isinstance(level, int):
        return level

    name = str(level).strip().upper()
    if name not in _VALID_LEVELS:
        raise ValueError(
            f"Invalid log level '{level}'. Expected one of {sorted(_VALID_LEVELS)}."
        )
    return getattr(logging, name)


def _load_env_if_present() -> None:
    """Load ``.env`` if python-dotenv is installed (idempotent, best-effort).

    This guarantees ``LOG_LEVEL`` is available even when ``configure_logging``
    runs before :func:`src.config.get_settings`.
    """
    try:
        from dotenv import load_dotenv

        env_path = Path(__file__).resolve().parents[1] / ".env"
        load_dotenv(dotenv_path=env_path, override=False)
    except ImportError:
        # python-dotenv is a production dependency; this branch only protects
        # against unusual minimal environments.
        pass


def _build_console_handler(level: int) -> logging.Handler:
    """Build a stream handler, colorized when ``colorlog`` is available.

    Args:
        level: Handler logging level.

    Returns:
        A configured ``logging.Handler`` writing to stderr/stdout.
    """
    handler = logging.StreamHandler()
    handler.setLevel(level)

    # Declare the common base type up front so the try/except branches can
    # each assign a compatible formatter without confusing the type checker.
    # ``ColoredFormatter`` is a subclass of ``logging.Formatter``.
    formatter: logging.Formatter
    try:
        import colorlog  # type: ignore[import-not-found]

        formatter = colorlog.ColoredFormatter(
            COLOR_LOG_FORMAT,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )
    except ImportError:
        formatter = logging.Formatter(DEFAULT_LOG_FORMAT)

    handler.setFormatter(formatter)
    return handler


def _build_file_handler(level: int, log_path: Path) -> logging.Handler:
    """Build a rotating file handler.

    Args:
        level: Handler logging level.
        log_path: Destination log file path.

    Returns:
        A configured ``RotatingFileHandler``.
    """
    handler = RotatingFileHandler(
        filename=log_path,
        maxBytes=MAX_LOG_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
    return handler


def configure_logging(
    *,
    level: str | int | None = None,
    log_dir: str | Path | None = None,
    log_filename: str = DEFAULT_LOG_FILENAME,
    console: bool = True,
    file: bool = True,
) -> logging.Logger:
    """Configure the root logger for the application.

    The configuration is idempotent: calling this function repeatedly replaces
    the root handlers instead of stacking duplicates.

    Args:
        level: Log level name, numeric level, or ``None`` to read ``LOG_LEVEL``
            from the environment.
        log_dir: Directory for the log file. Defaults to ``<root>/logs`` and is
            created automatically when ``file`` is ``True``.
        log_filename: Name of the log file inside ``log_dir``.
        console: Whether to attach a console handler.
        file: Whether to attach a rotating file handler.

    Returns:
        The configured root ``logging.Logger``.

    Raises:
        ValueError: If ``level`` resolves to an invalid level name.

    Example:
        >>> configure_logging(level="DEBUG")  # doctest: +SKIP
        <RootLogger root (DEBUG)>
    """
    _load_env_if_present()

    numeric_level = _resolve_level(level)
    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Idempotency: clear previously attached handlers before rebuilding.
    for existing in list(root.handlers):
        root.removeHandler(existing)

    if console:
        root.addHandler(_build_console_handler(numeric_level))

    if file:
        directory = Path(log_dir) if log_dir is not None else _default_log_dir()
        directory.mkdir(parents=True, exist_ok=True)
        root.addHandler(_build_file_handler(numeric_level, directory / log_filename))

    return root


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.

    Modules should call this with ``__name__`` so log records carry the module
    path, which makes filtering and tracing straightforward.

    Args:
        name: Logger name, typically ``__name__``.

    Returns:
        A ``logging.Logger`` instance bound to ``name``.

    Example:
        >>> get_logger(__name__).name  # doctest: +SKIP
        'src.logger'
    """
    return logging.getLogger(name)
