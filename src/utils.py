"""Generic utility helpers.

Title:
    Generic Utility Helpers

Description:
    This module contains reusable, framework-agnostic helper functions that do
    not encode any RAG business logic. It intentionally avoids importing any
    other ``src`` module so it can be used (and tested) in isolation and so that
    no circular imports can ever form.

Responsibilities:
    - Filesystem helpers (directory creation, safe path resolution).
    - Environment validation helpers.
    - Deterministic seed initialization across random engines.
    - Compute-device detection (CUDA / MPS / CPU).
    - Human-readable formatting (file sizes, durations, banners).
    - Boolean parsing from environment-style strings.
    - Conservative text cleaning (whitespace normalisation) and filename
      sanitisation, reused by the data layer and later stages.

Author:
    Aml
"""

from __future__ import annotations

import os
import random
import re
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path


def ensure_directory(path: str | Path) -> Path:
    """Create a directory and all missing parents.

    The operation is idempotent: existing directories are left untouched.

    Args:
        path: Directory path to create.

    Returns:
        The resolved ``Path`` of the directory.

    Raises:
        OSError: If the directory cannot be created (for example, a permission
            error or a file already existing at that path).

    Example:
        >>> ensure_directory("data/processed")  # doctest: +SKIP
        PosixPath('.../data/processed')
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def safe_resolve_path(
    path: str | Path,
    *,
    base: str | Path | None = None,
    must_exist: bool = False,
) -> Path:
    """Resolve a path safely against an optional base directory.

    When ``base`` is provided the resolved path is checked to remain inside the
    resolved base, which prevents accidental path-traversal outside the project
    tree. This is a defensive helper for user-supplied paths.

    Args:
        path: Path to resolve.
        base: Optional base directory the result must stay within.
        must_exist: When ``True``, raise if the resolved path does not exist.

    Returns:
        The absolute, resolved ``Path``.

    Raises:
        FileNotFoundError: If ``must_exist`` is ``True`` and the path is absent.
        ValueError: If the resolved path escapes the provided ``base``.

    Example:
        >>> safe_resolve_path("data/raw/train.json", base=".")  # doctest: +SKIP
        PosixPath('.../data/raw/train.json')
    """
    resolved = Path(path).resolve()

    if base is not None:
        resolved_base = Path(base).resolve()
        try:
            resolved.relative_to(resolved_base)
        except ValueError as exc:
            raise ValueError(
                f"Path '{resolved}' escapes base directory '{resolved_base}'."
            ) from exc

    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved}")

    return resolved


def format_file_size(num_bytes: int | float) -> str:
    """Format a byte count as a human-readable string.

    Args:
        num_bytes: Number of bytes. Negative values are formatted with a sign.

    Returns:
        A string such as ``"1.50 KB"`` using binary units (KiB/MiB are mapped to
        the common KB/MB labels for readability).

    Raises:
        TypeError: If ``num_bytes`` is not numeric.

    Example:
        >>> format_file_size(1536)
        '1.50 KB'
        >>> format_file_size(0)
        '0 B'
    """
    if not isinstance(num_bytes, (int, float)):
        raise TypeError(f"num_bytes must be numeric, got {type(num_bytes).__name__}")

    sign = "-" if num_bytes < 0 else ""
    value = abs(float(num_bytes))

    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{sign}{int(value)} {unit}"
            return f"{sign}{value:.2f} {unit}"
        value /= 1024.0

    return f"{sign}{value:.2f} TB"  # pragma: no cover - unreachable guard


def format_duration(seconds: int | float) -> str:
    """Format a duration in seconds as a compact human-readable string.

    Args:
        seconds: Duration in seconds.

    Returns:
        A string such as ``"1h 2m 3s"`` or ``"12.34s"`` for sub-minute values.

    Raises:
        ValueError: If ``seconds`` is negative.

    Example:
        >>> format_duration(3723)
        '1h 2m 3s'
        >>> format_duration(0.5)
        '0.50s'
    """
    if seconds < 0:
        raise ValueError("Duration cannot be negative.")

    if seconds < 60:
        return f"{seconds:.2f}s"

    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)

    parts: list[str] = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def parse_bool(value: str | bool | int | None) -> bool:
    """Parse an environment-style value into a boolean.

    Args:
        value: Raw value. Booleans and integers pass through with sensible
            semantics; strings are matched case-insensitively.

    Returns:
        The parsed boolean.

    Raises:
        ValueError: If a string value cannot be interpreted as boolean.

    Example:
        >>> parse_bool("yes")
        True
        >>> parse_bool("0")
        False
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"Cannot interpret '{value}' as a boolean.")


def set_seed(seed: int) -> dict[str, bool]:
    """Seed every available random engine for reproducibility.

    Each engine is configured independently. Missing optional libraries are
    skipped gracefully and reported in the returned status dictionary rather
    than raising, so the caller always knows which engines were seeded.

    Args:
        seed: Integer seed value applied to all available engines.

    Returns:
        A mapping ``{"random": bool, "numpy": bool, "torch": bool}`` indicating
        which engines were successfully seeded.

    Example:
        >>> set_seed(42)["random"]
        True
    """
    status = {"random": False, "numpy": False, "torch": False}

    random.seed(seed)
    status["random"] = True

    try:
        import numpy as np

        np.random.seed(seed)
        status["numpy"] = True
    except ImportError:
        status["numpy"] = False

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        status["torch"] = True
    except ImportError:
        status["torch"] = False

    return status


@lru_cache(maxsize=1)
def get_device() -> str:
    """Detect the best available compute device.

    Detection order is CUDA, then MPS (Apple Silicon), then CPU. The result is
    cached because device availability does not change during a process.

    Note:
        A CPU fallback is returned *by design* when PyTorch is unavailable or
        no accelerator is present; this is intentional graceful behaviour, not a
        swallowed error.

    Returns:
        One of ``"cuda"``, ``"mps"``, or ``"cpu"``.

    Example:
        >>> get_device() in {"cuda", "mps", "cpu"}
        True
    """
    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"

    try:
        mps_available = bool(torch.backends.mps.is_available())
    except (RuntimeError, AttributeError):
        mps_available = False

    if mps_available:
        return "mps"

    return "cpu"


def validate_required_env(
    required: Iterable[str],
    *,
    source: str = "environment",
) -> list[str]:
    """Return the names of required environment variables that are missing.

    A variable is considered missing when it is unset or contains only
    whitespace. This helper is pure: it never raises and never mutates state,
    leaving the decision of how to react to the caller.

    Args:
        required: Iterable of environment variable names to check.
        source: Human-readable label used in downstream error messages.

    Returns:
        A (possibly empty) list of missing variable names, in input order.

    Example:
        >>> validate_required_env(["DEFINITELY_NOT_SET_XYZ"])
        ['DEFINITELY_NOT_SET_XYZ']
    """
    missing: list[str] = []
    for name in required:
        if not os.getenv(name, "").strip():
            missing.append(name)
    _ = source  # reserved for richer error formatting by callers
    return missing


def project_banner(title: str, *, width: int = 60, char: str = "=") -> str:
    """Build a simple ASCII banner for CLI or Streamlit startup output.

    Args:
        title: Text shown in the centre of the banner.
        width: Total banner width in characters.
        char: Border character used for the top and bottom rules.

    Returns:
        A multi-line banner string.

    Raises:
        ValueError: If ``width`` is too small to contain the title or ``char``
            is not a single character.

    Example:
        >>> "Full RAG" in project_banner("Full RAG")
        True
    """
    if len(char) != 1:
        raise ValueError("char must be exactly one character.")
    if width < len(title) + 4:
        raise ValueError("width is too small for the given title.")

    rule = char * width
    return f"{rule}\n  {title}\n{rule}"


def normalize_whitespace(text: str) -> str:
    """Strip edges and collapse repeated spaces (basic, lossless cleaning).

    This intentionally does **not** lowercase, remove punctuation, tokenize or
    chunk the text; it only normalises whitespace so it can be reused by the
    data layer and later preprocessing stages. Internal newlines are preserved.

    Args:
        text: Raw text to clean.

    Returns:
        The cleaned text with single internal spaces and trimmed edges.

    Raises:
        TypeError: If ``text`` is not a string.

    Example:
        >>> normalize_whitespace("  hello   world  ")
        'hello world'
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    return re.sub(r" {2,}", " ", text).strip()


def safe_filename(name: str, *, replacement: str = "_") -> str:
    """Sanitise a string so it is safe to use as a file name.

    Characters that are illegal or problematic on common filesystems are
    replaced, and runs of the replacement character are collapsed.

    Args:
        name: Raw name to sanitise.
        replacement: Character used in place of illegal ones.

    Returns:
        A filesystem-safe, non-empty name.

    Raises:
        ValueError: If ``name`` is empty or not a string.

    Example:
        >>> safe_filename("rajpurkar/squad_v2")
        'rajpurkar_squad_v2'
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string")
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', replacement, name)
    cleaned = re.sub(rf"{re.escape(replacement)}+", replacement, cleaned)
    cleaned = cleaned.strip(replacement)
    return cleaned or "unnamed"
