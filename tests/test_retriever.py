"""Retriever tests.

Title:
    Retriever Test Module

Description:
    Contains baseline tests for retriever behavior.

Responsibilities:
    - Verify retriever module imports.
    - Provide a stable place for future similarity search tests.

Author:
    Author Placeholder
"""

import pytest

from src.retriever import Retriever


def test_retriever_placeholder_raises_not_implemented() -> None:
    """Verify Phase 1 retriever placeholder is explicit.

    Args:
        None.

    Returns:
        None.
    """
    retriever = Retriever()

    with pytest.raises(NotImplementedError):
        retriever.retrieve("What is RAG?")
