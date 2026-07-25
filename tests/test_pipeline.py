"""Pipeline tests.

Title:
    Pipeline Test Module

Description:
    Contains baseline tests for the RAG pipeline package.

Responsibilities:
    - Verify pipeline module imports.
    - Provide a stable place for future orchestration tests.

Author:
    Author Placeholder
"""

import pytest
from src.rag_pipeline import RAGPipeline


def test_pipeline_placeholder_raises_not_implemented() -> None:
    """Verify Phase 1 pipeline placeholder is explicit.

    Args:
        None.

    Returns:
        None.
    """
    pipeline = RAGPipeline()

    with pytest.raises(NotImplementedError):
        pipeline.answer_question("What is RAG?")
