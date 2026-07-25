"""Prompt engineering templates.

Title:
    Prompt Module

Description:
    Contains prompt templates and prompt-building helpers for grounded RAG
    responses.

Responsibilities:
    - Define prompt templates.
    - Format retrieved context.
    - Keep prompting separate from model invocation.

Author:
    Author Placeholder
"""


RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer using only the retrieved context. "
    "If the answer is not in the context, say that the context is insufficient."
)


def build_rag_prompt(question: str, context: str) -> str:
    """Build a grounded RAG prompt.

    Args:
        question: User question.
        context: Retrieved context text.

    Returns:
        Formatted prompt string.

    Raises:
        ValueError: If question or context is empty.
    """
    if not question.strip():
        raise ValueError("question must not be empty.")
    if not context.strip():
        raise ValueError("context must not be empty.")

    return f"{RAG_SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion:\n{question}"
