"""Prompt engineering templates (Phase 8).

Title:
    Prompt Module
Description:
    Holds the grounded-RAG prompt templates and the helpers that assemble the
    chat messages sent to the LLM. Prompting is kept strictly separate from
    model invocation (which lives in ``src.rag_pipeline``) so the templates can
    be reviewed, tested and tuned in isolation.

    The system prompt enforces strict grounding: the model must answer *only*
    from the retrieved context and must refuse with a fixed phrase when the
    context is insufficient. That fixed phrase (``RAG_REFUSAL_PHRASE``) is
    shared with the evaluation layer so detection and instruction never drift
    apart.
Responsibilities:
    - Define the grounded system prompt and the refusal phrase.
    - Build the per-query user message (context + question).
    - Format retrieved chunks into a single context string.
    - Keep a backward-compatible ``build_rag_prompt`` single-string helper.
Author:
    Aml
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

# The exact sentence the model must emit when the context cannot answer the
# question. Kept as a constant so the evaluation layer (Phase 9) can detect a
# refusal without re-implementing the wording.
RAG_REFUSAL_PHRASE: str = "I could not find the answer in the retrieved documents."

RAG_SYSTEM_PROMPT: str = (
    "You are a precise, factual question-answering assistant.\n"
    "You must answer the user's question using ONLY the information contained "
    "in the 'Context' section provided with the question.\n"
    "Rules:\n"
    "1. Only use facts explicitly present in the retrieved context.\n"
    "2. If the context does not contain the answer, respond exactly with: "
    f'"{RAG_REFUSAL_PHRASE}"\n'
    "3. Never invent, assume, or hallucinate information not present in the "
    "context.\n"
    "4. Be concise — answer in one short sentence or phrase whenever possible."
)

# Separator placed between consecutive retrieved chunks inside the context.
CONTEXT_SEPARATOR: str = "\n\n---\n\n"


def _chunk_text(chunk: Any) -> str:
    """Extract the text of a chunk-like object.

    Accepts objects with a ``text`` attribute (e.g. ``RetrievedChunk``),
    mappings with a ``"text"`` key, or plain strings.

    Args:
        chunk: A chunk, mapping, or string.

    Returns:
        The chunk text.
    """
    if isinstance(chunk, str):
        return chunk
    text = getattr(chunk, "text", None)
    if text is None and isinstance(chunk, dict):
        text = chunk.get("text", "")
    return str(text or "")


def format_context(chunks: Iterable[Any]) -> str:
    """Join retrieved chunks into a single context string for the prompt.

    Empty chunks are skipped so the context never contains blank separators.

    Args:
        chunks: Iterable of chunk-like objects (``RetrievedChunk``, mapping or
            string). Typically a ``RetrievalResult`` from the retriever, which
            iterates over its ranked chunks.

    Returns:
        The concatenated context, or an empty string when there are no chunks.
    """
    parts = [text for text in (_chunk_text(c) for c in chunks) if text.strip()]
    return CONTEXT_SEPARATOR.join(parts)


def build_user_message(question: str, context: str) -> str:
    """Build the user-role message carrying the context and the question.

    Args:
        question: The user's natural-language question.
        context: The retrieved context string (see :func:`format_context`).

    Returns:
        The formatted user message.

    Raises:
        ValueError: If ``question`` is empty.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must not be empty.")
    return f"Context:\n{context}\n\nQuestion:\n{question}"


def build_rag_prompt(question: str, context: str) -> str:
    """Build a single grounded prompt string (backward-compatible helper).

    This concatenates the system prompt and the user message into one string.
    New code that talks to a chat API should prefer the role-separated
    ``RAG_SYSTEM_PROMPT`` + :func:`build_user_message` pair instead.

    Args:
        question: User question.
        context: Retrieved context text.

    Returns:
        Formatted prompt string.

    Raises:
        ValueError: If ``question`` or ``context`` is empty.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must not be empty.")
    if not isinstance(context, str) or not context.strip():
        raise ValueError("context must not be empty.")
    return f"{RAG_SYSTEM_PROMPT}\n\n{build_user_message(question, context)}"
