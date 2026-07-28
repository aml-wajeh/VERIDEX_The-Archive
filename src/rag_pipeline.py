"""RAG pipeline orchestration (Phase 8).

Title:
    RAG Pipeline Module
Description:
    Coordinates the pieces built in the previous phases into a single
    question-answering call: the retriever (Phase 7) supplies ranked, scored
    evidence; the prompt module (Phase 8) turns it into grounded chat messages;
    a thin Groq client generates the answer; and everything is assembled into a
    result dictionary whose shape is a *superset* of the legacy
    ``rag_pipeline.ask`` output, so the Streamlit UI can consume it unchanged.

    The package stays framework-agnostic: generation goes through the official
    ``groq`` SDK (imported lazily), never LangChain. Every dependency is
    injected, so the orchestration is fully unit-testable with fakes and never
    touches the network or an API key in tests.
Responsibilities:
    - Validate the question before any work is done.
    - Retrieve evidence via an injected ``Retriever``.
    - Build grounded chat messages and call an injected LLM client.
    - Time retrieval and generation separately.
    - Estimate retrieval confidence from the similarity scores.
    - Return a stable result dict; wrap runtime failures instead of crashing.
Author:
    Aml
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.config import LLMConfig, resolve_llm_config
from src.prompt import (
    RAG_SYSTEM_PROMPT,
    build_user_message,
    format_context,
)
from src.retriever import Retriever, RetrieverError

try:
    from src.logger import get_logger
except ImportError:  # pragma: no cover
    from logging import getLogger as get_logger


class PipelineError(Exception):
    """Base exception for pipeline orchestration errors."""


class PipelineValidationError(PipelineError):
    """Raised when the question or configuration is invalid."""


class PipelineConnectionError(PipelineError):
    """Raised when the LLM client cannot be built (missing key / SDK)."""


class PipelineGenerationError(PipelineError):
    """Raised when the LLM call itself fails."""


# Similarity thresholds for the retrieval-confidence buckets.
_HIGH_SIMILARITY: float = 0.75
_MEDIUM_SIMILARITY: float = 0.55

# Keys that every result dict carries (the legacy contract, plus confidence).
_RESULT_KEYS: tuple[str, ...] = (
    "question",
    "answer",
    "retrieved_pages",
    "metadata",
    "similarities",
    "retrieval_time_sec",
    "generation_time_sec",
    "total_time_sec",
    "error",
    "confidence",
)


@dataclass(frozen=True)
class ConfidenceEstimate:
    """A retrieval-confidence bucket derived from similarity scores.

    Attributes:
        label: One of ``"High"``, ``"Medium"`` or ``"Low"``.
        avg_similarity: Mean similarity across the retrieved chunks (0.0 when
            nothing was retrieved).
        text: Human-readable description of the estimate.
    """

    label: str
    avg_similarity: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        """Convert the estimate to a plain dictionary.

        Returns:
            Dictionary with ``label``, ``avg_similarity`` and ``text``.
        """
        return {
            "label": self.label,
            "avg_similarity": self.avg_similarity,
            "text": self.text,
        }


def estimate_retrieval_confidence(
    similarities: Sequence[float] | None,
) -> dict[str, Any]:
    """Bucket retrieval quality into High / Medium / Low.

    This is a *retrieval* confidence signal (how well the query matched the
    corpus), not a measure of whether the generated answer is factually
    correct. The returned mapping mirrors the legacy helper so the UI can read
    ``result["confidence"]["label"]`` and ``["avg_similarity"]`` directly.

    Args:
        similarities: Similarity scores of the retrieved chunks.

    Returns:
        A mapping with ``label``, ``avg_similarity`` and ``text``.
    """
    scores = [float(s) for s in (similarities or []) if s is not None]
    if not scores:
        return ConfidenceEstimate(
            label="Low",
            avg_similarity=0.0,
            text="Low (no chunks retrieved)",
        ).to_dict()

    avg_sim = sum(scores) / len(scores)
    if avg_sim >= _HIGH_SIMILARITY:
        label = "High"
    elif avg_sim >= _MEDIUM_SIMILARITY:
        label = "Medium"
    else:
        label = "Low"

    return ConfidenceEstimate(
        label=label,
        avg_similarity=avg_sim,
        text=f"{label} (avg similarity = {avg_sim:.3f})",
    ).to_dict()


class GroqLLMClient:
    """Thin wrapper around the official ``groq`` SDK chat completions API.

    The API key is resolved eagerly (from an explicit argument or the
    environment) and validated *before* the SDK is imported, so a missing key
    surfaces as a clear :class:`PipelineConnectionError` without requiring the
    ``groq`` package or any network access.

    Attributes:
        config: Resolved LLM configuration.
    """

    def __init__(
        self,
        config: LLMConfig | Any | None = None,
        api_key: str | None = None,
    ) -> None:
        """Resolve and validate the API key, then build the Groq client lazily.

        Args:
            config: Optional LLM configuration.
            api_key: Optional explicit API key; otherwise read from the
                environment variable named by ``config.api_key_env``.

        Raises:
            PipelineConnectionError: If the key is empty or the ``groq`` SDK is
                unavailable.
        """
        try:
            self._config = resolve_llm_config(config)
        except (TypeError, ValueError, AttributeError) as exc:
            raise PipelineConnectionError("Invalid LLM configuration.") from exc

        resolved_key = (
            api_key
            if api_key is not None
            else os.environ.get(self._config.api_key_env, "")
        )
        resolved_key = (resolved_key or "").strip()
        if not resolved_key:
            raise PipelineConnectionError(
                f"{self._config.api_key_env} is empty. Get a free Groq key at "
                "https://console.groq.com/keys and export it, or pass api_key."
            )

        try:
            from groq import Groq
        except ImportError as exc:
            raise PipelineConnectionError(
                "The 'groq' package is required for generation. "
                "Install the project requirements first."
            ) from exc

        client_kwargs: dict[str, Any] = {"api_key": resolved_key}
        if self._config.request_timeout is not None:
            client_kwargs["timeout"] = self._config.request_timeout

        try:
            self._client = Groq(**client_kwargs)
        except Exception as exc:  # noqa: BLE001
            raise PipelineConnectionError(
                "Failed to initialise the Groq client."
            ) from exc

    @property
    def config(self) -> LLMConfig:
        """Return the active LLM configuration.

        Returns:
            Active `LLMConfig`.
        """
        return self._config

    def chat(self, messages: Sequence[Mapping[str, str]], **_: Any) -> str:
        """Send chat messages to Groq and return the assistant text.

        Args:
            messages: Chat messages, each a mapping with ``role`` and
                ``content``.
            **_: Ignored extra keyword arguments (keeps the call signature
                compatible with alternative clients / fakes).

        Returns:
            The assistant's response text (empty string if the model returned
            no content).

        Raises:
            PipelineGenerationError: If the API call fails.
        """
        create_kwargs: dict[str, Any] = {
            "model": self._config.model_name,
            "messages": [dict(m) for m in messages],
            "temperature": self._config.temperature,
        }
        if self._config.max_tokens is not None:
            create_kwargs["max_tokens"] = self._config.max_tokens

        try:
            completion = self._client.chat.completions.create(**create_kwargs)
        except Exception as exc:  # noqa: BLE001
            raise PipelineGenerationError("Groq chat completion failed.") from exc

        try:
            content = completion.choices[0].message.content
        except (AttributeError, IndexError) as exc:
            raise PipelineGenerationError(
                "Groq returned an unexpected response shape."
            ) from exc

        return content or ""


def build_llm_client(
    config: LLMConfig | Any | None = None,
    api_key: str | None = None,
) -> GroqLLMClient:
    """Build the default Groq-backed LLM client.

    Args:
        config: Optional LLM configuration.
        api_key: Optional explicit API key.

    Returns:
        A ready :class:`GroqLLMClient`.

    Raises:
        PipelineConnectionError: If the key is missing or the SDK is absent.
    """
    return GroqLLMClient(config=config, api_key=api_key)


class RAGPipeline:
    """Orchestrates retrieval + grounded generation for one question.

    The retriever is mandatory (the pipeline has nothing to search otherwise).
    The LLM client is optional and built lazily on first use, so constructing a
    pipeline never touches the network; tests inject a fake client instead.

    Attributes:
        config: Resolved LLM configuration.
    """

    def __init__(
        self,
        retriever: Retriever | None = None,
        llm_client: Any | None = None,
        config: LLMConfig | Any | None = None,
    ) -> None:
        """Initialise the pipeline.

        Args:
            retriever: The retriever to query for evidence (required).
            llm_client: Optional chat client; any object exposing
                ``chat(messages, **kw) -> str``. Built lazily when omitted.
            config: Optional LLM configuration.

        Raises:
            PipelineValidationError: If no retriever is supplied or the
                configuration is invalid.
        """
        if retriever is None:
            raise PipelineValidationError(
                "A Retriever instance is required to build a RAGPipeline."
            )

        try:
            self._config = resolve_llm_config(config)
        except (TypeError, ValueError, AttributeError) as exc:
            raise PipelineValidationError(
                "Invalid pipeline / LLM configuration."
            ) from exc

        self._retriever = retriever
        self._llm_client = llm_client
        self._logger = get_logger("rag_pipeline")

    @property
    def config(self) -> LLMConfig:
        """Return the active LLM configuration.

        Returns:
            Active `LLMConfig`.
        """
        return self._config

    def _get_llm_client(self) -> Any:
        """Return the chat client, building the default one lazily.

        Returns:
            The active chat client.

        Raises:
            PipelineConnectionError: If the default client cannot be built.
        """
        if self._llm_client is None:
            self._llm_client = build_llm_client(self._config)
            self._logger.info("Built the default Groq LLM client.")
        return self._llm_client

    def answer_question(
        self,
        question: str,
        top_k: int | None = None,
        llm_client: Any | None = None,
    ) -> dict[str, Any]:
        """Run retrieval + grounded generation for a single question.

        Input validation (empty / non-string question) raises
        :class:`PipelineValidationError` so callers can distinguish bad input
        from a runtime failure. Any retrieval or generation failure is caught
        and returned as an ``error`` result dict instead of propagating, so a
        UI built on top of this never crashes on a single bad query.

        Args:
            question: The user's natural-language question.
            top_k: Optional override for the number of chunks to retrieve.

        Returns:
            A result mapping carrying the answer, the retrieved evidence, the
            similarity scores, per-stage timings, a confidence estimate and an
            ``error`` field (``None`` on success).

        Raises:
            PipelineValidationError: If ``question`` is not a non-empty string.
        """
        if not isinstance(question, str) or not question.strip():
            raise PipelineValidationError("question must be a non-empty string.")

        try:
            retrieval = self._retriever.retrieve(question, top_k=top_k)
            chunks = list(retrieval)
            context = format_context(chunks)
            retrieved_pages = [chunk.text for chunk in chunks]
            metadatas = [dict(chunk.metadata) for chunk in chunks]
            similarities = [float(chunk.similarity) for chunk in chunks]
            retrieval_time = float(retrieval.retrieval_time_sec)

            messages = [
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_user_message(question, context),
                },
            ]

            start = time.perf_counter()
            client = llm_client if llm_client is not None else self._get_llm_client()
            answer = client.chat(messages)
            generation_time = time.perf_counter() - start

        except PipelineValidationError:
            raise
        except RetrieverError as exc:
            self._logger.exception("Retrieval failed for question: %s", question)
            return self._error_result(question, str(exc))
        except (PipelineConnectionError, PipelineGenerationError) as exc:
            self._logger.exception("Generation failed for question: %s", question)
            return self._error_result(question, str(exc))
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("answer_question failed for question: %s", question)
            return self._error_result(question, str(exc))

        return {
            "question": question,
            "answer": answer.strip(),
            "retrieved_pages": retrieved_pages,
            "metadata": metadatas,
            "similarities": similarities,
            "retrieval_time_sec": round(retrieval_time, 4),
            "generation_time_sec": round(generation_time, 4),
            "total_time_sec": round(retrieval_time + generation_time, 4),
            "error": None,
            "confidence": estimate_retrieval_confidence(similarities),
        }

    @staticmethod
    def _error_result(question: str, error: str) -> dict[str, Any]:
        """Build a zeroed result dict describing a runtime failure.

        Args:
            question: The question that failed.
            error: Human-readable error description.

        Returns:
            A result mapping with empty evidence and the ``error`` set.
        """
        return {
            "question": question,
            "answer": "",
            "retrieved_pages": [],
            "metadata": [],
            "similarities": [],
            "retrieval_time_sec": 0.0,
            "generation_time_sec": 0.0,
            "total_time_sec": 0.0,
            "error": error,
            "confidence": estimate_retrieval_confidence([]),
        }


# Public surface of a result dict (handy for consumers / tests).
RESULT_KEYS: tuple[str, ...] = _RESULT_KEYS
