"""Tests for the RAG pipeline orchestration (Phase 8)."""

from __future__ import annotations

from typing import Any

import pytest
from src.config import LLMConfig
from src.prompt import RAG_SYSTEM_PROMPT
from src.rag_pipeline import (
    GroqLLMClient,
    PipelineConnectionError,
    PipelineValidationError,
    RAGPipeline,
    estimate_retrieval_confidence,
)
from src.retriever import RetrievalResult, RetrievedChunk


def _chunk(
    text: str,
    similarity: float,
    *,
    chunk_id: str = "c1",
    title: str = "Title",
) -> RetrievedChunk:
    """Build a retrieved chunk for tests.

    Args:
        text: Chunk text.
        similarity: Similarity score.
        chunk_id: Chunk identifier.
        title: Title stored in metadata.

    Returns:
        A `RetrievedChunk`.
    """
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="d1",
        text=text,
        metadata={"title": title},
        similarity=similarity,
        distance=round(1.0 - similarity, 4),
        rank=1,
    )


def _result(
    chunks: list[RetrievedChunk], retrieval_time: float = 0.01
) -> RetrievalResult:
    """Wrap chunks into a retrieval result.

    Args:
        chunks: Ranked chunks.
        retrieval_time: Fake retrieval time in seconds.

    Returns:
        A `RetrievalResult`.
    """
    return RetrievalResult(
        query="q",
        chunks=tuple(chunks),
        top_k=len(chunks),
        retrieval_time_sec=retrieval_time,
    )


class FakeRetriever:
    """Stand-in for `Retriever` returning a pre-built result or raising."""

    def __init__(
        self,
        result: RetrievalResult | None = None,
        raise_error: bool = False,
    ) -> None:
        """Initialise the fake retriever.

        Args:
            result: Result returned on every retrieve call.
            raise_error: Whether to raise instead of returning.
        """
        self._result = result or _result([_chunk("evidence text", 0.8)])
        self._raise = raise_error
        self.calls = 0
        self.last_top_k: int | None = None

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        where: object = None,
        min_similarity: float | None = None,
    ) -> RetrievalResult:
        """Return the pre-built result (or raise).

        Args:
            query: Ignored.
            top_k: Recorded for assertion.
            where: Ignored.
            min_similarity: Ignored.

        Returns:
            The pre-built result.

        Raises:
            RuntimeError: When ``raise_error`` is set.
        """
        self.calls += 1
        self.last_top_k = top_k
        if self._raise:
            raise RuntimeError("retrieval boom")
        return self._result


class FakeLLM:
    """Stand-in for a chat client exposing ``chat(messages) -> str``."""

    def __init__(
        self, response: str = "The answer.", raise_error: bool = False
    ) -> None:
        """Initialise the fake client.

        Args:
            response: Text returned by ``chat``.
            raise_error: Whether to raise instead of returning.
        """
        self._response = response
        self._raise = raise_error
        self.calls = 0
        self.last_messages: list[dict[str, Any]] = []

    def chat(self, messages: list[dict[str, Any]], **_: Any) -> str:
        """Record the messages and return the canned response.

        Args:
            messages: Chat messages.
            **_: Ignored.

        Returns:
            The canned response.

        Raises:
            RuntimeError: When ``raise_error`` is set.
        """
        self.calls += 1
        self.last_messages = [dict(m) for m in messages]
        if self._raise:
            raise RuntimeError("generation boom")
        return self._response


def _pipeline(
    retriever: FakeRetriever | None = None,
    llm: FakeLLM | None = None,
) -> tuple[RAGPipeline, FakeRetriever, FakeLLM]:
    """Build a pipeline wired to fakes.

    Args:
        retriever: Optional fake retriever.
        llm: Optional fake LLM client.

    Returns:
        A tuple of (pipeline, retriever, llm).
    """
    fake_retriever = retriever or FakeRetriever()
    fake_llm = llm or FakeLLM()
    pipeline = RAGPipeline(
        retriever=fake_retriever,  # type: ignore[arg-type]
        llm_client=fake_llm,
    )
    return pipeline, fake_retriever, fake_llm


def test_pipeline_requires_retriever() -> None:
    """Building a pipeline without a retriever is rejected eagerly."""
    with pytest.raises(PipelineValidationError):
        RAGPipeline()


def test_empty_question_raises_before_any_call() -> None:
    """An empty question raises and never reaches the retriever or LLM."""
    pipeline, retriever, llm = _pipeline()

    with pytest.raises(PipelineValidationError):
        pipeline.answer_question("   ")

    assert retriever.calls == 0
    assert llm.calls == 0


def test_happy_path_returns_full_result_dict() -> None:
    """A successful run returns every legacy key plus confidence."""
    chunks = [
        _chunk("Beyoncé is a singer.", 0.9, chunk_id="c1"),
        _chunk("She rose to fame.", 0.7, chunk_id="c2"),
    ]
    pipeline, retriever, llm = _pipeline(
        retriever=FakeRetriever(result=_result(chunks, retrieval_time=0.05)),
        llm=FakeLLM(response="  Beyoncé is a singer.  "),
    )

    result = pipeline.answer_question("Who is Beyoncé?")

    assert result["error"] is None
    assert result["answer"] == "Beyoncé is a singer."
    assert result["question"] == "Who is Beyoncé?"
    assert result["retrieved_pages"] == [c.text for c in chunks]
    assert result["metadata"] == [dict(c.metadata) for c in chunks]
    assert result["similarities"] == [0.9, 0.7]
    assert result["retrieval_time_sec"] == pytest.approx(0.05)
    assert result["generation_time_sec"] >= 0.0
    assert result["total_time_sec"] >= result["retrieval_time_sec"]
    assert result["confidence"]["label"] in {"High", "Medium", "Low"}
    assert retriever.calls == 1
    assert llm.calls == 1


def test_prompt_wiring_uses_system_and_user_roles() -> None:
    """The LLM receives a grounded system message and a context user message."""
    pipeline, _, llm = _pipeline(
        retriever=FakeRetriever(
            result=_result([_chunk("Solar energy is power from the sun.", 0.8)])
        ),
    )

    pipeline.answer_question("What is solar energy?")

    roles = [m["role"] for m in llm.last_messages]
    assert roles == ["system", "user"]
    assert llm.last_messages[0]["content"] == RAG_SYSTEM_PROMPT
    user_content = llm.last_messages[1]["content"]
    assert "Solar energy is power from the sun." in user_content
    assert "What is solar energy?" in user_content


def test_top_k_override_is_forwarded_to_retriever() -> None:
    """A per-call top_k reaches the retriever."""
    pipeline, retriever, _ = _pipeline()

    pipeline.answer_question("q", top_k=7)

    assert retriever.last_top_k == 7


def test_llm_failure_is_wrapped_as_error_result() -> None:
    """A generation failure yields an error dict instead of raising."""
    pipeline, _, _ = _pipeline(llm=FakeLLM(raise_error=True))

    result = pipeline.answer_question("q")

    assert result["error"]
    assert result["answer"] == ""
    assert result["retrieved_pages"] == []
    assert result["similarities"] == []
    assert result["total_time_sec"] == 0.0
    assert result["confidence"]["label"] == "Low"


def test_retriever_failure_is_wrapped_as_error_result() -> None:
    """A retrieval failure yields an error dict instead of raising."""
    pipeline, _, llm = _pipeline(retriever=FakeRetriever(raise_error=True))

    result = pipeline.answer_question("q")

    assert result["error"]
    assert result["answer"] == ""
    assert llm.calls == 0


def test_estimate_retrieval_confidence_buckets() -> None:
    """Confidence buckets map average similarity to High/Medium/Low."""
    assert estimate_retrieval_confidence([0.9, 0.8])["label"] == "High"
    assert estimate_retrieval_confidence([0.6, 0.7])["label"] == "Medium"
    assert estimate_retrieval_confidence([0.2, 0.3])["label"] == "Low"
    empty = estimate_retrieval_confidence([])
    assert empty["label"] == "Low"
    assert empty["avg_similarity"] == 0.0
    assert estimate_retrieval_confidence(None)["label"] == "Low"


def test_groq_client_missing_key_raises_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty key raises a connection error before any SDK / network use."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(PipelineConnectionError):
        GroqLLMClient(config=LLMConfig(), api_key="   ")


def test_invalid_llm_config_raises_validation_error() -> None:
    """An invalid configuration is rejected at construction time."""
    with pytest.raises(PipelineValidationError):
        RAGPipeline(
            retriever=FakeRetriever(),  # type: ignore[arg-type]
            config={"temperature": 5.0},
        )
