"""Tests for the evaluation module (Phase 9)."""

from __future__ import annotations

from typing import Any

import pytest
from src.config import EvaluationConfig
from src.evaluation import (
    EvaluationEngine,
    EvaluationValidationError,
    compute_exact_match,
    compute_f1,
    compute_mrr,
    compute_retrieval_hit,
    is_refusal,
    normalize_answer,
    tokenize_answer,
)
from src.prompt import RAG_REFUSAL_PHRASE


# ---------------------------------------------------------------------------
# Fake pipeline (no Groq / network / model)
# ---------------------------------------------------------------------------
def _ok(
    answer: str = "",
    pages: list[str] | None = None,
    sims: list[float] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Build a pipeline-style result dict.

    Args:
        answer: Model answer.
        pages: Retrieved chunk texts.
        sims: Similarity scores.
        error: Optional error string.

    Returns:
        A result mapping.
    """
    return {
        "answer": answer,
        "retrieved_pages": list(pages or []),
        "similarities": list(sims or []),
        "retrieval_time_sec": 0.1,
        "generation_time_sec": 0.2,
        "total_time_sec": 0.3,
        "error": error,
    }


class FakePipeline:
    """Deterministic stand-in for the RAG pipeline."""

    def __init__(
        self,
        responses: dict[str, dict[str, Any]] | None = None,
        raise_for: set[str] | None = None,
    ) -> None:
        """Initialise the fake pipeline.

        Args:
            responses: Mapping of question to result dict.
            raise_for: Questions that make ``answer_question`` raise.
        """
        self._responses = dict(responses or {})
        self._raise_for = set(raise_for or ())
        self.calls = 0
        self.last_top_k: int | None = None
        self.questions: list[str] = []

    def answer_question(
        self,
        question: str,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Return a canned result (or raise) for a question.

        Args:
            question: Question text.
            top_k: Recorded for assertion.

        Returns:
            A result mapping.

        Raises:
            RuntimeError: When the question is in ``raise_for``.
        """
        self.calls += 1
        self.last_top_k = top_k
        self.questions.append(question)
        if question in self._raise_for:
            raise RuntimeError("pipeline boom")
        return dict(self._responses.get(question, _ok()))


# ---------------------------------------------------------------------------
# Pure metric functions
# ---------------------------------------------------------------------------
def test_normalize_answer_lower_articles_and_punct() -> None:
    """Normalisation lower-cases and strips articles and punctuation."""
    assert normalize_answer("The Cat, sat.") == "cat sat"
    assert normalize_answer("An apple!") == "apple"
    assert normalize_answer(None) == ""  # type: ignore[arg-type]


def test_tokenize_answer() -> None:
    """Tokenisation splits the normalised answer on whitespace."""
    assert tokenize_answer("The quick brown fox") == [
        "quick",
        "brown",
        "fox",
    ]


def test_compute_exact_match_basic_and_empty_gold() -> None:
    """EM matches any gold; empty gold needs an empty prediction."""
    assert compute_exact_match("Paris", ["Paris", "paris."]) is True
    assert compute_exact_match("London", ["Paris"]) is False
    assert compute_exact_match("", []) is True
    assert compute_exact_match("something", []) is False


def test_compute_exact_match_multiple_golds() -> None:
    """EM is True when the prediction matches any one gold answer."""
    assert compute_exact_match("Beyoncé", ["Beyonce", "Beyoncé"]) is True


def test_compute_f1_partial_full_and_empty() -> None:
    """F1 reflects partial overlap, full match, and the empty-gold rule."""
    assert compute_f1("the capital of France", ["Paris is the capital"]) > 0.0
    assert compute_f1("Paris", ["Paris"]) == pytest.approx(1.0)
    assert compute_f1("wrong", ["Paris"]) == 0.0
    assert compute_f1("", []) == pytest.approx(1.0)
    assert compute_f1("hallucination", []) == 0.0


def test_compute_retrieval_hit_present_absent_empty() -> None:
    """A hit is found when a gold span is inside a retrieved chunk."""
    assert compute_retrieval_hit(["Paris is the capital of France."], ["Paris"]) is True
    assert compute_retrieval_hit(["Cats are furry animals."], ["Paris"]) is False
    assert compute_retrieval_hit(["anything"], []) is False
    assert compute_retrieval_hit([], ["Paris"]) is False


def test_compute_mrr_rank_and_absent() -> None:
    """MRR is 1/rank of the first matching chunk, else 0."""
    assert compute_mrr(
        ["noise", "Paris is the capital of France."], ["Paris"]
    ) == pytest.approx(0.5)
    assert compute_mrr(["Paris is the capital."], ["Paris"]) == pytest.approx(1.0)
    assert compute_mrr(["noise"], ["Paris"]) == 0.0


def test_is_refusal_detection() -> None:
    """Refusal detection is normalised and ignores empty answers."""
    assert is_refusal(RAG_REFUSAL_PHRASE) is True
    assert is_refusal(f"Sorry, {RAG_REFUSAL_PHRASE}") is True
    assert is_refusal("Paris is the capital of France.") is False
    assert is_refusal("") is False


# ---------------------------------------------------------------------------
# Engine behaviour
# ---------------------------------------------------------------------------
def test_engine_requires_pipeline() -> None:
    """Building an engine without a pipeline is rejected eagerly."""
    with pytest.raises(EvaluationValidationError):
        EvaluationEngine(pipeline=None)


def test_engine_invalid_config_raises() -> None:
    """An invalid configuration is rejected at construction time."""
    with pytest.raises(EvaluationValidationError):
        EvaluationEngine(FakePipeline(), config={"top_k": 0})


def test_engine_happy_answerable() -> None:
    """A correct answer on an answerable question scores EM=1, F1=1."""
    pipeline = FakePipeline(
        {"Q?": _ok("Paris", ["Paris is the capital of France."], [0.9])}
    )
    engine = EvaluationEngine(pipeline)

    report = engine.evaluate([{"question": "Q?", "gold_answers": ["Paris"]}])

    row = report.per_question[0]
    assert row.exact_match is True
    assert row.f1 == pytest.approx(1.0)
    assert row.is_refusal is False
    assert row.retrieval_hit is True
    assert row.top_similarity == pytest.approx(0.9)
    assert row.error is None


def test_engine_unanswerable_correct_refusal() -> None:
    """A correct refusal on an unanswerable question scores EM=1, F1=1."""
    pipeline = FakePipeline({"Q?": _ok(RAG_REFUSAL_PHRASE, ["noise"], [0.3])})
    engine = EvaluationEngine(pipeline)

    report = engine.evaluate(
        [{"question": "Q?", "gold_answers": [], "is_impossible": True}]
    )

    row = report.per_question[0]
    assert row.expected_refusal is True
    assert row.is_refusal is True
    assert row.exact_match is True
    assert row.f1 == pytest.approx(1.0)
    assert row.retrieval_hit is None
    assert row.mrr is None


def test_engine_unanswerable_hallucination_penalised() -> None:
    """Hallucinating on an unanswerable question scores EM=0."""
    pipeline = FakePipeline({"Q?": _ok("the answer is 42", ["noise"], [0.3])})
    engine = EvaluationEngine(pipeline)

    report = engine.evaluate(
        [{"question": "Q?", "gold_answers": [], "is_impossible": True}]
    )

    row = report.per_question[0]
    assert row.expected_refusal is True
    assert row.is_refusal is False
    assert row.exact_match is False
    assert row.f1 == 0.0


def test_engine_answerable_wrong_refusal_penalised() -> None:
    """Refusing a question that has an answer scores EM=0."""
    pipeline = FakePipeline({"Q?": _ok(RAG_REFUSAL_PHRASE, ["noise"], [0.3])})
    engine = EvaluationEngine(pipeline)

    report = engine.evaluate(
        [{"question": "Q?", "gold_answers": ["Paris"], "is_impossible": False}]
    )

    row = report.per_question[0]
    assert row.expected_refusal is False
    assert row.is_refusal is True
    assert row.exact_match is False


def test_engine_retrieval_hit_and_mrr() -> None:
    """Hit@k and MRR reflect where the gold span appears."""
    pipeline = FakePipeline(
        {
            "hit": _ok("Paris", ["Paris is the capital of France."], [0.9]),
            "late": _ok("Paris", ["noise first", "Paris is the capital."], [0.2, 0.8]),
            "miss": _ok("Paris", ["Cats are furry."], [0.1]),
        }
    )
    engine = EvaluationEngine(pipeline)

    rows = {
        row.question: row
        for row in engine.evaluate(
            [
                {"question": "hit", "gold_answers": ["Paris"]},
                {"question": "late", "gold_answers": ["Paris"]},
                {"question": "miss", "gold_answers": ["Paris"]},
            ]
        ).per_question
    }

    assert rows["hit"].retrieval_hit is True
    assert rows["hit"].mrr == pytest.approx(1.0)
    assert rows["late"].retrieval_hit is True
    assert rows["late"].mrr == pytest.approx(0.5)
    assert rows["miss"].retrieval_hit is False
    assert rows["miss"].mrr == 0.0


def test_engine_pipeline_error_recorded_not_crash() -> None:
    """A result carrying an error is recorded and excluded from means."""
    pipeline = FakePipeline({"Q?": _ok("", ["x"], [0.1], error="boom")})
    engine = EvaluationEngine(pipeline)

    report = engine.evaluate([{"question": "Q?", "gold_answers": ["Paris"]}])

    assert report.num_errors == 1
    assert report.num_evaluated == 0
    assert report.per_question[0].error == "boom"
    assert report.per_question[0].exact_match is False


def test_engine_pipeline_exception_wrapped() -> None:
    """A raised exception is captured into the row instead of propagating."""
    pipeline = FakePipeline(raise_for={"Q?"})
    engine = EvaluationEngine(pipeline)

    report = engine.evaluate([{"question": "Q?", "gold_answers": ["Paris"]}])

    assert report.num_errors == 1
    assert report.per_question[0].error
    assert "boom" in report.per_question[0].error


def test_engine_include_unanswerable_false_skips() -> None:
    """Unanswerable questions are skipped (and never reach the pipeline)."""
    pipeline = FakePipeline({"A?": _ok("Paris", ["Paris."], [0.9])})
    engine = EvaluationEngine(
        pipeline, config=EvaluationConfig(include_unanswerable=False)
    )

    report = engine.evaluate(
        [
            {"question": "A?", "gold_answers": ["Paris"]},
            {"question": "U?", "gold_answers": [], "is_impossible": True},
        ]
    )

    assert report.num_questions == 1
    assert report.num_unanswerable == 0
    assert pipeline.calls == 1
    assert pipeline.questions == ["A?"]


def test_engine_max_questions_slices() -> None:
    """max_questions caps how many items are evaluated."""
    pipeline = FakePipeline(
        {f"Q{i}?": _ok("Paris", ["Paris."], [0.9]) for i in range(5)}
    )
    engine = EvaluationEngine(pipeline, config=EvaluationConfig(max_questions=2))

    report = engine.evaluate(
        [{"question": f"Q{i}?", "gold_answers": ["Paris"]} for i in range(5)]
    )

    assert report.num_questions == 2
    assert pipeline.calls == 2


def test_engine_top_k_forwarded() -> None:
    """The configured top_k is forwarded to the pipeline."""
    pipeline = FakePipeline({"Q?": _ok("Paris", ["Paris."], [0.9])})
    engine = EvaluationEngine(pipeline, config=EvaluationConfig(top_k=7))

    engine.evaluate([{"question": "Q?", "gold_answers": ["Paris"]}])

    assert pipeline.last_top_k == 7


def test_engine_aggregation_report_values() -> None:
    """Aggregate metrics match the hand-computed values of a 4-row scenario."""
    pipeline = FakePipeline(
        {
            "q1": _ok("Paris", ["Paris is the capital of France."], [0.9]),
            "q2": _ok("wrong answer", ["Cats are furry animals."], [0.2]),
            "q3": _ok(RAG_REFUSAL_PHRASE, ["noise"], [0.3]),
            "q4": _ok("the answer is 42", ["noise"], [0.3]),
        }
    )
    engine = EvaluationEngine(pipeline)

    report = engine.evaluate(
        [
            {"question": "q1", "gold_answers": ["Paris"], "is_impossible": False},
            {"question": "q2", "gold_answers": ["Berlin"], "is_impossible": False},
            {"question": "q3", "gold_answers": [], "is_impossible": True},
            {"question": "q4", "gold_answers": [], "is_impossible": True},
        ]
    )

    assert report.num_questions == 4
    assert report.num_evaluated == 4
    assert report.num_errors == 0
    assert report.num_answerable == 2
    assert report.num_unanswerable == 2
    assert report.exact_match == pytest.approx(0.5)
    assert report.f1 == pytest.approx(0.5)
    assert report.retrieval_hit_rate == pytest.approx(0.5)
    assert report.mrr == pytest.approx(0.5)
    assert report.num_refusals == 1
    assert report.num_correct_refusals == 1
    assert report.num_wrong_refusals == 0
    assert report.num_missed_refusals == 1
    assert report.refusal_precision == pytest.approx(1.0)
    assert report.refusal_recall == pytest.approx(0.5)
    assert report.answerable_refusal_rate == pytest.approx(0.0)
    assert report.avg_retrieval_time_sec == pytest.approx(0.1)
    assert report.avg_generation_time_sec == pytest.approx(0.2)
    assert report.avg_total_time_sec == pytest.approx(0.3)


def test_engine_empty_questions_report_zeros() -> None:
    """An empty question set yields a zeroed report without crashing."""
    engine = EvaluationEngine(FakePipeline())

    report = engine.evaluate([])

    assert report.num_questions == 0
    assert report.num_evaluated == 0
    assert report.exact_match == 0.0
    assert report.retrieval_hit_rate is None
    assert report.mrr is None
    assert report.per_question == ()


def test_engine_missing_question_raises() -> None:
    """An item without a question raises a validation error."""
    engine = EvaluationEngine(FakePipeline())

    with pytest.raises(EvaluationValidationError):
        engine.evaluate([{"gold_answers": ["Paris"]}])


def test_engine_extraction_dict_with_text_and_is_impossible() -> None:
    """SQuAD-style ``answers.text`` plus ``is_impossible`` are understood."""
    pipeline = FakePipeline(
        {"Q?": _ok("Paris", ["Paris is the capital of France."], [0.9])}
    )
    engine = EvaluationEngine(pipeline)

    report = engine.evaluate(
        [
            {
                "question": "Q?",
                "answers": {"text": ["Paris"]},
                "is_impossible": False,
            }
        ]
    )

    row = report.per_question[0]
    assert row.gold_answers == ("Paris",)
    assert row.expected_refusal is False
    assert row.exact_match is True
    assert row.retrieval_hit is True


def test_engine_extraction_attribute_object() -> None:
    """Plain attribute objects are accepted as evaluation items."""

    class _Item:
        question = "Q?"
        gold_answers = ["Paris"]
        is_impossible = False

    pipeline = FakePipeline(
        {"Q?": _ok("Paris", ["Paris is the capital of France."], [0.9])}
    )
    engine = EvaluationEngine(pipeline)

    report = engine.evaluate([_Item()])

    assert report.per_question[0].exact_match is True


def test_report_to_dict_and_summary_keys() -> None:
    """to_dict embeds per-question rows; summary omits them."""
    pipeline = FakePipeline(
        {"Q?": _ok("Paris", ["Paris is the capital of France."], [0.9])}
    )
    engine = EvaluationEngine(pipeline)
    report = engine.evaluate([{"question": "Q?", "gold_answers": ["Paris"]}])

    full = report.to_dict()
    assert "per_question" in full
    assert isinstance(full["per_question"], list)
    assert isinstance(full["per_question"][0]["gold_answers"], list)

    summary = report.summary()
    assert "per_question" not in summary
    for key in ("exact_match", "f1", "retrieval_hit_rate", "mrr"):
        assert key in summary
