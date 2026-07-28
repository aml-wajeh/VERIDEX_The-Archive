"""RAG evaluation metrics and engine (Phase 9).

Title:
    Evaluation Module
Description:
    Scores a RAG pipeline against gold (question, answers) pairs using the
    SQuAD-style metrics that matter for a grounded system:

    * Exact Match (EM) and token-level F1 for the generated answer, computed
      with the official SQuAD normalisation (lower-case, strip punctuation,
      strip the articles ``a``/``an``/``the``).
    * Retrieval Hit@k and Mean Reciprocal Rank (MRR): does any retrieved chunk
      actually contain a gold answer string? (Computed for answerable
      questions only — an unanswerable question has no gold span to look for.)
    * Refusal behaviour: did the model correctly say "I could not find the
      answer" on unanswerable questions, and did it avoid refusing questions
      that do have an answer?

    A subtle but important detail: in official SQuAD v2 scoring the *correct*
    prediction for an unanswerable question is the empty string, but a
    grounded RAG model refuses with a fixed phrase instead. To reward the
    right behaviour, the engine maps a refusal to an empty *effective*
    prediction before scoring EM/F1. This makes a correct refusal score 1.0 on
    an unanswerable question, a hallucination score 0.0, and a wrong refusal
    on an answerable question score 0.0 — exactly the semantics we want.

    The engine is fully dependency-injected: it only requires an object with
    an ``answer_question(question, top_k=...)`` method (the ``RAGPipeline``
    from Phase 8 satisfies this), so the whole module is unit-testable with a
    fake pipeline and never touches Groq, the network, or a model.
Responsibilities:
    - Provide pure, reusable SQuAD-style metric functions.
    - Detect grounded refusals via the shared ``RAG_REFUSAL_PHRASE``.
    - Run the pipeline over a set of questions and aggregate the results.
    - Isolate pipeline errors so one bad question never aborts a benchmark.
Author:
    Aml
"""

from __future__ import annotations

import dataclasses
import re
import string
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.config import EvaluationConfig, resolve_evaluation_config
from src.prompt import RAG_REFUSAL_PHRASE

try:
    from src.logger import get_logger
except ImportError:  # pragma: no cover
    from logging import getLogger as get_logger


class EvaluationError(Exception):
    """Base exception for evaluation errors."""


class EvaluationValidationError(EvaluationError):
    """Raised when evaluation inputs or configuration are invalid."""


# ---------------------------------------------------------------------------
# SQuAD-style answer normalisation (matches the official SQuAD eval script)
# ---------------------------------------------------------------------------
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _remove_punctuation(text: str) -> str:
    """Strip ASCII punctuation from ``text``.

    Args:
        text: Raw text.

    Returns:
        Text without punctuation.
    """
    return text.translate(_PUNCT_TABLE)


def normalize_answer(text: str) -> str:
    """Normalise an answer the SQuAD way (lower, no punct, no articles).

    Args:
        text: Raw answer text. Non-strings yield an empty string.

    Returns:
        The normalised answer.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _remove_punctuation(text)
    text = _ARTICLES_RE.sub(" ", text)
    return " ".join(text.split())


def tokenize_answer(text: str) -> list[str]:
    """Tokenise a normalised answer by whitespace.

    Args:
        text: Raw answer text.

    Returns:
        List of normalised tokens.
    """
    return normalize_answer(text).split()


# ---------------------------------------------------------------------------
# Answer metrics
# ---------------------------------------------------------------------------
def compute_exact_match(
    prediction: str,
    gold_answers: Sequence[str],
) -> bool:
    """Compute exact match against one or more acceptable gold answers.

    An empty gold list means the question is unanswerable, in which case the
    prediction matches only when it normalises to the empty string.

    Args:
        prediction: Model prediction.
        gold_answers: Acceptable reference answers.

    Returns:
        ``True`` when the normalised prediction equals any normalised gold.
    """
    norm_pred = normalize_answer(prediction)
    if not gold_answers:
        return norm_pred == ""
    return any(norm_pred == normalize_answer(gold) for gold in gold_answers)


def _f1_single(prediction: str, gold: str) -> float:
    """Token-level F1 between a prediction and a single gold answer.

    Args:
        prediction: Model prediction.
        gold: One reference answer.

    Returns:
        The F1 score in ``[0, 1]``.
    """
    pred_tokens = tokenize_answer(prediction)
    gold_tokens = tokenize_answer(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_f1(prediction: str, gold_answers: Sequence[str]) -> float:
    """Compute the best token-level F1 over the acceptable gold answers.

    Args:
        prediction: Model prediction.
        gold_answers: Acceptable reference answers.

    Returns:
        The maximum F1 over the gold answers (``[0, 1]``).
    """
    if not gold_answers:
        return 1.0 if not tokenize_answer(prediction) else 0.0
    return max(_f1_single(prediction, gold) for gold in gold_answers)


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------
def _normalised_non_empty(values: Sequence[str]) -> list[str]:
    """Return the non-empty normalised forms of ``values``.

    Args:
        values: Raw strings.

    Returns:
        Normalised strings with empties dropped.
    """
    return [norm for norm in (normalize_answer(v) for v in values) if norm]


def compute_retrieval_hit(
    retrieved_texts: Sequence[str],
    gold_answers: Sequence[str],
) -> bool:
    """Check whether any retrieved chunk contains a gold answer span.

    Args:
        retrieved_texts: Retrieved chunk texts, in rank order.
        gold_answers: Acceptable reference answers.

    Returns:
        ``True`` when a normalised gold is a substring of some normalised
        chunk. ``False`` when there is no gold or no chunk to match.
    """
    norm_golds = _normalised_non_empty(gold_answers)
    if not norm_golds or not retrieved_texts:
        return False
    for page in retrieved_texts:
        norm_page = normalize_answer(page)
        if any(gold in norm_page for gold in norm_golds):
            return True
    return False


def compute_mrr(
    retrieved_texts: Sequence[str],
    gold_answers: Sequence[str],
) -> float:
    """Mean reciprocal rank for a single query (1/rank of first hit).

    Args:
        retrieved_texts: Retrieved chunk texts, in rank order.
        gold_answers: Acceptable reference answers.

    Returns:
        ``1 / rank`` of the first chunk containing a gold span, or ``0.0``
        when no chunk matches (or there is nothing to match).
    """
    norm_golds = _normalised_non_empty(gold_answers)
    if not norm_golds or not retrieved_texts:
        return 0.0
    for rank, page in enumerate(retrieved_texts, start=1):
        norm_page = normalize_answer(page)
        if any(gold in norm_page for gold in norm_golds):
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Refusal detection
# ---------------------------------------------------------------------------
def is_refusal(
    answer: str,
    refusal_phrase: str = RAG_REFUSAL_PHRASE,
) -> bool:
    """Detect whether an answer is a grounded refusal.

    The comparison is normalised, so minor casing or punctuation differences
    around the shared refusal phrase still count as a refusal. An empty or
    non-string answer is *not* treated as a refusal (it is the absence of an
    answer, not a deliberate refusal).

    Args:
        answer: Model answer.
        refusal_phrase: The canonical refusal phrase (defaults to the one
            shared with the prompt module).

    Returns:
        ``True`` when the normalised phrase appears in the normalised answer.
    """
    if not isinstance(answer, str) or not answer.strip():
        return False
    return normalize_answer(refusal_phrase) in normalize_answer(answer)


# ---------------------------------------------------------------------------
# Field extraction (tolerant of several common shapes)
# ---------------------------------------------------------------------------
_QUESTION_KEYS = ("question", "query", "q")
_GOLD_KEYS = ("gold_answers", "gold", "answers", "answer")
_IMPOSSIBLE_KEYS = ("is_impossible", "unanswerable", "expected_refusal")


def _first(item: Any, keys: Sequence[str]) -> Any | None:
    """Return the first present value from a mapping or object.

    Args:
        item: Mapping or object.
        keys: Candidate keys / attribute names.

    Returns:
        The first found value, or ``None``.
    """
    if isinstance(item, Mapping):
        for key in keys:
            if key in item:
                return item[key]
        return None
    for key in keys:
        if hasattr(item, key):
            return getattr(item, key)
    return None


def _extract_question(item: Any) -> str:
    """Extract the question string from an evaluation item.

    Args:
        item: Evaluation item.

    Returns:
        The question string (possibly empty).
    """
    value = _first(item, _QUESTION_KEYS)
    return str(value) if value is not None else ""


def _extract_gold_answers(item: Any) -> list[str]:
    """Extract acceptable gold answers, tolerating several shapes.

    Handles a plain string, a list of strings, a SQuAD-style
    ``{"text": [...]}`` mapping, and a list mixing strings and such mappings.

    Args:
        item: Evaluation item.

    Returns:
        A list of non-empty gold answer strings.
    """
    raw = _first(item, _GOLD_KEYS)
    if raw is None:
        return []
    if isinstance(raw, Mapping):
        texts = raw.get("text")
        if isinstance(texts, (list, tuple)):
            return [str(t) for t in texts if isinstance(t, str) and t.strip()]
        return []
    if isinstance(raw, str):
        return [raw] if raw.strip() else []
    if isinstance(raw, (list, tuple)):
        out: list[str] = []
        for element in raw:
            if isinstance(element, str) and element.strip():
                out.append(element)
            elif isinstance(element, Mapping):
                text = element.get("text")
                if isinstance(text, str) and text.strip():
                    out.append(text)
        return out
    return []


def _extract_is_impossible(item: Any, golds: Sequence[str]) -> bool:
    """Decide whether an item is an unanswerable question.

    An explicit boolean / flag wins; otherwise an item with no gold answers is
    treated as unanswerable.

    Args:
        item: Evaluation item.
        golds: Extracted gold answers.

    Returns:
        ``True`` when the question should be refused.
    """
    value = _first(item, _IMPOSSIBLE_KEYS)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    if isinstance(value, int):
        return bool(value)
    return len(golds) == 0


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class QuestionEvaluation:
    """Evaluation outcome for a single question.

    Attributes:
        question: The evaluated question.
        gold_answers: Acceptable reference answers (empty = unanswerable).
        predicted_answer: Raw model answer.
        expected_refusal: Whether the question should have been refused.
        is_refusal: Whether the model actually refused.
        exact_match: SQuAD exact match on the *effective* prediction.
        f1: SQuAD token F1 on the *effective* prediction.
        retrieval_hit: Whether a retrieved chunk contained a gold span
            (``None`` for unanswerable questions).
        mrr: Reciprocal rank of the first matching chunk (``None`` for
            unanswerable questions).
        top_similarity: Highest similarity score among retrieved chunks.
        retrieval_time_sec: Retrieval latency reported by the pipeline.
        generation_time_sec: Generation latency reported by the pipeline.
        total_time_sec: End-to-end latency reported by the pipeline.
        error: Error description if the pipeline failed, else ``None``.
    """

    question: str
    gold_answers: tuple[str, ...]
    predicted_answer: str
    expected_refusal: bool
    is_refusal: bool
    exact_match: bool
    f1: float
    retrieval_hit: bool | None
    mrr: float | None
    top_similarity: float | None
    retrieval_time_sec: float
    generation_time_sec: float
    total_time_sec: float
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-friendly dictionary.

        Returns:
            Dictionary representation (tuples become lists).
        """
        data: dict[str, Any] = {}
        for field_def in dataclasses.fields(self):
            value = getattr(self, field_def.name)
            data[field_def.name] = list(value) if isinstance(value, tuple) else value
        return data


@dataclass(frozen=True)
class EvaluationReport:
    """Aggregated evaluation report across many questions.

    All means and rates are computed over *successfully evaluated* questions
    (``error is None``); crashed questions are reported via ``num_errors`` and
    excluded from the averages so a single failure cannot silently skew a
    benchmark. Retrieval metrics and the answerable-refusal rate are computed
    over answerable questions only.

    Attributes:
        num_questions: Questions attempted (after skipping / capping).
        num_evaluated: Questions evaluated without a pipeline error.
        num_errors: Questions whose pipeline call failed.
        num_answerable: Evaluated answerable questions.
        num_unanswerable: Evaluated unanswerable questions.
        exact_match: Mean exact match over evaluated questions.
        f1: Mean token F1 over evaluated questions.
        retrieval_hit_rate: Fraction of answerable questions with a hit.
        mrr: Mean reciprocal rank over answerable questions.
        refusal_precision: Of all refusals, fraction on unanswerable questions.
        refusal_recall: Of unanswerable questions, fraction correctly refused.
        answerable_refusal_rate: Fraction of answerable questions wrongly
            refused (lower is better).
        num_refusals: Total refusals emitted on evaluated questions.
        num_correct_refusals: Refusals on unanswerable questions.
        num_wrong_refusals: Refusals on answerable questions.
        num_missed_refusals: Unanswerable questions not refused.
        avg_retrieval_time_sec: Mean retrieval latency.
        avg_generation_time_sec: Mean generation latency.
        avg_total_time_sec: Mean end-to-end latency.
        per_question: Per-question outcomes, in evaluation order.
    """

    num_questions: int
    num_evaluated: int
    num_errors: int
    num_answerable: int
    num_unanswerable: int
    exact_match: float
    f1: float
    retrieval_hit_rate: float | None
    mrr: float | None
    refusal_precision: float | None
    refusal_recall: float | None
    answerable_refusal_rate: float | None
    num_refusals: int
    num_correct_refusals: int
    num_wrong_refusals: int
    num_missed_refusals: int
    avg_retrieval_time_sec: float
    avg_generation_time_sec: float
    avg_total_time_sec: float
    per_question: tuple[QuestionEvaluation, ...]

    def to_dict(self, include_per_question: bool = True) -> dict[str, Any]:
        """Convert the report to a dictionary.

        Args:
            include_per_question: Whether to embed the per-question list.

        Returns:
            Dictionary representation.
        """
        data = self.summary()
        if include_per_question:
            data["per_question"] = [q.to_dict() for q in self.per_question]
        return data

    def summary(self) -> dict[str, Any]:
        """Return the aggregate metrics without the per-question rows.

        Returns:
            Dictionary of aggregate metrics.
        """
        return {
            field_def.name: getattr(self, field_def.name)
            for field_def in dataclasses.fields(self)
            if field_def.name != "per_question"
        }


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
def _mean(values: Iterable[float | bool]) -> float:
    """Return the rounded mean of ``values``, or 0.0 when empty.

    Args:
        values: Numeric (or boolean) values.

    Returns:
        The mean rounded to 4 decimals.
    """
    items = [float(v) for v in values]
    return round(sum(items) / len(items), 4) if items else 0.0


def _rate(numerator: int, denominator: int) -> float | None:
    """Return a rounded ratio, or ``None`` when the denominator is zero.

    Args:
        numerator: Count of successes.
        denominator: Count of trials.

    Returns:
        The ratio rounded to 4 decimals, or ``None``.
    """
    return round(numerator / denominator, 4) if denominator > 0 else None


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class EvaluationEngine:
    """Runs a RAG pipeline over gold questions and aggregates the metrics.

    The pipeline is injected and only needs an ``answer_question`` method that
    returns a mapping with at least ``answer`` and, ideally,
    ``retrieved_pages`` / ``similarities`` / timing / ``error`` keys (the
    ``RAGPipeline`` result dict from Phase 8 provides all of them).

    Attributes:
        config: Resolved evaluation configuration.
    """

    def __init__(
        self,
        pipeline: Any | None = None,
        config: EvaluationConfig | Any | None = None,
    ) -> None:
        """Initialise the engine.

        Args:
            pipeline: An object exposing ``answer_question(question, top_k=)``.
            config: Optional evaluation configuration.

        Raises:
            EvaluationValidationError: If no pipeline is supplied or the
                configuration is invalid.
        """
        if pipeline is None:
            raise EvaluationValidationError(
                "A pipeline with answer_question() is required to evaluate."
            )

        try:
            self._config = resolve_evaluation_config(config)
        except (TypeError, ValueError, AttributeError) as exc:
            raise EvaluationValidationError(
                "Invalid evaluation configuration."
            ) from exc

        self._pipeline = pipeline
        self._logger = get_logger("evaluation")

    @property
    def config(self) -> EvaluationConfig:
        """Return the active configuration.

        Returns:
            Active `EvaluationConfig`.
        """
        return self._config

    def evaluate(self, questions: Iterable[Any]) -> EvaluationReport:
        """Evaluate the pipeline over a sequence of gold questions.

        Args:
            questions: Iterable of evaluation items (mappings or objects)
                carrying a question, gold answers and, optionally, an
                ``is_impossible`` flag.

        Returns:
            An aggregated :class:`EvaluationReport`.

        Raises:
            EvaluationValidationError: If any item lacks a question.
        """
        items = list(questions)
        if self._config.max_questions is not None:
            items = items[: self._config.max_questions]

        self._logger.info(
            "Evaluating %d question(s) (top_k=%d, include_unanswerable=%s).",
            len(items),
            self._config.top_k,
            self._config.include_unanswerable,
        )

        rows: list[QuestionEvaluation] = []
        for item in items:
            row = self._evaluate_one(item)
            if row is not None:
                rows.append(row)

        report = self._aggregate(rows)
        self._logger.info(
            "Evaluation done: EM=%.4f F1=%.4f Hit@k=%s MRR=%s "
            "(evaluated=%d, errors=%d).",
            report.exact_match,
            report.f1,
            report.retrieval_hit_rate,
            report.mrr,
            report.num_evaluated,
            report.num_errors,
        )
        return report

    def _evaluate_one(self, item: Any) -> QuestionEvaluation | None:
        """Evaluate a single item, returning ``None`` when it is skipped.

        Args:
            item: Evaluation item.

        Returns:
            A :class:`QuestionEvaluation`, or ``None`` when the item is an
            unanswerable question and ``include_unanswerable`` is ``False``.

        Raises:
            EvaluationValidationError: If the item has no question.
        """
        question = _extract_question(item)
        if not question.strip():
            raise EvaluationValidationError(
                "Evaluation item is missing a non-empty question."
            )

        golds = tuple(_extract_gold_answers(item))
        expected_refusal = _extract_is_impossible(item, golds)

        if expected_refusal and not self._config.include_unanswerable:
            return None

        error: str | None = None
        prediction = ""
        retrieved: tuple[str, ...] = ()
        similarities: tuple[float, ...] = ()
        retrieval_time = 0.0
        generation_time = 0.0
        total_time = 0.0

        try:
            result = self._pipeline.answer_question(question, top_k=self._config.top_k)
            if not isinstance(result, Mapping):
                raise EvaluationError("pipeline.answer_question must return a mapping.")
            error = result.get("error")
            prediction = str(result.get("answer", "") or "")
            retrieved = tuple(
                str(page) for page in (result.get("retrieved_pages") or ())
            )
            similarities = tuple(
                float(score)
                for score in (result.get("similarities") or ())
                if score is not None
            )
            retrieval_time = float(result.get("retrieval_time_sec", 0.0) or 0.0)
            generation_time = float(result.get("generation_time_sec", 0.0) or 0.0)
            total_time = float(result.get("total_time_sec", 0.0) or 0.0)
        except EvaluationError:
            raise
        except Exception as exc:  # noqa: BLE001
            self._logger.exception(
                "Pipeline failed while evaluating question: %s", question
            )
            error = str(exc)

        if error is not None:
            refusal = False
            exact_match = False
            f1 = 0.0
            retrieval_hit: bool | None = None
            mrr: float | None = None
            top_similarity: float | None = None
        else:
            refusal = is_refusal(prediction)
            effective = "" if refusal else prediction
            exact_match = compute_exact_match(effective, golds)
            f1 = compute_f1(effective, golds)
            retrieval_hit = compute_retrieval_hit(retrieved, golds) if golds else None
            mrr = compute_mrr(retrieved, golds) if golds else None
            top_similarity = max(similarities) if similarities else None

        return QuestionEvaluation(
            question=question,
            gold_answers=golds,
            predicted_answer=prediction,
            expected_refusal=expected_refusal,
            is_refusal=refusal,
            exact_match=exact_match,
            f1=f1,
            retrieval_hit=retrieval_hit,
            mrr=mrr,
            top_similarity=top_similarity,
            retrieval_time_sec=retrieval_time,
            generation_time_sec=generation_time,
            total_time_sec=total_time,
            error=error,
        )

    def _aggregate(self, rows: list[QuestionEvaluation]) -> EvaluationReport:
        """Aggregate per-question rows into a report.

        Args:
            rows: Per-question outcomes (skipped items already removed).

        Returns:
            The aggregated :class:`EvaluationReport`.
        """
        valid = [row for row in rows if row.error is None]
        answerable = [row for row in valid if not row.expected_refusal]
        unanswerable = [row for row in valid if row.expected_refusal]

        num_refusals = sum(1 for row in valid if row.is_refusal)
        correct_refusals = sum(
            1 for row in valid if row.is_refusal and row.expected_refusal
        )
        wrong_refusals = sum(
            1 for row in valid if row.is_refusal and not row.expected_refusal
        )
        missed_refusals = sum(
            1 for row in valid if not row.is_refusal and row.expected_refusal
        )

        return EvaluationReport(
            num_questions=len(rows),
            num_evaluated=len(valid),
            num_errors=len(rows) - len(valid),
            num_answerable=len(answerable),
            num_unanswerable=len(unanswerable),
            exact_match=_mean(row.exact_match for row in valid),
            f1=_mean(row.f1 for row in valid),
            retrieval_hit_rate=_rate(
                sum(1 for row in answerable if row.retrieval_hit),
                len(answerable),
            ),
            mrr=(
                _mean(row.mrr for row in answerable if row.mrr is not None)
                if answerable
                else None
            ),
            refusal_precision=_rate(correct_refusals, num_refusals),
            refusal_recall=_rate(correct_refusals, len(unanswerable)),
            answerable_refusal_rate=_rate(wrong_refusals, len(answerable)),
            num_refusals=num_refusals,
            num_correct_refusals=correct_refusals,
            num_wrong_refusals=wrong_refusals,
            num_missed_refusals=missed_refusals,
            avg_retrieval_time_sec=_mean(row.retrieval_time_sec for row in valid),
            avg_generation_time_sec=_mean(row.generation_time_sec for row in valid),
            avg_total_time_sec=_mean(row.total_time_sec for row in valid),
            per_question=tuple(rows),
        )
