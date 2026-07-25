"""RAG evaluation.

Title:
    Evaluation Module

Description:
    Provides evaluation metric and benchmarking placeholders for the RAG
    pipeline.

Responsibilities:
    - Define evaluation interfaces.
    - Support future benchmark runs.
    - Keep evaluation independent from the Streamlit UI.

Author:
    Author Placeholder
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationResult:
    """Container for evaluation results.

    Args:
        metric_name: Name of the evaluation metric.
        score: Numeric metric score.
    """

    metric_name: str
    score: float


def evaluate_answer(predicted_answer: str, reference_answer: str) -> EvaluationResult:
    """Evaluate a predicted answer against a reference answer.

    Args:
        predicted_answer: Model-generated answer.
        reference_answer: Ground-truth answer.

    Returns:
        Evaluation result with metric name and score.

    Raises:
        NotImplementedError: Until Phase 9 implements evaluation metrics.
    """
    raise NotImplementedError("Evaluation metrics will be implemented in Phase 9.")
