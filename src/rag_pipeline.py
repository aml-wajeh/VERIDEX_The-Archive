"""RAG pipeline orchestration.

Title:
    RAG Pipeline Module

Description:
    Coordinates retrieval, prompt construction, and LLM response generation.

Responsibilities:
    - Orchestrate the complete RAG workflow.
    - Keep UI code out of pipeline logic.
    - Coordinate retriever, prompt, vector store, embeddings, and LLM modules.

Author:
    Author Placeholder
"""

from dataclasses import dataclass


@dataclass
class RAGPipeline:
    """Pipeline placeholder for RAG orchestration.

    This class intentionally contains no production behavior in Phase 1 because
    the current workspace has no existing implementation to preserve.
    """

    def answer_question(self, question: str) -> str:
        """Answer a user question using the RAG pipeline.

        Args:
            question: User question.

        Returns:
            Generated answer.

        Raises:
            NotImplementedError: Until Phase 3 implements pipeline orchestration.
        """
        raise NotImplementedError("RAG orchestration will be implemented in Phase 3.")
