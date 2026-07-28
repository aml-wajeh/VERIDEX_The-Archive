"""07 - Prompting: build the grounded system + context prompt.

Production logic lives in ``src.prompt`` (and ``src.rag_pipeline`` for the
actual generation). This entry point is fully offline.

Run from the project root::

    python 07_prompting.py
"""

from src.prompt import RAG_SYSTEM_PROMPT, build_user_message, format_context


def main() -> None:
    """Print the grounded system prompt and a sample user message."""
    context = format_context(["Paris is the capital of France."])
    user = build_user_message("What is the capital of France?", context)
    print("=== SYSTEM ===")
    print(RAG_SYSTEM_PROMPT)
    print("=== USER ===")
    print(user)


if __name__ == "__main__":
    main()
