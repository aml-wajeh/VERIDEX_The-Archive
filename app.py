"""Streamlit application.

Title:
    Streamlit UI

Description:
    Provides the user interface for the RAG application.

Responsibilities:
    - Render the Streamlit UI.
    - Collect user input.
    - Display pipeline outputs.
    - Keep business logic inside src modules.

Author:
    Author Placeholder
"""

import streamlit as st


def main() -> None:
    """Render the Streamlit application.

    Args:
        None.

    Returns:
        None.
    """
    st.set_page_config(page_title="Full RAG Pipeline", page_icon="RAG", layout="wide")
    st.title("Full RAG Pipeline using SQuAD v2")
    st.info("Phase 1 scaffold is ready. Pipeline behavior will be added in later phases.")

    with st.sidebar:
        st.header("Project Status")
        st.success("Architecture initialized")
        st.caption("Groq, ChromaDB, retrieval, and evaluation modules are scaffolded.")

    question = st.text_input("Ask a question", placeholder="Example: What is retrieval augmented generation?")
    if question:
        st.warning("RAG execution is not implemented in Phase 1.")


if __name__ == "__main__":
    main()
