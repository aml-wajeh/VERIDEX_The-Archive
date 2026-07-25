# Full RAG Pipeline using SQuAD v2 Dataset with Groq LLM and Streamlit

> **Title:** README
> **Description:** Project overview and setup guide for the Full RAG Pipeline project.
> **Responsibilities:**
>
> - Explain the project purpose.
> - Document setup and usage.
> - Track documentation growth across implementation phases.
>
> **Author:** Author Placeholder

![Python](https://img.shields.io/badge/Python-3.12-blue)
![UI](https://img.shields.io/badge/UI-Streamlit-red)
![Vector DB](https://img.shields.io/badge/Vector_DB-Chroma-green)
![LLM](https://img.shields.io/badge/LLM-Groq-orange)

## Project Overview

This project is a production-oriented educational Retrieval-Augmented Generation
system using the SQuAD v2 dataset, ChromaDB, Groq, and Streamlit.

Phase 1 initializes the professional project structure. The original workspace
did not contain implementation files, so no application logic was changed.

## Planned Features

- Dataset preprocessing
- Text chunking
- Embedding generation
- Chroma vector database
- Similarity retrieval
- Prompt engineering
- Groq LLM integration
- Streamlit application
- Evaluation and benchmarking
- Logging, tests, linting, and documentation

## Architecture

```mermaid
flowchart LR
    A[SQuAD v2 Dataset] --> B[Preprocessing]
    B --> C[Chunking]
    C --> D[Embeddings]
    D --> E[ChromaDB]
    F[User Question] --> G[Retriever]
    E --> G
    G --> H[Prompt Builder]
    H --> I[Groq LLM]
    I --> J[Streamlit UI]
```
