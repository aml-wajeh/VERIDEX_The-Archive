"""06 - Retrieve context: similarity search over the vector store.

Production logic lives in ``src.retriever`` (+ embeddings + vector store).

Run from the project root::

    python 06_retrieve_context.py
"""

from src.config import get_settings
from src.embeddings import EmbeddingGenerator
from src.retriever import Retriever
from src.vector_store import VectorStoreManager

_TEXTS: list[dict] = [
    {
        "chunk_id": "c1",
        "document_id": "d1",
        "text": "Paris is the capital of France.",
        "metadata": {},
    }
]


def main() -> None:
    """Store the sample chunk, then retrieve it for a query."""
    try:
        settings = get_settings()
        embedder = EmbeddingGenerator(settings.embedding)
        records = embedder.encode_chunks(_TEXTS, show_progress=False)
        store = VectorStoreManager(settings.vector_store)
        store.connect()
        store.add_records(records, [item["text"] for item in _TEXTS])
        retriever = Retriever(vector_store=store, embedding_generator=embedder)
        result = retriever.retrieve("capital of France?", top_k=3)
        for chunk in result:
            print(chunk.rank, round(chunk.similarity, 3), chunk.text)
    except Exception as exc:
        print("Retrieval step needs the model/network:", exc)


if __name__ == "__main__":
    main()
