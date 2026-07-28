"""05 - Create Chroma store: persist embeddings in a cosine collection.

Production logic lives in ``src.vector_store`` (+ ``src.embeddings``).

Run from the project root::

    python 05_create_chroma_store.py
"""

from src.config import get_settings
from src.embeddings import EmbeddingGenerator
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
    """Encode the sample chunk, store it in Chroma, print the count."""
    try:
        settings = get_settings()
        records = EmbeddingGenerator(settings.embedding).encode_chunks(
            _TEXTS, show_progress=False
        )
        store = VectorStoreManager(settings.vector_store)
        store.connect()
        store.add_records(records, [item["text"] for item in _TEXTS])
        print("stored vectors:", store.count())
    except Exception as exc:
        print("Vector-store step needs the model/network:", exc)


if __name__ == "__main__":
    main()
